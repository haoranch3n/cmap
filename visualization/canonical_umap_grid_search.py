#!/usr/bin/env python3
"""Grid search UMAP on pooled per-cell canonical QC features (both datasets).

Each hyperparameter combination fits UMAP on the full ``(N, F)`` matrix where
each row is one cell and each column is a scaled canonical feature. Writes
per-combo coordinate CSV plus a small JSON score file; ``--merge-scores``
aggregates JSONs into one CSV; ``--publish-best`` picks the best combo and
writes the Napari master table.

Default grid (12 combos): ``n_neighbors`` in ``{5,15,30}``, ``min_dist`` in
``{0.0,0.1}``, ``metric`` in ``{cosine,euclidean}`` (``3×2×2``).

Intrinsic score: ``sklearn.manifold.trustworthiness`` at ``n_neighbors=5`` and
``15``.

Feature columns (StandardScaler-normalised before UMAP):
    volume, mean/median/std/total/cv/pct95 per channel (642,488,560),
    corr_642_488, corr_642_560, corr_488_560, min_channel_mean,
    mean_pairwise_corr.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import OUTPUT_DIR
except ModuleNotFoundError:  # pragma: no cover
    from config import OUTPUT_DIR  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    "volume",
    "mean_642", "median_642", "std_642", "total_642", "cv_642", "pct95_642",
    "mean_488", "median_488", "std_488", "total_488", "cv_488", "pct95_488",
    "mean_560", "median_560", "std_560", "total_560", "cv_560", "pct95_560",
    "corr_642_488", "corr_642_560", "corr_488_560",
    "min_channel_mean", "mean_pairwise_corr",
]

QC_CSV_NAME = "qc_features.csv"

# (dataset_relpath under output/, cell_qc_dirname)
DATASET_SPECS: dict[str, tuple[tuple[str, str], ...]] = {
    "bg_sigma_488560_shape": (
        ("4_18_25", "cell_qc_bg_sigma_488560_shape"),
        ("4_24_25_CGN_6_10_2", "cell_qc_bg_sigma_488560_shape"),
    ),
    # No-deconv filtered_642 runs: per-sample ``cell_qc/qc_features.csv`` under
    # e.g. ``output_no_deconv/4_18_25/<sample>/``. Use ``--output-root`` pointing
    # at ``output_no_deconv``.
    "no_deconv_filtered_642": (
        ("4_18_25", "cell_qc"),
        ("4_24_25_CGN_6_10_2", "cell_qc"),
    ),
    # No-deconv union_488_560 + bg_sigma:3 runs. Use ``--output-root`` pointing
    # at ``output_no_deconv``.
    "no_deconv_bg_sigma_488560_shape": (
        ("4_18_25", "cell_qc_bg_sigma_488560_shape"),
        ("4_24_25_CGN_6_10_2", "cell_qc_bg_sigma_488560_shape"),
    ),
}

DEFAULT_N_NEIGHBORS = (5, 15, 30)
DEFAULT_MIN_DIST = (0.0, 0.1)
DEFAULT_METRICS = ("cosine", "euclidean")


def combo_slug(n_neighbors: int, min_dist: float, metric: str) -> str:
    md = "0" if min_dist == 0.0 else str(min_dist).replace(".", "p")
    return f"nn{n_neighbors}_md{md}_{metric}"


def _trustworthiness_pair(X: np.ndarray, embedding: np.ndarray) -> tuple[float, float]:
    from sklearn.manifold import trustworthiness
    n = X.shape[0]
    k5 = min(5, max(2, n - 1))
    k15 = min(15, max(2, n - 1))
    return float(trustworthiness(X, embedding, n_neighbors=k5)), \
           float(trustworthiness(X, embedding, n_neighbors=k15))


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, metric: str, random_state: int) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("UMAP is required. Install umap-learn.") from exc
    nn = int(min(int(n_neighbors), max(2, X.shape[0] - 1)))
    model = umap.UMAP(
        n_components=2, n_neighbors=nn, min_dist=min_dist,
        metric=metric, random_state=random_state,
    )
    return model.fit_transform(X)


def discover_and_pool(
    output_root: Path,
    dataset_specs: tuple[tuple[str, str], ...],
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Pool qc_features.csv across all samples; return (X_scaled, meta_df, feature_cols_used)."""
    frames: list[pd.DataFrame] = []
    for dataset_relpath, qc_dir in dataset_specs:
        ds_root = output_root / dataset_relpath
        if not ds_root.is_dir():
            logger.warning("Dataset root missing, skipping: %s", ds_root)
            continue
        for csv_path in sorted(ds_root.rglob(f"{qc_dir}/{QC_CSV_NAME}")):
            sample_name = csv_path.parent.parent.name
            df = pd.read_csv(csv_path)
            df = df.copy()
            df.insert(0, "sample", sample_name)
            df.insert(1, "dataset", dataset_relpath)
            bg_csv = csv_path.parent / "bg_stats.csv"
            if bg_csv.is_file():
                bg_df = pd.read_csv(bg_csv)
                if len(bg_df) == 1:
                    for col in bg_df.columns:
                        df[col] = bg_df.iloc[0][col]
            frames.append(df)
            print(f"  {dataset_relpath}/{sample_name}: {len(df)} cells")

    if not frames:
        raise RuntimeError(f"No {QC_CSV_NAME} found for {dataset_specs}")

    pooled = pd.concat(frames, ignore_index=True)
    print(f"\nPooled: {len(pooled)} cells from {len(frames)} samples")

    feat_cols = [c for c in FEATURE_COLS if c in pooled.columns]
    missing = [c for c in FEATURE_COLS if c not in pooled.columns]
    if missing:
        logger.warning("Feature columns absent in CSV, skipping: %s", missing)

    X_raw = pooled[feat_cols].values.astype(np.float64)
    nan_rows = np.isnan(X_raw).any(axis=1)
    if nan_rows.any():
        logger.warning("Dropping %d rows with NaN in features", int(nan_rows.sum()))
        pooled = pooled.loc[~nan_rows].reset_index(drop=True)
        X_raw = X_raw[~nan_rows]

    X_scaled = StandardScaler().fit_transform(X_raw).astype(np.float32)
    print(f"Feature matrix: {X_scaled.shape}  ({len(feat_cols)} features after scaling)")
    return X_scaled, pooled, feat_cols


def run_single_combo(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    outdir: Path,
    variant: str,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    slug = combo_slug(n_neighbors, min_dist, metric)
    t0 = time.time()
    coords = run_umap(X, n_neighbors, min_dist, metric, random_state)
    elapsed = time.time() - t0
    tw5, tw15 = _trustworthiness_pair(X, coords)

    out_meta = meta.copy()
    out_meta["umap_canonical_1"] = coords[:, 0]
    out_meta["umap_canonical_2"] = coords[:, 1]
    out_meta["umap_n_neighbors"] = int(min(n_neighbors, max(2, X.shape[0] - 1)))
    out_meta["umap_min_dist"] = min_dist
    out_meta["umap_metric"] = metric

    out_csv = outdir / f"canonical_umap_coords_{variant}_{slug}.csv"
    out_meta.to_csv(out_csv, index=False)

    score = {
        "combo_slug": slug,
        "variant": variant,
        "n_neighbors_requested": n_neighbors,
        "n_neighbors_used": int(min(n_neighbors, max(2, X.shape[0] - 1))),
        "min_dist": min_dist,
        "metric": metric,
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "elapsed_sec": round(elapsed, 3),
        "trustworthiness_k5": tw5,
        "trustworthiness_k15": tw15,
        "coords_csv": str(out_csv.name),
    }
    score_path = outdir / f"canonical_umap_score_{variant}_{slug}.json"
    score_path.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}  ({len(out_meta)} rows)")
    print(f"Wrote {score_path}  tw_k5={tw5:.4f} tw_k15={tw15:.4f}  elapsed={elapsed:.1f}s")
    return score_path


def merge_scores(outdir: Path, variant: str) -> int:
    rows: list[dict] = []
    for p in sorted(outdir.glob(f"canonical_umap_score_{variant}_*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        print(f"ERROR: no canonical_umap_score_{variant}_*.json under {outdir}")
        return 1
    df = pd.DataFrame(rows)
    out = outdir / f"canonical_umap_sweep_scores_{variant}.csv"
    df.to_csv(out, index=False)
    print(f"Merged {len(rows)} combos -> {out}")
    return 0


def publish_best(
    output_root: Path,
    *,
    variant: str,
    sweep_dir: Path,
    dataset_specs: tuple[tuple[str, str], ...],
    score_column: str = "trustworthiness_k15",
) -> int:
    merged_path = sweep_dir / f"canonical_umap_sweep_scores_{variant}.csv"
    if not merged_path.is_file():
        rc = merge_scores(sweep_dir, variant)
        if rc != 0:
            return rc

    scores = pd.read_csv(merged_path)
    if score_column not in scores.columns:
        print(f"ERROR: column {score_column!r} missing from {merged_path}")
        return 1

    best_idx = int(scores[score_column].astype(float).idxmax())
    best = scores.iloc[best_idx]
    coords_name = str(best["coords_csv"])
    coords_path = sweep_dir / coords_name
    if not coords_path.is_file():
        print(f"ERROR: coords file missing: {coords_path}")
        return 1

    print(
        f"Best combo ({score_column}={best[score_column]:.4f}): "
        f"{best.get('combo_slug', coords_name)}"
    )

    X_scaled, meta, _ = discover_and_pool(output_root, dataset_specs)
    coords = pd.read_csv(coords_path)
    key_cols = ["dataset", "sample", "cell_id"]
    use = key_cols + ["umap_canonical_1", "umap_canonical_2"]
    missing_cols = [c for c in use if c not in coords.columns]
    if missing_cols:
        print(f"ERROR: coords CSV missing columns {missing_cols}")
        return 1

    merged = meta.merge(coords[use], on=key_cols, how="left", validate="one_to_one")
    if merged["umap_canonical_1"].isna().any():
        n_bad = int(merged["umap_canonical_1"].isna().sum())
        print(f"ERROR: {n_bad} rows did not match coords CSV on {key_cols}")
        return 1

    for c in ("umap_n_neighbors", "umap_min_dist", "umap_metric"):
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    merged["umap_sweep_best_slug"] = str(best.get("combo_slug", ""))
    merged["umap_sweep_score_column"] = score_column
    merged["umap_sweep_score_value"] = float(best[score_column])

    out_dir = output_root / "cell_qc_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"canonical_embedding_{variant}_all.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Wrote master CSV: {out_csv}  ({len(merged)} rows, {len(merged.columns)} cols)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument(
        "--variant",
        choices=list(DATASET_SPECS.keys()),
        default="bg_sigma_488560_shape",
        help="Which cell_qc_* dirs to pool (default: bg_sigma_488560_shape).",
    )
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--score-column", type=str, default="trustworthiness_k15")
    ap.add_argument(
        "--run-all-local",
        action="store_true",
        help="Run full default grid sequentially in one process.",
    )
    ap.add_argument(
        "--merge-scores",
        action="store_true",
        help="Only merge score JSONs into sweep_scores CSV.",
    )
    ap.add_argument(
        "--publish-best",
        action="store_true",
        help="Pick best combo; write canonical_embedding_<variant>_all.csv to cell_qc_all/.",
    )
    ap.add_argument("--n-neighbors", type=int, default=None)
    ap.add_argument("--min-dist", type=float, default=None)
    ap.add_argument("--metric", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_root = args.output_root or Path(OUTPUT_DIR)
    variant = args.variant
    dataset_specs = DATASET_SPECS[variant]
    sweep_dir = output_root / "cell_qc_all" / f"umap_canonical_{variant}_sweep"
    print(f"Variant: {variant}  sweep_dir={sweep_dir}")

    if args.merge_scores:
        return merge_scores(sweep_dir, variant)

    if args.publish_best:
        return publish_best(
            output_root,
            variant=variant,
            sweep_dir=sweep_dir,
            dataset_specs=dataset_specs,
            score_column=args.score_column,
        )

    X, meta, _ = discover_and_pool(output_root, dataset_specs)

    if args.run_all_local:
        for nn in DEFAULT_N_NEIGHBORS:
            for md in DEFAULT_MIN_DIST:
                for metric in DEFAULT_METRICS:
                    run_single_combo(
                        X, meta,
                        n_neighbors=nn, min_dist=md, metric=metric,
                        random_state=args.random_state,
                        outdir=sweep_dir, variant=variant,
                    )
        return merge_scores(sweep_dir, variant)

    if args.n_neighbors is None or args.min_dist is None or args.metric is None:
        print(
            "ERROR: pass --n-neighbors, --min-dist, and --metric for a single combo, "
            "or use --run-all-local, or --merge-scores / --publish-best."
        )
        return 1

    run_single_combo(
        X, meta,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric,
        random_state=args.random_state,
        outdir=sweep_dir, variant=variant,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
