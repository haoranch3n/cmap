#!/usr/bin/env python3
"""Grid search UMAP on pooled per-cell DINOv2 embeddings (4_18 + 4_24 combined).

Each hyperparameter combination fits UMAP on the full ``(N, D)`` matrix where
each row is one cell. Writes per-combo coordinate CSV plus a small JSON score
file for LSF parallel jobs; ``--merge-scores`` aggregates JSONs into one CSV.

Default grid (12 combos): ``n_neighbors`` in ``{5,15,30}``, ``min_dist`` in
``{0.0,0.1}``, ``metric`` in ``{cosine,euclidean}`` (``3×2×2``).

Intrinsic scores: ``sklearn.manifold.trustworthiness`` at ``n_neighbors=5`` and
``15`` (capped by ``N-1``) comparing high-D ``X`` to 2D embedding (euclidean in
both spaces for the metric API).
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import OUTPUT_DIR
except ModuleNotFoundError:  # pragma: no cover
    from config import OUTPUT_DIR  # type: ignore[no-redef]

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS
from visualization.dinov2_pool import (
    discover_embeddings_multi_dataset,
    load_pool_from_triplets,
)

logger = logging.getLogger(__name__)

# (dataset_relpath under output/, cell_qc_dirname)
DATASET_SPECS_FILTERED_642: tuple[tuple[str, str], ...] = (
    ("4_18_25", "cell_qc_filtered"),
    ("4_24_25_CGN_6_10_2", "cell_qc_filtered"),
)
DATASET_SPECS_UNION_488_560: tuple[tuple[str, str], ...] = (
    ("4_18_25", "cell_qc_union_488_560"),
    ("4_24_25_CGN_6_10_2", "cell_qc_union_488_560"),
)
DATASET_SPECS_BY_VARIANT: dict[str, tuple[tuple[str, str], ...]] = {
    "filtered_642": DATASET_SPECS_FILTERED_642,
    "union_488_560": DATASET_SPECS_UNION_488_560,
}
DEFAULT_DATASET_SPECS: tuple[tuple[str, str], ...] = DATASET_SPECS_FILTERED_642


def sweep_dirname_for_variant(variant: str) -> str:
    """Per-variant sweep folder under ``cell_qc_all/`` (642 keeps legacy name)."""
    if variant == "filtered_642":
        return "umap_dinov2_sweep"
    return f"umap_dinov2_{variant}_sweep"


def master_filenames_for_variant(variant: str) -> tuple[str, str]:
    """Per-variant pooled master CSV/NPY (matches dinov2_visualize.py)."""
    if variant == "filtered_642":
        return MASTER_DINOV2_CSV, MASTER_DINOV2_NPY
    return (
        f"dinov2_embedding_{variant}_all.csv",
        f"dinov2_embeddings_{variant}_all.npy",
    )

DEFAULT_N_NEIGHBORS = (5, 15, 30)
DEFAULT_MIN_DIST = (0.0, 0.1)
DEFAULT_METRICS = ("cosine", "euclidean")


def combo_slug(n_neighbors: int, min_dist: float, metric: str) -> str:
    md = "0" if min_dist == 0.0 else str(min_dist).replace(".", "p")
    return f"nn{n_neighbors}_md{md}_{metric}"


def _trustworthiness_pair(
    X: np.ndarray,
    embedding: np.ndarray,
) -> tuple[float, float]:
    from sklearn.manifold import trustworthiness

    n = X.shape[0]
    k5 = min(5, max(2, n - 1))
    k15 = min(15, max(2, n - 1))
    t5 = float(trustworthiness(X, embedding, n_neighbors=k5))
    t15 = float(trustworthiness(X, embedding, n_neighbors=k15))
    return t5, t15


def run_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("UMAP is required. Install umap-learn.") from exc

    nn = int(min(int(n_neighbors), max(2, X.shape[0] - 1)))
    model = umap.UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return model.fit_transform(X)


def load_combined_matrix(
    output_root: Path,
    dataset_specs: tuple[tuple[str, str], ...],
) -> tuple[np.ndarray, pd.DataFrame]:
    triplets = discover_embeddings_multi_dataset(output_root, dataset_specs)
    if not triplets:
        raise RuntimeError(f"No embeddings found under {output_root} for {dataset_specs}")
    return load_pool_from_triplets(list(triplets))


def run_single_combo(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    outdir: Path,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    slug = combo_slug(n_neighbors, min_dist, metric)
    t0 = time.time()
    coords = run_umap(X, n_neighbors, min_dist, metric, random_state)
    elapsed = time.time() - t0
    tw5, tw15 = _trustworthiness_pair(X, coords)

    out_csv = outdir / f"dinov2_umap_coords_{slug}.csv"
    out_meta = meta.copy()
    out_meta["umap_1"] = coords[:, 0]
    out_meta["umap_2"] = coords[:, 1]
    out_meta["umap_n_neighbors"] = int(min(n_neighbors, max(2, X.shape[0] - 1)))
    out_meta["umap_min_dist"] = min_dist
    out_meta["umap_metric"] = metric
    out_meta.to_csv(out_csv, index=False)

    score = {
        "combo_slug": slug,
        "n_neighbors_requested": n_neighbors,
        "n_neighbors_used": int(min(n_neighbors, max(2, X.shape[0] - 1))),
        "min_dist": min_dist,
        "metric": metric,
        "n_cells": int(X.shape[0]),
        "embed_dim": int(X.shape[1]),
        "elapsed_sec": round(elapsed, 3),
        "trustworthiness_k5": tw5,
        "trustworthiness_k15": tw15,
        "coords_csv": str(out_csv.name),
    }
    score_path = outdir / f"dinov2_umap_score_{slug}.json"
    score_path.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}  ({len(out_meta)} rows)")
    print(f"Wrote {score_path}  trustworthiness_k5={tw5:.4f} k15={tw15:.4f}")
    return score_path


def merge_scores(outdir: Path, merged_name: str = "dinov2_umap_sweep_scores.csv") -> int:
    rows: list[dict] = []
    for p in sorted(outdir.glob("dinov2_umap_score_*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        print(f"ERROR: no dinov2_umap_score_*.json under {outdir}")
        return 1
    df = pd.DataFrame(rows)
    out = outdir / merged_name
    df.to_csv(out, index=False)
    print(f"Merged {len(rows)} rows -> {out}")
    return 0


MASTER_DINOV2_CSV = "dinov2_embedding_all.csv"
MASTER_DINOV2_NPY = "dinov2_embeddings_all.npy"


def publish_best_umap(
    output_root: Path,
    *,
    sweep_dir: Path | None = None,
    dataset_specs: tuple[tuple[str, str], ...] = DEFAULT_DATASET_SPECS,
    score_column: str = "trustworthiness_k15",
    master_csv_name: str = MASTER_DINOV2_CSV,
    master_npy_name: str = MASTER_DINOV2_NPY,
    sweep_dirname: str = "umap_dinov2_sweep",
) -> int:
    """Merge sweep scores (if needed), pick best combo by *score_column*, write Napari master."""
    sweep_dir = sweep_dir or (output_root / "cell_qc_all" / sweep_dirname)
    merged_path = sweep_dir / "dinov2_umap_sweep_scores.csv"
    if not merged_path.is_file():
        rc = merge_scores(sweep_dir)
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
        f"Best combo by {score_column}: {best.get('combo_slug', coords_name)} "
        f"({score_column}={best[score_column]})"
    )

    X, meta = load_combined_matrix(output_root, dataset_specs)
    coords = pd.read_csv(coords_path)
    key_cols = ["sample", "cell_id"]
    if "dataset" in meta.columns and "dataset" in coords.columns:
        key_cols = ["dataset", "sample", "cell_id"]
    use = key_cols + [c for c in ("umap_1", "umap_2") if c in coords.columns]
    missing = [c for c in use if c not in coords.columns]
    if missing:
        print(f"ERROR: coords CSV missing columns {missing}")
        return 1
    merged = meta.merge(coords[use], on=key_cols, how="left", validate="one_to_one")
    if merged["umap_1"].isna().any():
        n_bad = int(merged["umap_1"].isna().sum())
        print(f"ERROR: {n_bad} rows did not match coords CSV join keys {key_cols}")
        return 1
    merged = merged.rename(columns={"umap_1": "umap_dinov2_1", "umap_2": "umap_dinov2_2"})
    for c in ("umap_n_neighbors", "umap_min_dist", "umap_metric"):
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    merged["umap_sweep_best_slug"] = str(best.get("combo_slug", ""))
    merged["umap_sweep_score_column"] = score_column
    merged["umap_sweep_score_value"] = float(best[score_column])

    out_dir = output_root / "cell_qc_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / master_csv_name
    out_npy = out_dir / master_npy_name
    merged.to_csv(out_csv, index=False)
    np.save(str(out_npy), X.astype(np.float32, copy=False))
    print(f"Wrote {out_csv}  ({len(merged)} rows, {len(merged.columns)} cols)")
    print(f"Wrote {out_npy}  shape={X.shape}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=None, help="Pipeline output root (default OUTPUT_DIR)")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Sweep output directory (default: <output-root>/cell_qc_all/umap_dinov2_sweep)",
    )
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--merge-scores",
        action="store_true",
        help="Only merge dinov2_umap_score_*.json under --outdir into dinov2_umap_sweep_scores.csv.",
    )
    ap.add_argument(
        "--publish-best",
        action="store_true",
        help="Merge scores if needed; pick max trustworthiness_k15; write "
        "cell_qc_all/dinov2_embedding_all.csv + dinov2_embeddings_all.npy.",
    )
    ap.add_argument(
        "--score-column",
        type=str,
        default="trustworthiness_k15",
        help="Column in dinov2_umap_sweep_scores.csv to maximize (default: trustworthiness_k15).",
    )
    ap.add_argument("--n-neighbors", type=int, default=None, help="Single combo (with --min-dist, --metric)")
    ap.add_argument("--min-dist", type=float, default=None)
    ap.add_argument("--metric", type=str, default=None)
    ap.add_argument(
        "--run-all-local",
        action="store_true",
        help="Run full default grid sequentially in one process (no LSF).",
    )
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            f"Mask variant; selects DATASET_SPECS, sweep folder, and master "
            f"CSV/NPY filenames (default: {DEFAULT_VARIANT})."
        ),
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_root = args.output_root or Path(OUTPUT_DIR)
    dataset_specs = DATASET_SPECS_BY_VARIANT[args.variant]
    sweep_dirname = sweep_dirname_for_variant(args.variant)
    master_csv_name, master_npy_name = master_filenames_for_variant(args.variant)
    outdir = args.outdir or (output_root / "cell_qc_all" / sweep_dirname)
    print(
        f"Variant: {args.variant}  sweep_dir={outdir}  "
        f"master_csv={master_csv_name}  master_npy={master_npy_name}"
    )

    if args.merge_scores:
        return merge_scores(outdir)

    if args.publish_best:
        return publish_best_umap(
            output_root,
            sweep_dir=outdir,
            dataset_specs=dataset_specs,
            score_column=args.score_column,
            master_csv_name=master_csv_name,
            master_npy_name=master_npy_name,
            sweep_dirname=sweep_dirname,
        )

    if args.run_all_local:
        try:
            X, meta = load_combined_matrix(output_root, dataset_specs)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1
        for nn in DEFAULT_N_NEIGHBORS:
            for md in DEFAULT_MIN_DIST:
                for metric in DEFAULT_METRICS:
                    run_single_combo(
                        X,
                        meta,
                        n_neighbors=nn,
                        min_dist=md,
                        metric=metric,
                        random_state=args.random_state,
                        outdir=outdir,
                    )
        return merge_scores(outdir)

    if args.n_neighbors is None or args.min_dist is None or args.metric is None:
        print(
            "ERROR: pass --n-neighbors, --min-dist, and --metric for a single combo, "
            "or use --run-all-local, or --merge-scores."
        )
        return 1

    try:
        X, meta = load_combined_matrix(output_root, dataset_specs)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    run_single_combo(
        X,
        meta,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        outdir=outdir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
