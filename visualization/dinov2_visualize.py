#!/usr/bin/env python3
"""Pool per-sample DINOv2 embeddings and compute a UMAP projection.

Walks the configured output root for ``<cell-qc-dir>/dinov2_embeddings.npy`` plus
matching CSV files, **concatenates** them (each row remains **one cell**; there
is no averaging across cells), optionally filters to QC-passing cells, and
writes a master table under ``cell_qc_all/`` for the napari plugin.

Defaults match ``napari-plugin/scripts/compute_filtered_umap.py`` so the new
DINOv2 columns sit alongside the classical ``umap_pss_*`` ones:
``n_neighbors=5, min_dist=0.0, metric="cosine"``.
"""
from __future__ import annotations

import argparse
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

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files
from visualization.dinov2_pool import (
    EMB_NPY_NAME,
    apply_qc_filter,
    discover_embeddings_under_qc_dir,
    load_pool_from_triplets,
)

MASTER_DIR_NAME = "cell_qc_all"
MASTER_NPY_NAME = "dinov2_embeddings_all.npy"
MASTER_CSV_NAME = "dinov2_embedding_all.csv"


def master_filenames_for_variant(variant: str) -> tuple[str, str]:
    """Return (npy_name, csv_name) for the pooled master files of ``variant``.

    The 642 default keeps the legacy filenames so existing pooled outputs and
    napari paths do not change.  Other variants land under
    ``dinov2_embeddings_<variant>_all.npy`` / ``dinov2_embedding_<variant>_all.csv``.
    """
    if variant == "filtered_642":
        return MASTER_NPY_NAME, MASTER_CSV_NAME
    return (
        f"dinov2_embeddings_{variant}_all.npy",
        f"dinov2_embedding_{variant}_all.csv",
    )

UMAP_DEFAULTS = {"n_neighbors": 5, "min_dist": 0.0, "metric": "cosine"}


def _run_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("UMAP is required. Install umap-learn.") from exc

    n_neighbors = int(min(n_neighbors, max(2, X.shape[0] - 1)))
    print(
        f"Running UMAP (n={X.shape[0]}, n_neighbors={n_neighbors}, "
        f"min_dist={min_dist}, metric={metric})..."
    )
    t0 = time.time()
    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = model.fit_transform(X)
    print(f"  UMAP done in {time.time() - t0:.1f} s")
    return coords


def run(
    base: Path,
    *,
    outdir: Path | None = None,
    filter_col: str | None = None,
    suffix: str | None = None,
    n_neighbors: int = UMAP_DEFAULTS["n_neighbors"],
    min_dist: float = UMAP_DEFAULTS["min_dist"],
    metric: str = UMAP_DEFAULTS["metric"],
    random_state: int = 42,
    cell_qc_dirname: str = "cell_qc",
    master_npy_name: str = MASTER_NPY_NAME,
    master_csv_name: str = MASTER_CSV_NAME,
) -> int:
    print(f"Scanning: {base}  (cell_qc_dir={cell_qc_dirname})\n")
    triplets = discover_embeddings_under_qc_dir(base, cell_qc_dirname)
    if not triplets:
        print(f"ERROR: no {EMB_NPY_NAME} found under {base}/**/{cell_qc_dirname}/")
        return 1

    pooled, pooled_df = load_pool_from_triplets(triplets)
    if filter_col:
        pooled, pooled_df = apply_qc_filter(
            pooled,
            pooled_df,
            base,
            filter_col,
            qc_csv_parent_names=(cell_qc_dirname,),
        )

    coords = _run_umap(pooled, n_neighbors, min_dist, metric, random_state)

    suffix_part = f"_{suffix}" if suffix else ""
    x_col = f"umap_dinov2_1{suffix_part}"
    y_col = f"umap_dinov2_2{suffix_part}"
    pooled_df[x_col] = coords[:, 0]
    pooled_df[y_col] = coords[:, 1]

    out_root = outdir if outdir is not None else (base / MASTER_DIR_NAME)
    out_root.mkdir(parents=True, exist_ok=True)
    if suffix:
        # Splice the suffix in before "_all" so e.g.
        # dinov2_embeddings_union_488_560_all.npy -> dinov2_embeddings_union_488_560_pxfiltered_all.npy
        npy_name = master_npy_name.replace("_all.npy", f"_{suffix}_all.npy")
        csv_name = master_csv_name.replace("_all.csv", f"_{suffix}_all.csv")
    else:
        npy_name = master_npy_name
        csv_name = master_csv_name
    out_npy = out_root / npy_name
    out_csv = out_root / csv_name
    np.save(str(out_npy), pooled)
    pooled_df.to_csv(str(out_csv), index=False)
    print(f"Saved: {out_npy}  shape={pooled.shape}")
    print(f"Saved: {out_csv}  ({len(pooled_df)} rows, {len(pooled_df.columns)} cols)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-root", type=Path, default=None,
                    help=f"Root output directory to scan (default: {OUTPUT_DIR})")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Override the cell_qc_all/ destination directory")
    ap.add_argument("--filter-col", type=str, default=None,
                    help="Column in qc_features_filtered.csv to filter on (keep ==1), e.g. pass_intensity")
    ap.add_argument("--suffix", type=str, default=None,
                    help="Suffix for output column / file names, e.g. 'pxfiltered' "
                         "produces umap_dinov2_1_pxfiltered etc.")
    ap.add_argument(
        "--cell-qc-dir",
        type=str,
        default=None,
        help="Subfolder under each sample containing dinov2_embeddings.* "
             "(e.g. cell_qc_filtered). Defaults to the cell_qc dir for --variant.",
    )
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            f"Mask variant; sets --cell-qc-dir and the pooled master CSV/NPY name "
            f"from VARIANT_FILES (default: {DEFAULT_VARIANT}). Per-flag overrides win."
        ),
    )
    ap.add_argument(
        "--master-npy-name",
        type=str,
        default=None,
        help="Override pooled master NPY filename (defaults from --variant).",
    )
    ap.add_argument(
        "--master-csv-name",
        type=str,
        default=None,
        help="Override pooled master CSV filename (defaults from --variant).",
    )
    ap.add_argument("--n-neighbors", type=int, default=UMAP_DEFAULTS["n_neighbors"])
    ap.add_argument("--min-dist", type=float, default=UMAP_DEFAULTS["min_dist"])
    ap.add_argument("--metric", type=str, default=UMAP_DEFAULTS["metric"])
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    base = args.output_root or Path(OUTPUT_DIR)
    v = variant_files(args.variant)
    cell_qc_dirname = args.cell_qc_dir or v["cell_qc"]
    npy_default, csv_default = master_filenames_for_variant(args.variant)
    master_npy_name = args.master_npy_name or npy_default
    master_csv_name = args.master_csv_name or csv_default
    print(
        f"Variant: {args.variant}  cell_qc_dir={cell_qc_dirname}  "
        f"master_npy={master_npy_name}  master_csv={master_csv_name}"
    )
    return run(
        base,
        outdir=args.outdir,
        filter_col=args.filter_col,
        suffix=args.suffix,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        cell_qc_dirname=cell_qc_dirname,
        master_npy_name=master_npy_name,
        master_csv_name=master_csv_name,
    )


if __name__ == "__main__":
    sys.exit(main())
