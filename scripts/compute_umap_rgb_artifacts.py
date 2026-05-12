#!/usr/bin/env python3
"""Build UMAP RGB artifacts (bounds + per-cell colors) for the Napari QC plugin.

Writes under ``<output>/cell_qc_all/``:

1. ``umap_rgb_intensity_bounds.csv`` — one row: pLOW / pHIGH for mean_488,
   mean_560, mean_642 over cells whose optional path column is not a full-Z
   crop directory.
2. ``umap_rgb_per_cell.csv`` — one row per cell in the master feature CSV:
   sample, cell_id, optional dataset, umap_rgb_r/g/b (488→R, 560→G, 642→B).

Requires a master feature CSV (same discovery as Napari ``build_feature_table``).

Examples:

  export PIPELINE_OUTPUT_DIR=/path/to/pipeline/output
  PYTHONPATH=napari-plugin/src python scripts/compute_umap_rgb_artifacts.py

  python scripts/compute_umap_rgb_artifacts.py --output-root /path/to/output
  python scripts/compute_umap_rgb_artifacts.py --percentile-high 99
  python scripts/compute_umap_rgb_artifacts.py --bounds-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from cell_qc_plugin._constants import (
    CELL_QC_ALL_DIR,
    UMAP_RGB_BOUNDS_CSV_NAME,
    UMAP_RGB_PER_CELL_CSV_NAME,
)
from cell_qc_plugin._io import _find_master_csv
from cell_qc_plugin._umap_rgb import (
    build_per_cell_rgb_dataframe,
    compute_bounds_row,
    read_bounds_csv,
)


def _resolve_output_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().absolute()
    env = os.environ.get("PIPELINE_OUTPUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().absolute()
    return Path("output").absolute()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Pipeline output root (default: $PIPELINE_OUTPUT_DIR or ./output).",
    )
    ap.add_argument(
        "--bounds-only",
        action="store_true",
        help="Only write umap_rgb_intensity_bounds.csv.",
    )
    ap.add_argument(
        "--per-cell-only",
        action="store_true",
        help="Only write umap_rgb_per_cell.csv (reads existing bounds CSV).",
    )
    ap.add_argument(
        "--percentile-low",
        type=float,
        default=0.0,
        help="Lower percentile for intensity clipping (default: 0).",
    )
    ap.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper percentile for intensity clipping (default: 99). "
             "Use a lower value (e.g. 99) to prevent extreme outliers from "
             "washing out the majority of cells.",
    )
    args = ap.parse_args(argv)

    if args.bounds_only and args.per_cell_only:
        print("Choose at most one of --bounds-only and --per-cell-only", file=sys.stderr)
        return 2

    root = _resolve_output_root(args.output_root)
    master = _find_master_csv(root, image_files=None)
    if master is None:
        print(
            f"No master feature CSV found under {root} (expected {CELL_QC_ALL_DIR}/).",
            file=sys.stderr,
        )
        return 1

    qc_all = master.parent
    bounds_path = qc_all / UMAP_RGB_BOUNDS_CSV_NAME
    per_cell_path = qc_all / UMAP_RGB_PER_CELL_CSV_NAME

    try:
        df = pd.read_csv(str(master))
    except Exception as exc:
        print(f"Failed to read {master}: {exc}", file=sys.stderr)
        return 1

    if "cell_id" not in df.columns or "sample" not in df.columns:
        print(f"Master CSV {master} must have cell_id and sample columns.", file=sys.stderr)
        return 1

    need_mean = {"mean_488", "mean_560", "mean_642"}
    if not need_mean.issubset(df.columns):
        print(
            f"Master CSV must contain columns {sorted(need_mean)} for RGB; missing.",
            file=sys.stderr,
        )
        return 1

    run_bounds = not args.per_cell_only
    run_per_cell = not args.bounds_only

    bounds: dict
    if run_bounds:
        bounds = compute_bounds_row(
            df,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )
        pd.DataFrame([bounds]).to_csv(str(bounds_path), index=False)
        print(f"Wrote {bounds_path} (n_cells_bounds_cohort={bounds.get('n_cells_bounds_cohort')})")

    if run_per_cell:
        if not run_bounds:
            if not bounds_path.is_file():
                print(f"Missing {bounds_path}; run without --per-cell-only first.", file=sys.stderr)
                return 1
            bounds = read_bounds_csv(bounds_path)
        per_cell = build_per_cell_rgb_dataframe(df, bounds)
        per_cell.to_csv(str(per_cell_path), index=False)
        print(f"Wrote {per_cell_path} ({len(per_cell)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
