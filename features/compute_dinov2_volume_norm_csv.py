#!/usr/bin/env python3
"""Compute per-channel min and 99.99th percentile from ``filtered_642_combined.tif``.

Run **once per sample** after the combined TIFF exists. DINOv2 volume
normalization then reads only the small CSV (see ``dinov2_volume_norm_csv``).

Example::

    python features/compute_dinov2_volume_norm_csv.py --data-rel my_sample
    python features/compute_dinov2_volume_norm_csv.py --output-dir /path/to/output/my_sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT  # type: ignore[no-redef]

from features.dinov2.volume_norm_csv import (
    compute_bounds_from_combined_tif,
    default_bounds_csv_path,
    write_bounds_csv,
)
from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files


def _resolve_output_dir(args) -> Path:
    if args.data_rel:
        return (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
    if args.output_dir:
        return Path(args.output_dir).resolve()
    raise SystemExit("Provide --data-rel or --output-dir")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-rel", type=str, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument(
        "--bounds-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <output_dir>/dinov2_volume_norm_bounds.csv)",
    )
    ap.add_argument(
        "--p-high",
        type=float,
        default=99.99,
        help="Upper percentile per channel (default: 99.99)",
    )
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=f"Mask variant whose combined TIFF and bounds CSV name to use (default: {DEFAULT_VARIANT})",
    )
    args = ap.parse_args()
    output_dir = _resolve_output_dir(args)
    v = variant_files(args.variant)
    combined = output_dir / v["combined"]
    if not combined.is_file():
        print(f"ERROR: combined TIFF not found: {combined}")
        return 1
    out_csv = (
        Path(args.bounds_csv)
        if args.bounds_csv
        else default_bounds_csv_path(output_dir, filename=v["dinov2_volume_bounds"])
    )
    lo, hi = compute_bounds_from_combined_tif(combined, p_high=args.p_high)
    write_bounds_csv(out_csv, lo, hi, source_tif=str(combined))
    print(f"Variant: {args.variant}  combined={v['combined']}  bounds_csv={out_csv.name}")
    print(f"Wrote {out_csv}")
    for i in range(3):
        print(f"  channel {i}:  lo={lo[i]:.6g}  hi={hi[i]:.6g}  (p_high={args.p_high})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
