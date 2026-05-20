#!/usr/bin/env python3
"""Stage z±Z_PAD clipped cell crops for canonical feature extraction.

For each cell in ``output_registered_crop_seg/<batch>/<sample>/<cell_id>/``,
loads ``cell_XXXX_combined_union.tif`` (Z, 5, Y, X), finds the z-range where
the mask channel (index 3) is non-zero, clips to ±Z_PAD slices, and writes
a flat per-sample staging TIF:

    <output_root>/<batch>/<sample>/cell_box_union_registered/cell_XXXX.tif

This staging directory is the input for ``extract_features.py``:

    python features/extract_features.py \\
        --output-dir <sample_dir> \\
        --cell-boxing-dir cell_box_union_registered \\
        --cell-qc-dir cell_qc_union_registered \\
        --force

Usage:
    python prepare_crop_cells.py [--output-root PATH] [--batch BATCH] [--z-pad N] [--force]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

Z_PAD_DEFAULT = 5
OUTPUT_SUBDIR = "cell_box_union_registered"
UNION_SUFFIX = "_combined_union.tif"


def _mask_z_range(mask3d: np.ndarray) -> tuple[int, int] | None:
    """Return (z_min, z_max) inclusive where mask has any True pixel; None if empty."""
    zvals = np.where(mask3d.any(axis=(1, 2)))[0]
    if len(zvals) == 0:
        return None
    return int(zvals[0]), int(zvals[-1])


def prepare_one_cell(
    cell_dir: Path,
    sample_dir: Path,
    z_pad: int = Z_PAD_DEFAULT,
    force: bool = False,
) -> str:
    """Clip and stage one cell's combined_union TIF.  Returns 'done', 'skip', or 'error:<msg>'."""
    # Find the combined_union TIF (named cell_XXXX_combined_union.tif)
    tifs = list(cell_dir.glob(f"*{UNION_SUFFIX}"))
    if not tifs:
        return "error:no combined_union.tif"
    src = tifs[0]

    # Parse cell id from filename (e.g. cell_0005_combined_union.tif → 5)
    m = re.match(r"cell_(\d+)", src.name)
    if m is None:
        return f"error:cannot parse cell_id from {src.name}"
    cell_id = int(m.group(1))

    out_dir = sample_dir / OUTPUT_SUBDIR
    out_tif = out_dir / f"cell_{cell_id:04d}.tif"
    if out_tif.exists() and not force:
        return "skip"

    try:
        crop = tifffile.imread(str(src))
    except Exception as exc:
        return f"error:read failed: {exc}"

    if crop.ndim != 4 or crop.shape[1] < 4:
        return f"error:unexpected shape {crop.shape}"

    Z = crop.shape[0]
    mask3d = crop[:, 3, :, :] > 0
    rng = _mask_z_range(mask3d)

    if rng is None:
        # Empty mask — keep full volume (extract_features will skip via mask check)
        clipped = crop
    else:
        z_min, z_max = rng
        z_lo = max(0, z_min - z_pad)
        z_hi = min(Z, z_max + z_pad + 1)
        clipped = crop[z_lo:z_hi]

    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(out_tif),
        clipped,
        imagej=True,
        metadata={"axes": "ZCYX"},
    )
    return "done"


def prepare_sample(
    sample_dir: Path,
    z_pad: int = Z_PAD_DEFAULT,
    force: bool = False,
) -> dict[str, int]:
    """Process all cell_XXXX subdirs under sample_dir. Returns counts."""
    counts: dict[str, int] = {"done": 0, "skip": 0, "error": 0}
    cell_dirs = sorted(d for d in sample_dir.iterdir() if d.is_dir() and re.match(r"cell_\d+$", d.name))
    if not cell_dirs:
        return counts
    print(f"  {sample_dir.name}: {len(cell_dirs)} cells")
    for cd in cell_dirs:
        result = prepare_one_cell(cd, sample_dir, z_pad=z_pad, force=force)
        key = "error" if result.startswith("error") else result
        counts[key] += 1
        if result.startswith("error"):
            print(f"    ERROR {cd.name}: {result}")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root of segmentation output (default: <repo>/output_registered_crop_seg)",
    )
    ap.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Process only this batch subdirectory (e.g. 4_24_25_CGN_6_10_2). Default: all.",
    )
    ap.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Process only this sample subdirectory (e.g. Sample10_Position1). Default: all.",
    )
    ap.add_argument("--z-pad", type=int, default=Z_PAD_DEFAULT, help=f"Z slices to pad around mask (default: {Z_PAD_DEFAULT})")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output TIFs")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    args = ap.parse_args()

    # Resolve output root
    if args.output_root:
        output_root = args.output_root.resolve()
    else:
        project_root = Path(__file__).resolve().parents[2]
        output_root = project_root / "output_registered_crop_seg"

    if not output_root.is_dir():
        print(f"ERROR: output_root not found: {output_root}")
        return 1

    print(f"Output root:  {output_root}")
    print(f"Z-pad:        {args.z_pad}")
    print(f"Force:        {args.force}")
    print(f"Dry-run:      {args.dry_run}")
    print()

    if args.dry_run:
        # Just count what would be processed
        total = sum(
            1
            for b in (output_root.iterdir() if not args.batch else [output_root / args.batch])
            if b.is_dir()
            for s in (b.iterdir() if not args.sample else [b / args.sample])
            if s.is_dir()
            for c in s.iterdir()
            if c.is_dir() and re.match(r"cell_\d+$", c.name)
            for t in c.glob(f"*{UNION_SUFFIX}")
        )
        print(f"[dry-run] Would process {total} cells")
        return 0

    batches = [output_root / args.batch] if args.batch else [d for d in sorted(output_root.iterdir()) if d.is_dir()]

    totals: dict[str, int] = {"done": 0, "skip": 0, "error": 0}
    for batch_dir in batches:
        if not batch_dir.is_dir():
            print(f"WARN: batch dir not found: {batch_dir}")
            continue
        samples = [batch_dir / args.sample] if args.sample else sorted(
            d for d in batch_dir.iterdir() if d.is_dir()
        )
        for sample_dir in samples:
            if not sample_dir.is_dir():
                continue
            c = prepare_sample(sample_dir, z_pad=args.z_pad, force=args.force)
            for k in totals:
                totals[k] += c[k]

    print()
    print(f"=== Summary: done={totals['done']}  skip={totals['skip']}  error={totals['error']} ===")
    return 0 if totals["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
