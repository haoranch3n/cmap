#!/usr/bin/env python3
"""
Sanity check: verify that each original (padded-Z) crop in cell_boxing/
is an exact 3-D sub-volume of its corresponding full-Z crop in
cell_boxing_full_z/.

For every sample that has both directories, the script:
  1. Reads the summary CSVs to get the Z-offsets (z0, z1) of each cell.
  2. For a random subset of cells (--max-cells), loads both TIFs.
  3. Checks that original == full_z[z0:z1] (all 5 channels, voxel-exact).

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import tifffile


def _load(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        arr = arr[:, np.newaxis, :, :]
    return arr


def check_sample(sample_dir: Path, max_cells: int, verbose: bool) -> tuple[int, int]:
    """Return (checked, failed) counts for one sample."""
    box_dir = sample_dir / "cell_boxing"
    fz_dir = sample_dir / "cell_boxing_full_z"
    box_csv = box_dir / "summary.csv"
    fz_csv = fz_dir / "summary.csv"

    if not box_csv.exists() or not fz_csv.exists():
        return 0, 0

    with open(box_csv) as f:
        padded_rows = {int(r["cell_id"]): r for r in csv.DictReader(f)}
    with open(fz_csv) as f:
        fz_rows = {int(r["cell_id"]): r for r in csv.DictReader(f)}

    common = sorted(set(padded_rows) & set(fz_rows))
    if not common:
        return 0, 0

    if max_cells and len(common) > max_cells:
        common = sorted(random.sample(common, max_cells))

    checked = 0
    failed = 0
    for cid in common:
        pr = padded_rows[cid]
        fr = fz_rows[cid]
        z0_pad = int(pr["z0"])
        z1_pad = int(pr["z1"])
        z0_fz = int(fr["z0"])

        pad_path = box_dir / f"cell_{cid:04d}.tif"
        fz_path = fz_dir / f"cell_{cid:04d}.tif"
        if not pad_path.exists() or not fz_path.exists():
            continue

        pad_arr = _load(pad_path)
        fz_arr = _load(fz_path)

        # XY dims must match
        if pad_arr.shape[1:] != fz_arr.shape[1:]:
            print(f"  FAIL cell {cid}: shape mismatch — padded {pad_arr.shape} vs full_z {fz_arr.shape}")
            failed += 1
            checked += 1
            continue

        # The padded crop spans z0_pad..z1_pad in the original volume.
        # The full-z crop starts at z0_fz (should be 0).
        offset = z0_pad - z0_fz
        expected = fz_arr[offset : offset + pad_arr.shape[0]]

        if expected.shape != pad_arr.shape:
            print(
                f"  FAIL cell {cid}: slice shape mismatch — "
                f"expected {expected.shape}, padded {pad_arr.shape} "
                f"(z0_pad={z0_pad}, z0_fz={z0_fz}, fz_z_depth={fz_arr.shape[0]})"
            )
            failed += 1
            checked += 1
            continue

        if np.array_equal(pad_arr, expected):
            if verbose:
                print(f"  OK   cell {cid}  padded Z=[{z0_pad}:{z1_pad}]  shape={pad_arr.shape}")
        else:
            diff_mask = pad_arr != expected
            n_diff = int(diff_mask.sum())
            max_abs = float(np.abs(pad_arr.astype(np.float64) - expected.astype(np.float64)).max())
            print(
                f"  FAIL cell {cid}: {n_diff} voxels differ "
                f"(max |delta|={max_abs:.6g})  padded Z=[{z0_pad}:{z1_pad}]"
            )
            failed += 1

        checked += 1

    return checked, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "output_root",
        type=Path,
        help="Root dir containing sample subdirectories (e.g. segmentation_multiscale_cellpose_3D/output)",
    )
    ap.add_argument("--max-cells", type=int, default=5, help="Max cells to check per sample (0 = all)")
    ap.add_argument("--max-samples", type=int, default=0, help="Max samples to check (0 = all)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = args.output_root.resolve()

    samples = sorted(
        d
        for d in root.rglob("cell_boxing_full_z")
        if d.is_dir() and (d.parent / "cell_boxing").is_dir()
    )
    samples = [s.parent for s in samples]

    if args.max_samples and len(samples) > args.max_samples:
        samples = sorted(random.sample(samples, args.max_samples))

    print(f"Checking {len(samples)} sample(s) under {root}")
    print(f"Max cells per sample: {args.max_cells or 'all'}\n")

    total_checked = 0
    total_failed = 0
    for s in samples:
        rel = s.relative_to(root)
        print(f"[{rel}]")
        c, f = check_sample(s, max_cells=args.max_cells, verbose=args.verbose)
        if c == 0:
            print("  (no cells to check)")
        elif f == 0:
            print(f"  PASS  {c} cells checked")
        total_checked += c
        total_failed += f

    print(f"\n{'='*60}")
    print(f"Total: {total_checked} cells checked, {total_failed} failures")
    if total_failed:
        print("RESULT: FAIL")
        return 1
    else:
        print("RESULT: PASS — all original crops are exact sub-volumes of full-z crops")
        return 0


if __name__ == "__main__":
    sys.exit(main())
