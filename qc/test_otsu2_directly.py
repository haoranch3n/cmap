#!/usr/bin/env python3
"""
Quick test of Otsu2 implementation.
Runs both methods and creates filtered masks for comparison.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile


def filter_mask_by_csv_column(
    mask_path: Path,
    csv_path: Path,
    pass_column: str,
    output_path: Path,
) -> int:
    """Filter label mask based on pass/fail column in CSV."""
    if not mask_path.exists():
        print(f"ERROR: Mask not found: {mask_path}")
        return 1
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    
    # Read CSV
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    
    if not rows or pass_column not in rows[0]:
        print(f"ERROR: CSV missing column '{pass_column}'")
        return 1
    
    # Get pass IDs
    keep_ids = set()
    for row in rows:
        try:
            if int(float(row.get(pass_column, 0))) == 1:
                cell_id = int(float(row["cell_id"]))
                keep_ids.add(cell_id)
        except (ValueError, TypeError):
            pass
    
    # Load and filter mask
    print(f"  Loading mask from: {mask_path}")
    mask = tifffile.imread(str(mask_path))
    
    # Create lookup table for label mapping
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=mask.dtype)
    for i in keep_ids:
        if 0 < i <= max_label:
            lut[i] = i
    
    # Apply lookup table
    filtered = lut[mask.astype(np.int64)]
    filtered = filtered.astype(mask.dtype)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing filtered mask to: {output_path}")
    tifffile.imwrite(str(output_path), filtered)
    
    # Count cells
    n_cells = len(keep_ids)
    n_kept = len(np.unique(filtered[filtered > 0]))
    print(f"  Cells kept: {n_kept}/{n_cells}")
    
    return 0


def test_otsu2(sample_dir: Path) -> int:
    sample_dir = Path(sample_dir).resolve()
    
    # Paths
    csv_path = sample_dir / "cell_qc" / "qc_features_filtered.csv"
    mask_path = sample_dir / "filtered_642.tif"
    
    mask_otsu_out = sample_dir / "filtered_642_pass_otsu_shape.tif"
    mask_otsu2_out = sample_dir / "filtered_642_pass_otsu2_shape.tif"
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    
    if not mask_path.exists():
        print(f"ERROR: Mask not found: {mask_path}")
        return 1
    
    print(f"\n{'='*70}")
    print(f"Otsu vs Otsu2 Test")
    print(f"{'='*70}")
    print(f"Sample: {sample_dir.name}\n")
    
    # Check CSV columns
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
    
    print(f"Available pass columns:")
    print(f"  pass_otsu_shape: {'pass_otsu_shape' in cols}")
    print(f"  pass_otsu2_shape: {'pass_otsu2_shape' in cols}")
    
    if "pass_otsu_shape" not in cols:
        print(f"\nERROR: pass_otsu_shape not in CSV")
        return 1
    
    if "pass_otsu2_shape" not in cols:
        print(f"\nERROR: pass_otsu2_shape not in CSV (did filter_by_intensity.py run with --method otsu2?)")
        return 1
    
    # Create filtered masks
    print(f"\nStep 1: Creating pass_otsu_shape mask...")
    if filter_mask_by_csv_column(mask_path, csv_path, "pass_otsu_shape", mask_otsu_out) != 0:
        return 1
    
    print(f"\nStep 2: Creating pass_otsu2_shape mask...")
    if filter_mask_by_csv_column(mask_path, csv_path, "pass_otsu2_shape", mask_otsu2_out) != 0:
        return 1
    
    # Compare
    print(f"\n{'Comparison':^70}")
    print(f"{'-'*70}")
    
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    
    total = len(rows)
    pass_otsu = sum(1 for r in rows if int(float(r.get("pass_otsu_shape", 0))) == 1)
    pass_otsu2 = sum(1 for r in rows if int(float(r.get("pass_otsu2_shape", 0))) == 1)
    
    print(f"CSV Statistics:")
    print(f"  Total cells: {total}")
    print(f"  Pass log_otsu (pass_otsu_shape): {pass_otsu:5d} ({100*pass_otsu/total:6.2f}%)")
    print(f"  Pass otsu2 (pass_otsu2_shape):  {pass_otsu2:5d} ({100*pass_otsu2/total:6.2f}%)")
    
    diff = pass_otsu2 - pass_otsu
    pct_diff = 100.0 * diff / total if total > 0 else 0
    print(f"  Difference:                    {diff:+5d} ({pct_diff:+6.2f}%)")
    if diff > 0:
        print(f"    → Otsu2 PASSES {diff} more cells (less stringent)")
    elif diff < 0:
        print(f"    → Otsu2 FILTERS OUT {-diff} more cells (more stringent)")
    else:
        print(f"    → Otsu2 and Otsu give identical results")
    
    # Count labels in masks
    print(f"\nLabel Mask Counts:")
    if mask_otsu_out.exists() and mask_otsu2_out.exists():
        labels_otsu = np.unique(tifffile.imread(str(mask_otsu_out)))
        labels_otsu2 = np.unique(tifffile.imread(str(mask_otsu2_out)))
        n_otsu = len(labels_otsu[labels_otsu > 0])
        n_otsu2 = len(labels_otsu2[labels_otsu2 > 0])
        
        print(f"  Cells in filtered_642_pass_otsu_shape.tif: {n_otsu}")
        print(f"  Cells in filtered_642_pass_otsu2_shape.tif: {n_otsu2}")
        
        diff_labels = n_otsu2 - n_otsu
        pct_diff_labels = 100.0 * diff_labels / n_otsu if n_otsu > 0 else 0
        print(f"  Difference:                              {diff_labels:+5d} ({pct_diff_labels:+6.2f}%)")
    
    # Threshold comparison
    print(f"\nThreshold Values (from first cell):")
    for ch in ["642", "488", "560"]:
        key_otsu = f"threshold_{ch}"
        key_otsu2 = f"threshold_{ch}"
        
        if key_otsu in cols and rows:
            try:
                val = float(rows[0][key_otsu])
                print(f"  Channel {ch}: {val:.4f}")
            except (ValueError, KeyError):
                pass
    
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <sample_dir>")
        sys.exit(1)
    
    sys.exit(test_otsu2(Path(sys.argv[1])))
