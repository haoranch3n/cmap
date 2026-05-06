#!/usr/bin/env python3
"""
Compare Otsu vs Otsu2 thresholding results.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile


def compare_otsu_methods(sample_dir: Path) -> int:
    sample_dir = Path(sample_dir).resolve()
    csv_path = sample_dir / "cell_qc" / "qc_features_filtered.csv"
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    
    print(f"\n{'='*60}")
    print(f"Otsu vs Otsu2 Comparison")
    print(f"{'='*60}")
    print(f"Sample: {sample_dir.name}\n")
    
    # Read CSV
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    
    if not rows:
        print("ERROR: CSV has no rows")
        return 1
    
    # Check what columns are available
    cols = set(rows[0].keys())
    has_otsu = "pass_otsu_shape" in cols
    has_otsu2 = "pass_otsu2_shape" in cols
    
    print(f"Available columns:")
    print(f"  pass_otsu_shape: {has_otsu}")
    print(f"  pass_otsu2_shape: {has_otsu2}")
    
    if not (has_otsu or has_otsu2):
        print("\nERROR: Neither pass_otsu_shape nor pass_otsu2_shape found in CSV")
        return 1
    
    print(f"\n{'CSV Statistics':^60}")
    print(f"{'-'*60}")
    
    total = len(rows)
    print(f"Total cells: {total}")
    
    if has_otsu:
        pass_otsu = sum(1 for r in rows if int(float(r.get("pass_otsu_shape", 0))) == 1)
        pct_otsu = 100.0 * pass_otsu / total if total > 0 else 0
        print(f"Pass log_otsu (pass_otsu_shape): {pass_otsu:5d} ({pct_otsu:6.2f}%)")
    else:
        pass_otsu = None
    
    if has_otsu2:
        pass_otsu2 = sum(1 for r in rows if int(float(r.get("pass_otsu2_shape", 0))) == 1)
        pct_otsu2 = 100.0 * pass_otsu2 / total if total > 0 else 0
        print(f"Pass otsu2 (pass_otsu2_shape):  {pass_otsu2:5d} ({pct_otsu2:6.2f}%)")
    else:
        pass_otsu2 = None
    
    if pass_otsu is not None and pass_otsu2 is not None:
        diff = pass_otsu2 - pass_otsu
        pct_diff = 100.0 * diff / total if total > 0 else 0
        print(f"Difference:                    {diff:+5d} ({pct_diff:+6.2f}%)")
        print(f"  (otsu2 {'allows' if diff > 0 else 'filters out'} {abs(diff)} more cells)")
    
    # Threshold comparison
    print(f"\n{'Per-channel Threshold Comparison':^60}")
    print(f"{'-'*60}")
    
    for ch in ["642", "488", "560"]:
        key_otsu = f"threshold_{ch}"
        key_otsu2 = f"threshold_{ch}"  # Will be the same column in current impl
        
        if key_otsu in cols and rows:
            try:
                otsu_val = float(rows[0][key_otsu])
                print(f"Channel {ch}: {otsu_val:.4f}")
            except (ValueError, KeyError):
                pass
    
    # Try to load masks
    print(f"\n{'Label Mask Comparison':^60}")
    print(f"{'-'*60}")
    
    mask_otsu_path = sample_dir / "filtered_642_pass_otsu_shape.tif"
    mask_otsu2_path = sample_dir / "filtered_642_pass_otsu2_shape.tif"
    
    if mask_otsu_path.exists():
        labels_otsu = np.unique(tifffile.imread(str(mask_otsu_path)))
        n_otsu = len(labels_otsu[labels_otsu > 0])
        print(f"Cells in filtered_642_pass_otsu_shape.tif: {n_otsu}")
    else:
        print(f"filtered_642_pass_otsu_shape.tif: NOT FOUND")
        n_otsu = None
    
    if mask_otsu2_path.exists():
        labels_otsu2 = np.unique(tifffile.imread(str(mask_otsu2_path)))
        n_otsu2 = len(labels_otsu2[labels_otsu2 > 0])
        print(f"Cells in filtered_642_pass_otsu2_shape.tif: {n_otsu2}")
    else:
        print(f"filtered_642_pass_otsu2_shape.tif: NOT FOUND")
        n_otsu2 = None
    
    if n_otsu is not None and n_otsu2 is not None:
        diff_labels = n_otsu2 - n_otsu
        pct_diff_labels = 100.0 * diff_labels / n_otsu if n_otsu > 0 else 0
        print(f"Difference:                           {diff_labels:+5d} ({pct_diff_labels:+6.2f}%)")
    
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <sample_dir>")
        sys.exit(1)
    
    sample_dir = Path(sys.argv[1])
    sys.exit(compare_otsu_methods(sample_dir))
