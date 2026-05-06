#!/usr/bin/env python3
"""
Compare Otsu vs Otsu2 from separate CSV files.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile


def compare_from_csvs(sample_dir: Path) -> int:
    sample_dir = Path(sample_dir).resolve()
    
    csv_otsu_path = sample_dir / "cell_qc" / "qc_features_filtered_log_otsu.csv"
    csv_otsu2_path = sample_dir / "cell_qc" / "qc_features_filtered_otsu2.csv"
    mask_path = sample_dir / "filtered_642.tif"
    
    if not csv_otsu_path.exists():
        print(f"ERROR: log_otsu CSV not found: {csv_otsu_path}")
        return 1
    
    if not csv_otsu2_path.exists():
        print(f"ERROR: otsu2 CSV not found: {csv_otsu2_path}")
        return 1
    
    # Read both CSVs
    with open(csv_otsu_path, newline="") as f:
        rows_otsu = list(csv.DictReader(f))
    
    with open(csv_otsu2_path, newline="") as f:
        rows_otsu2 = list(csv.DictReader(f))
    
    print(f"\n{'='*70}")
    print(f"Otsu vs Otsu2 Comparison (Separate CSVs)")
    print(f"{'='*70}")
    print(f"Sample: {sample_dir.name}\n")
    
    # Check columns
    cols_otsu = set(rows_otsu[0].keys()) if rows_otsu else set()
    cols_otsu2 = set(rows_otsu2[0].keys()) if rows_otsu2 else set()
    
    print(f"Log_otsu CSV columns: {len(cols_otsu)}")
    print(f"Otsu2 CSV columns: {len(cols_otsu2)}")
    
    pass_col_otsu = "pass_otsu_shape" if "pass_otsu_shape" in cols_otsu else None
    pass_col_otsu2 = "pass_otsu2_shape" if "pass_otsu2_shape" in cols_otsu2 else None
    
    if not pass_col_otsu:
        print(f"ERROR: pass_otsu_shape not in log_otsu CSV")
        return 1
    if not pass_col_otsu2:
        print(f"ERROR: pass_otsu2_shape not in otsu2 CSV")
        return 1
    
    print(f"\n{'CSV Statistics':^70}")
    print(f"{'-'*70}")
    
    total = len(rows_otsu)
    pass_otsu = sum(1 for r in rows_otsu if int(float(r.get(pass_col_otsu, 0))) == 1)
    pass_otsu2 = sum(1 for r in rows_otsu2 if int(float(r.get(pass_col_otsu2, 0))) == 1)
    
    print(f"Total cells: {total}")
    print(f"Pass log_otsu: {pass_otsu:5d} ({100*pass_otsu/total:6.2f}%)")
    print(f"Pass otsu2:   {pass_otsu2:5d} ({100*pass_otsu2/total:6.2f}%)")
    
    diff = pass_otsu2 - pass_otsu
    pct_diff = 100.0 * diff / total if total > 0 else 0
    print(f"Difference:   {diff:+5d} ({pct_diff:+6.2f}%)")
    if diff > 0:
        print(f"  ← Otsu2 passes {diff} more cells")
    elif diff < 0:
        print(f"  ← Otsu2 is more stringent ({-diff} fewer cells)")
    
    # Threshold comparison
    print(f"\n{'Threshold Comparison':^70}")
    print(f"{'-'*70}")
    
    for ch in ["642", "488", "560"]:
        thresh_col = f"threshold_{ch}"
        if thresh_col in cols_otsu and thresh_col in cols_otsu2:
            try:
                t_otsu = float(rows_otsu[0][thresh_col])
                t_otsu2 = float(rows_otsu2[0][thresh_col])
                ratio = t_otsu2 / t_otsu if t_otsu > 0 else 1.0
                print(f"Channel {ch}:")
                print(f"  log_otsu: {t_otsu:.4f}")
                print(f"  otsu2:    {t_otsu2:.4f}  ({ratio:.2f}x)")
            except (ValueError, KeyError):
                pass
    
    # Create and compare filtered masks
    print(f"\n{'Filtered Mask Comparison':^70}")
    print(f"{'-'*70}")
    
    if mask_path.exists():
        mask = tifffile.imread(str(mask_path))
        
        for method, row_list, pass_col, name in [
            ("log_otsu", rows_otsu, pass_col_otsu, "pass_otsu_shape"),
            ("otsu2", rows_otsu2, pass_col_otsu2, "pass_otsu2_shape"),
        ]:
            # Create filtered mask
            max_label = int(mask.max())
            lut = np.zeros(max_label + 1, dtype=mask.dtype)
            
            n_pass = 0
            for row in row_list:
                try:
                    if int(float(row.get(pass_col, 0))) == 1:
                        cell_id = int(float(row["cell_id"]))
                        if 0 < cell_id <= max_label:
                            lut[cell_id] = cell_id
                            n_pass += 1
                except (ValueError, TypeError):
                    pass
            
            filtered = lut[mask.astype(np.int64)]
            n_labels = len(np.unique(filtered[filtered > 0]))
            
            print(f"{method.capitalize():12}: {n_pass:3d} pass → {n_labels:3d} labels in mask")
    
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <sample_dir>")
        sys.exit(1)
    
    sys.exit(compare_from_csvs(Path(sys.argv[1])))
