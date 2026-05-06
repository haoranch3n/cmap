#!/usr/bin/env python3
"""
Create otsu2_shape_filtered.tif mask from otsu2 pass column.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile


def create_otsu2_filtered_mask(sample_dir: Path, csv_path: Path | None = None) -> int:
    """Create otsu2_shape_filtered.tif from pass_otsu2_shape column."""
    sample_dir = Path(sample_dir).resolve()
    
    if csv_path is None:
        csv_path = sample_dir / "cell_qc" / "qc_features_filtered.csv"
    else:
        csv_path = Path(csv_path)
    
    mask_path = sample_dir / "filtered_642.tif"
    output_path = sample_dir / "filtered_642_pass_otsu2_shape.tif"
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    
    if not mask_path.exists():
        print(f"ERROR: Mask not found: {mask_path}")
        return 1
    
    # Read CSV to get pass_otsu2_shape
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    
    if not rows or "pass_otsu2_shape" not in rows[0]:
        print(f"ERROR: CSV missing pass_otsu2_shape column")
        return 1
    
    # Load mask
    print(f"Loading mask: {mask_path}")
    mask = tifffile.imread(str(mask_path))
    
    # Create lookup table
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=mask.dtype)
    
    n_pass = 0
    for row in rows:
        try:
            if int(float(row.get("pass_otsu2_shape", 0))) == 1:
                cell_id = int(float(row["cell_id"]))
                if 0 < cell_id <= max_label:
                    lut[cell_id] = cell_id
                    n_pass += 1
        except (ValueError, TypeError):
            pass
    
    # Apply filter
    filtered = lut[mask.astype(np.int64)]
    filtered = filtered.astype(mask.dtype)
    
    # Count cells
    n_labels = len(np.unique(filtered[filtered > 0]))
    
    # Write output
    print(f"Writing: {output_path}")
    tifffile.imwrite(str(output_path), filtered)
    
    print(f"Done! Cells passing otsu2_shape: {n_pass} → {n_labels} labels in mask")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sample_dir> [csv_path]")
        sys.exit(1)
    
    csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(create_otsu2_filtered_mask(Path(sys.argv[1]), csv_path))
