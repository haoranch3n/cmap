#!/bin/bash
#BSUB -n 1
#BSUB -q standard
#BSUB -J otsu2_test
#BSUB -W 60
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/logs/otsu2_test_%J.log

# Test Otsu2 implementation on a single sample

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/4_24_25_CGN_6_10_2/Sample10_Position4_decon_dsr"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Testing Otsu2 Thresholding Implementation"
echo "=========================================="
echo "Sample: $SAMPLE_DIR"
echo

# Run original log_otsu method
echo ">>> Running log_otsu (original method)..."
python qc/filter_by_intensity.py \
  --from-volume \
  --method log_otsu \
  --variant filtered_642 \
  --output-dir "$SAMPLE_DIR" \
  --force

if [ $? -ne 0 ]; then
  echo "ERROR: log_otsu failed"
  exit 1
fi

echo
echo ">>> Running otsu2 (new method with 99.99th percentile clipping)..."
python qc/filter_by_intensity.py \
  --from-volume \
  --method otsu2 \
  --variant filtered_642 \
  --output-dir "$SAMPLE_DIR" \
  --force

if [ $? -ne 0 ]; then
  echo "ERROR: otsu2 failed"
  exit 1
fi

echo
echo ">>> Generating filtered masks and comparison..."
CSV_PATH="$SAMPLE_DIR/cell_qc/qc_features_filtered.csv"

# Create masks for both methods
python qc/apply_qc_pass_to_label_mask.py \
  --sample-dir "$SAMPLE_DIR" \
  --pass-column pass_otsu_shape \
  --variant filtered_642

if [ $? -ne 0 ]; then
  echo "ERROR: Creating pass_otsu_shape mask failed"
  exit 1
fi

python qc/apply_qc_pass_to_label_mask.py \
  --sample-dir "$SAMPLE_DIR" \
  --pass-column pass_otsu2_shape \
  --variant filtered_642

if [ $? -ne 0 ]; then
  echo "ERROR: Creating pass_otsu2_shape mask failed"
  exit 1
fi

echo
echo ">>> Comparison Results"
python3 << 'EOF'
import csv
import numpy as np
import tifffile
from pathlib import Path

csv_path = Path("$SAMPLE_DIR") / "cell_qc" / "qc_features_filtered.csv"
mask_otsu = Path("$SAMPLE_DIR") / "filtered_642_pass_otsu_shape.tif"
mask_otsu2 = Path("$SAMPLE_DIR") / "filtered_642_pass_otsu2_shape.tif"

# Read CSV
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

# Get pass counts
pass_otsu = sum(int(float(r.get('pass_otsu_shape', 0))) for r in rows)
pass_otsu2 = sum(int(float(r.get('pass_otsu2_shape', 0))) for r in rows)
total = len(rows)

print(f"\nCSV Statistics:")
print(f"  Total cells: {total}")
print(f"  Pass log_otsu (pass_otsu_shape): {pass_otsu} ({100*pass_otsu/total:.1f}%)")
print(f"  Pass otsu2 (pass_otsu2_shape): {pass_otsu2} ({100*pass_otsu2/total:.1f}%)")
print(f"  Difference: {pass_otsu2 - pass_otsu} ({100*(pass_otsu2-pass_otsu)/total:+.1f}%)")

# Get per-channel thresholds
print(f"\nPer-channel Threshold Comparison:")
for ch in ["642", "488", "560"]:
    otsu_thresh = [float(r.get(f'threshold_{ch}', 0)) for r in rows]
    otsu2_thresh = [float(r.get(f'threshold_{ch}_otsu2', 0) if f'threshold_{ch}_otsu2' in rows[0] else r.get(f'threshold_{ch}', 0)) for r in rows]
    
    if otsu_thresh:
        print(f"\n  Channel {ch}:")
        print(f"    log_otsu threshold: {otsu_thresh[0]:.2f}")

# Check label counts in masks
if mask_otsu.exists() and mask_otsu2.exists():
    labels_otsu = np.unique(tifffile.imread(str(mask_otsu)))
    labels_otsu2 = np.unique(tifffile.imread(str(mask_otsu2)))
    n_otsu = len(labels_otsu[labels_otsu > 0])
    n_otsu2 = len(labels_otsu2[labels_otsu2 > 0])
    print(f"\nLabel Mask Comparison:")
    print(f"  Cells in pass_otsu_shape.tif: {n_otsu}")
    print(f"  Cells in pass_otsu2_shape.tif: {n_otsu2}")
    print(f"  Difference: {n_otsu2 - n_otsu} ({100*(n_otsu2-n_otsu)/n_otsu:+.1f}%)")

EOF

echo
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
