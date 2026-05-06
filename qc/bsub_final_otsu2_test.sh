#!/bin/bash
#BSUB -n 1
#BSUB -q standard
#BSUB -J otsu2_final_test
#BSUB -W 120
#BSUB -M 8000
#BSUB -R "rusage[mem=8000]"

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"

# Use large sample for realistic test (will need 8GB memory)
SAMPLE_DIR="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/4_24_25_CGN_6_10_2/Sample10_Position4_decon_dsr"

# Fallback to small sample if large one doesn't exist
if [ ! -f "$SAMPLE_DIR/filtered_642_combined.tif" ]; then
  SAMPLE_DIR="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/_smoke_crop_demo"
fi

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "Otsu2 Thresholding Test - Final"
echo "=========================================="
echo "Sample: $SAMPLE_DIR"
echo "Time: $(date)"
echo

# Step 1: Run log_otsu
echo ">>> Step 1: Running log_otsu filter..."
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

# Save log_otsu CSV
CSV_BASE="$SAMPLE_DIR/cell_qc/qc_features_filtered"
mv "$CSV_BASE.csv" "$CSV_BASE""_log_otsu.csv"
echo "Saved: $CSV_BASE""_log_otsu.csv"
echo

# Step 2: Run otsu2
echo ">>> Step 2: Running otsu2 filter..."
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

# Save otsu2 CSV
mv "$CSV_BASE.csv" "$CSV_BASE""_otsu2.csv"
echo "Saved: $CSV_BASE""_otsu2.csv"
echo

# Step 3: Create filtered masks
echo ">>> Step 3: Creating filtered masks..."

python qc/create_otsu2_shape_filtered_mask.py \
  "$SAMPLE_DIR" \
  "$CSV_BASE""_otsu2.csv"

if [ $? -ne 0 ]; then
  echo "ERROR: Creating otsu2_shape_filtered mask failed"
  exit 1
fi

echo

# Step 4: Compare
echo ">>> Step 4: Comparing results..."
python qc/compare_otsu2_methods_separate.py "$SAMPLE_DIR"

if [ $? -ne 0 ]; then
  echo "ERROR: Comparison failed"
  exit 1
fi

echo "=========================================="
echo "Test Complete! ($(date))"
echo ""
echo "Output files:"
echo "  - $CSV_BASE""_log_otsu.csv"
echo "  - $CSV_BASE""_otsu2.csv"
echo "  - $SAMPLE_DIR/filtered_642_pass_otsu2_shape.tif"
echo "=========================================="
