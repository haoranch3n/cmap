#!/bin/bash
#BSUB -n 1
#BSUB -q standard
#BSUB -J otsu2_test
#BSUB -W 120

# Test Otsu2 implementation on a single sample

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/4_24_25_CGN_6_10_2/Sample10_Position4_decon_dsr"

mkdir -p "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "Testing Otsu2 Thresholding Implementation"
echo "=========================================="
echo "Sample: $SAMPLE_DIR"
echo "Time: $(date)"
echo

# Run original log_otsu method
echo ">>> Step 1: Running log_otsu (original method)..."
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
echo ">>> Step 2: Running otsu2 (new method with 99.99th percentile clipping)..."
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
echo ">>> Step 3: Creating filtered masks using pass_otsu_shape..."
python qc/apply_qc_pass_to_label_mask.py \
  --sample-dir "$SAMPLE_DIR" \
  --pass-column pass_otsu_shape \
  --variant filtered_642

if [ $? -ne 0 ]; then
  echo "ERROR: Creating pass_otsu_shape mask failed"
  exit 1
fi

echo
echo ">>> Step 4: Creating filtered masks using pass_otsu2_shape..."
python qc/apply_qc_pass_to_label_mask.py \
  --sample-dir "$SAMPLE_DIR" \
  --pass-column pass_otsu2_shape \
  --variant filtered_642

if [ $? -ne 0 ]; then
  echo "ERROR: Creating pass_otsu2_shape mask failed"
  exit 1
fi

echo
echo ">>> Step 5: Comparing results..."
python qc/compare_otsu_methods.py "$SAMPLE_DIR"

if [ $? -ne 0 ]; then
  echo "ERROR: Comparison failed"
  exit 1
fi

echo "=========================================="
echo "Test Complete! ($(date))"
echo "=========================================="
