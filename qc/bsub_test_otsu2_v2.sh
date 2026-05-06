#!/bin/bash
#BSUB -n 1
#BSUB -q standard
#BSUB -J otsu2_test_v3
#BSUB -W 120
#BSUB -M 8000
#BSUB -R "rusage[mem=8000]"

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/4_24_25_CGN_6_10_2/Sample10_Position4_decon_dsr"

cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "Testing Otsu2 Thresholding (V2)"
echo "=========================================="
echo "Time: $(date)"
echo

# Step 1: Run original log_otsu
echo ">>> Step 1: Running log_otsu filter..."
python qc/filter_by_intensity.py \
  --from-volume \
  --method log_otsu \
  --variant filtered_642 \
  --output-dir "$SAMPLE_DIR" \
  --force

RC=$?
if [ $RC -ne 0 ]; then
  echo "ERROR: log_otsu failed with code $RC"
  exit 1
fi

# Step 2: Run otsu2
echo
echo ">>> Step 2: Running otsu2 filter..."
python qc/filter_by_intensity.py \
  --from-volume \
  --method otsu2 \
  --variant filtered_642 \
  --output-dir "$SAMPLE_DIR" \
  --force

RC=$?
if [ $RC -ne 0 ]; then
  echo "ERROR: otsu2 failed with code $RC"
  exit 1
fi

# Step 3: Compare
echo
echo ">>> Step 3: Comparing results..."
python qc/test_otsu2_directly.py "$SAMPLE_DIR"

RC=$?
if [ $RC -ne 0 ]; then
  echo "ERROR: Comparison failed with code $RC"
  exit 1
fi

echo "=========================================="
echo "Test Complete! ($(date))"
echo "=========================================="
