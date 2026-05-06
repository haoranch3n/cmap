#!/bin/bash
#BSUB -n 1
#BSUB -q standard
#BSUB -J otsu2_large_sample
#BSUB -W 180
#BSUB -M 10000
#BSUB -R "rusage[mem=10000]"

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/4_24_25_CGN_6_10_2/Sample10_Position4_decon_dsr"

cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "Otsu2 Test on Large Sample"
echo "=========================================="
echo "Sample: $SAMPLE_DIR"
echo "Start: $(date)"
echo

# Step 1: Log_otsu
echo ">>> Running log_otsu..."
python qc/filter_by_intensity.py \
  --from-volume \
  --method log_otsu \
  --variant filtered_642 \
  --output-dir "$SAMPLE_DIR" \
  --force 2>&1 | tee /tmp/log_otsu.log

RC=$?
if [ $RC -ne 0 ]; then
  echo "ERROR: log_otsu failed ($RC)"
  exit 1
fi

# Save CSV
mv "$SAMPLE_DIR/cell_qc/qc_features_filtered.csv" \
   "$SAMPLE_DIR/cell_qc/qc_features_filtered_log_otsu.csv"
echo

# Step 2: Otsu2
echo ">>> Running otsu2..."
python qc/filter_by_intensity.py \
  --from-volume \
  --method otsu2 \
  --variant filtered_642 \
  --output-dir "$SAMPLE_DIR" \
  --force 2>&1 | tee /tmp/otsu2.log

RC=$?
if [ $RC -ne 0 ]; then
  echo "ERROR: otsu2 failed ($RC)"
  exit 1
fi

# Save CSV
mv "$SAMPLE_DIR/cell_qc/qc_features_filtered.csv" \
   "$SAMPLE_DIR/cell_qc/qc_features_filtered_otsu2.csv"
echo

# Step 3: Create mask
echo ">>> Creating otsu2_shape_filtered mask..."
python qc/create_otsu2_shape_filtered_mask.py \
  "$SAMPLE_DIR" \
  "$SAMPLE_DIR/cell_qc/qc_features_filtered_otsu2.csv" 2>&1 | tee /tmp/create_mask.log

RC=$?
if [ $RC -ne 0 ]; then
  echo "ERROR: create mask failed ($RC)"
  exit 1
fi

echo

# Step 4: Compare
echo ">>> Comparison Results:"
python qc/compare_otsu2_methods_separate.py "$SAMPLE_DIR" 2>&1 | tee /tmp/compare.log

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
