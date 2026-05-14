#!/usr/bin/env bash
# Per-sample worker: full union_488_560 chain for one no-deconv sample.
#
# Steps:
#   1. postprocess/union_488_560_mask.py        -- strict 488+560 overlap mask
#   2. qc/filter_by_intensity.py                -- bg_sigma:3, filter-channels 488,560
#   3. qc/apply_qc_pass_to_label_mask.py        -- write pass mask
#   4. features/crop_cells.py                   -- cell_box_bg_sigma_488560_shape/
#   5. features/extract_features.py             -- cell_qc_bg_sigma_488560_shape/
#
# Arguments:
#   $1 = PROJECT_ROOT
#   $2 = SAMPLE_DIR (absolute path to per-sample merge root under output_no_deconv)
#
# Env:
#   CMAP_FORCE_OVERWRITE=1  -- pass --force / --overwrite to each step

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${1:?PROJECT_ROOT required as \$1}"
SAMPLE_DIR="${2:?SAMPLE_DIR required as \$2}"
FORCE="${CMAP_FORCE_OVERWRITE:-0}"

eval "$(conda shell.bash hook)"
conda activate cmap
cd "$PROJECT_ROOT"

PY="${CONDA_PREFIX}/bin/python"

FORCE_FLAG=""
OVERWRITE_FLAG=""
if [[ "$FORCE" == "1" ]]; then
  FORCE_FLAG="--force"
  OVERWRITE_FLAG="--overwrite"
fi

PASS_MASK="union_488_560_strict_overlap_pass_bg_sigma_488560_shape.tif"
QC_DIR="cell_qc_bg_sigma_488560_shape"

echo "=== $(date -Is) START union chain (no-deconv): $SAMPLE_DIR ==="

# 1. Strict 488+560 overlap mask
echo "" && echo ">>> postprocess.union_488_560_mask"
"$PY" postprocess/union_488_560_mask.py \
  --output-dir "$SAMPLE_DIR" \
  --data-dir   "$SAMPLE_DIR" \
  $FORCE_FLAG

# 2. Intensity + shape QC (bg_sigma:3, only 488+560 must pass)
echo "" && echo ">>> qc.filter_by_intensity (bg_sigma:3, filter-channels 488,560)"
"$PY" qc/filter_by_intensity.py \
  --variant union_488_560 \
  --from-volume \
  --method bg_sigma:3 \
  --filter-channels 488,560 \
  --cell-qc-dir-name "$QC_DIR" \
  --output-dir "$SAMPLE_DIR" \
  $FORCE_FLAG

# 3. Apply pass mask
echo "" && echo ">>> qc.apply_qc_pass (pass_bg_sigma_488560_shape)"
"$PY" qc/apply_qc_pass_to_label_mask.py \
  --variant    union_488_560 \
  --sample-dir "$SAMPLE_DIR" \
  --pass-column pass_bg_sigma_488560_shape \
  --csv-rel    "$QC_DIR/qc_features_filtered.csv" \
  --output-name "$PASS_MASK" \
  $OVERWRITE_FLAG

# 4. Crop cells into cell_box_bg_sigma_488560_shape/
echo "" && echo ">>> features.crop (cell_box_bg_sigma_488560_shape)"
"$PY" features/crop_cells.py \
  --variant            union_488_560 \
  --label-mask-name    "$PASS_MASK" \
  --output-subdir-key  cell_box_bg_sigma_488560_shape \
  --also-full-z \
  --output-dir "$SAMPLE_DIR" \
  --data-dir   "$SAMPLE_DIR" \
  $FORCE_FLAG

# 5. Extract features into cell_qc_bg_sigma_488560_shape/
echo "" && echo ">>> features.extract (cell_qc_bg_sigma_488560_shape)"
"$PY" features/extract_features.py \
  --variant              union_488_560 \
  --cell-box-subdir-key  cell_box_bg_sigma_488560_shape \
  --cell-qc-dir          "$QC_DIR" \
  --output-dir "$SAMPLE_DIR" \
  $FORCE_FLAG

echo "" && echo "=== $(date -Is) DONE union chain (no-deconv): $SAMPLE_DIR ==="
