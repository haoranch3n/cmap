#!/usr/bin/env bash
# Worker: full downstream chain for one no-deconv sample (filtered_642 variant).
#
# Arguments:
#   $1 = PROJECT_ROOT   (cmap repo root)
#   $2 = SAMPLE_DIR     (absolute path to per-sample merge root,
#                        e.g. .../output_no_deconv/4_18_25/CGNSample1_Position0)
#
# Steps:
#   1. postprocess/filter_642_mask.py      -- filter 642 by triple overlap, write filtered_642.tif + combined
#   2. postprocess/combine_with_mask.py    -- per-channel combined TIFs
#   3. qc/filter_by_intensity.py           -- Otsu + shape gate, write cell_qc/qc_features_filtered.csv
#   4. qc/apply_qc_pass_to_label_mask.py   -- write filtered_642_pass_otsu_shape.tif
#   5. features/crop_cells.py             -- crop per-cell TIFs into cell_boxing_filtered/
#   6. features/extract_features.py       -- per-cell feature CSV
#   7. qc/merge_qc_features_filtered.py   -- join precrop pass flags + postcrop features

set -euo pipefail

PROJECT_ROOT="${1:?PROJECT_ROOT required as \$1}"
SAMPLE_DIR="${2:?SAMPLE_DIR required as \$2}"

eval "$(conda shell.bash hook)"
conda activate cmap

cd "$PROJECT_ROOT"
PY="${CONDA_PREFIX}/bin/python"

echo "=== $(date -Is) START downstream: $SAMPLE_DIR ==="

# 1. Filter 642 mask by triple overlap; writes filtered_642.tif + filtered_642_combined.tif
echo "" && echo ">>> postprocess.filter"
"$PY" postprocess/filter_642_mask.py \
  --output-dir "$SAMPLE_DIR" \
  --data-dir   "$SAMPLE_DIR"

# 2. Per-channel combined TIFs (original + indexed mask)
echo "" && echo ">>> postprocess.combine"
"$PY" postprocess/combine_with_mask.py \
  --output-dir "$SAMPLE_DIR" \
  --data-dir   "$SAMPLE_DIR"

# 3. Intensity + shape QC filter on the filtered_642_combined.tif volume
echo "" && echo ">>> qc.filter_by_intensity (--from-volume)"
"$PY" qc/filter_by_intensity.py \
  --variant filtered_642 \
  --from-volume \
  --output-dir "$SAMPLE_DIR"

# 4. Apply QC pass to write filtered_642_pass_otsu_shape.tif
echo "" && echo ">>> qc.apply_qc_pass"
"$PY" qc/apply_qc_pass_to_label_mask.py \
  --variant   filtered_642 \
  --sample-dir "$SAMPLE_DIR"

# 5. Crop per-cell 3D TIFs into cell_boxing_filtered/
echo "" && echo ">>> features.crop"
"$PY" features/crop_cells.py \
  --variant           filtered_642 \
  --label-mask-name   filtered_642_pass_otsu_shape.tif \
  --output-subdir-key cell_box_filtered \
  --output-dir "$SAMPLE_DIR" \
  --data-dir   "$SAMPLE_DIR"

# 6. Extract per-cell features
echo "" && echo ">>> features.extract"
"$PY" features/extract_features.py \
  --variant             filtered_642 \
  --cell-box-subdir-key cell_box_filtered \
  --output-dir "$SAMPLE_DIR" \
  --data-dir   "$SAMPLE_DIR"

# 7. Merge precrop pass flags + postcrop features into final qc_features_filtered.csv
echo "" && echo ">>> qc.merge"
"$PY" qc/merge_qc_features_filtered.py \
  --variant    filtered_642 \
  --output-dir "$SAMPLE_DIR"

echo "" && echo "=== $(date -Is) DONE downstream: $SAMPLE_DIR ==="
