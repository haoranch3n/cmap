#!/usr/bin/env bash
# Batch job: prepare z±Z_PAD crops + extract canonical features
# for the registered-crop segmentation output using the 488+560 union mask.
#
# Stage 1 – prepare_crop_cells_488_560.py (all batches at once):
#   Clips each cell_XXXX_combined_union_488_560.tif to z±Z_PAD of the mask,
#   writes to <sample>/cell_box_union_488_560/cell_XXXX.tif.
#
# Stage 2 – extract_features.py (per sample, both batches):
#   Reads cell_box_union_488_560/, writes cell_qc_union_488_560/qc_features.csv.
#
# Usage:
#   bsub < segmentation_registered_crop/scripts/bsub_prepare_extract_crop_488_560.sh
#   # or run directly (no bsub):
#   bash segmentation_registered_crop/scripts/bsub_prepare_extract_crop_488_560.sh
#
# Env overrides:
#   CMAP_SEG_CROP_OUTPUT_ROOT   default: <repo>/output_registered_crop_seg
#   CMAP_SEG_CROP_Z_PAD         default: 5

#BSUB -J prep_extract_crop_488_560
#BSUB -q standard
#BSUB -n 2
#BSUB -W 120
#BSUB -M 16000
#BSUB -R "rusage[mem=16000] span[hosts=1]"
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/logs/prep_extract_crop_488_560_%J.out
#BSUB -e /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/logs/prep_extract_crop_488_560_%J.err

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
OUTPUT_ROOT="${CMAP_SEG_CROP_OUTPUT_ROOT:-${PROJECT_ROOT}/output_registered_crop_seg}"
Z_PAD="${CMAP_SEG_CROP_Z_PAD:-5}"

declare -a BATCHES=("4_18_25" "4_24_25_CGN_6_10_2")

echo "=========================================="
echo "Prepare + Extract: registered crop cells (488+560 union)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_ROOT:  ${OUTPUT_ROOT}"
echo "BATCHES:      ${BATCHES[*]}"
echo "Z_PAD:        ${Z_PAD}"
echo "Started:      $(date)"
echo "Host:         $(hostname)"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate cmap
cd "${PROJECT_ROOT}"

# ── Stage 1: clip to z±Z_PAD for all batches ────────────────────────────────
echo
echo ">>> Stage 1: prepare z±${Z_PAD} crops  (prepare_crop_cells_488_560.py)"
python3 -u "${PROJECT_ROOT}/segmentation_registered_crop/scripts/prepare_crop_cells_488_560.py" \
  --output-root "${OUTPUT_ROOT}" \
  --z-pad "${Z_PAD}" \
  --force
echo "  prepare exit $?"

# ── Stage 2: extract canonical QC features per sample ───────────────────────
echo
echo ">>> Stage 2: extract_features.py → cell_qc_union_488_560/"
n_ok=0
n_skip=0

for batch in "${BATCHES[@]}"; do
  batch_dir="${OUTPUT_ROOT}/${batch}"
  [[ -d "${batch_dir}" ]] || { echo "  SKIP (missing batch): ${batch}"; continue; }

  for sample_dir in "${batch_dir}"/*/; do
    sample_dir="${sample_dir%/}"
    [[ -d "${sample_dir}" ]] || continue
    if [[ ! -d "${sample_dir}/cell_box_union_488_560" ]]; then
      echo "  SKIP (no crops): ${batch}/$(basename "${sample_dir}")"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "  ${batch}/$(basename "${sample_dir}")"
    python3 -u "${PROJECT_ROOT}/features/extract_features.py" \
      --output-dir "${sample_dir}" \
      --cell-box-subdir-key cell_box_union_488_560 \
      --cell-qc-dir cell_qc_union_488_560 \
      --force
    n_ok=$((n_ok + 1))
  done
done

echo
echo "  samples done: ${n_ok}  skipped: ${n_skip}"
echo
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
