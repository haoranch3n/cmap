#!/bin/bash
# Per-sample worker: crop cells from union_488_560_pass_bg_sigma_488560_shape.tif.
# Usage (called by submit_crop_bg_sigma_488560_batch.sh via bsub):
#   bash features/_crop_bg_sigma_488560_one.sh <SAMPLE_DIR>
#
# Writes cell_box_bg_sigma_488560_shape/ and cell_box_full_z_bg_sigma_488560_shape/.
# Log: <SAMPLE_DIR>/_lsf_logs/crop_bg_sigma_488560.log

set -uo pipefail

SAMPLE_DIR="${1:?ERROR: SAMPLE_DIR argument required}"
PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"

LOG_DIR="${SAMPLE_DIR}/_lsf_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/crop_bg_sigma_488560.log"

{
  echo "=========================================="
  echo "Crop bg_sigma_488560_shape — union_488_560"
  echo "Sample:  ${SAMPLE_DIR}"
  echo "Started: $(date)"
  echo "Host:    $(hostname)"
  echo "=========================================="

  python "${PROJECT_ROOT}/features/crop_cells.py" \
    --variant union_488_560 \
    --label-mask-name union_488_560_pass_bg_sigma_488560_shape.tif \
    --output-subdir-key cell_box_bg_sigma_488560_shape \
    --also-full-z \
    --output-dir "${SAMPLE_DIR}" \
    --data-dir "${SAMPLE_DIR}" \
    --force
  echo "  exit $?"

  echo "=========================================="
  echo "Finished: $(date)"
  echo "=========================================="
} 2>&1 | tee "${LOG}"
