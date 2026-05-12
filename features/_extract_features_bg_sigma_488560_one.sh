#!/bin/bash
# Per-sample worker: canonical features from cell_box_bg_sigma_488560_shape crops.
# Usage (called by submit_extract_features_bg_sigma_488560_batch.sh via bsub):
#   bash features/_extract_features_bg_sigma_488560_one.sh <SAMPLE_DIR>
#
# Output: <SAMPLE_DIR>/cell_qc_bg_sigma_488560_shape/qc_features.csv
# Log:    <SAMPLE_DIR>/_lsf_logs/extract_features_bg_sigma_488560.log

set -uo pipefail

SAMPLE_DIR="${1:?ERROR: SAMPLE_DIR argument required}"
PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"

LOG_DIR="${SAMPLE_DIR}/_lsf_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/extract_features_bg_sigma_488560.log"

{
  echo "=========================================="
  echo "Extract features — cell_box_bg_sigma_488560_shape"
  echo "Sample:  ${SAMPLE_DIR}"
  echo "Started: $(date)"
  echo "Host:    $(hostname)"
  echo "=========================================="

  python "${PROJECT_ROOT}/features/extract_features.py" \
    --variant union_488_560 \
    --cell-box-subdir-key cell_box_bg_sigma_488560_shape \
    --cell-qc-dir cell_qc_bg_sigma_488560_shape \
    --output-dir "${SAMPLE_DIR}" \
    --force
  echo "  exit $?"

  echo "=========================================="
  echo "Finished: $(date)"
  echo "=========================================="
} 2>&1 | tee "${LOG}"
