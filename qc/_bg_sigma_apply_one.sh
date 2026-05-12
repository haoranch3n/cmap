#!/bin/bash
# Per-sample worker for bg_sigma:3 batch run.
# Usage (called by submit_bg_sigma_batch.sh via bsub):
#   bash qc/_bg_sigma_apply_one.sh <SAMPLE_DIR>
#
# Runs filter_by_intensity (bg_sigma:3) + apply_qc_pass on the strict-overlap
# union variant (union_488_560_strict_overlap.tif; gates 1+2+3).
# Logs are written to <SAMPLE_DIR>/_lsf_logs/bg_sigma_apply.log

set -uo pipefail

SAMPLE_DIR="${1:?ERROR: SAMPLE_DIR argument required}"
PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
METHOD="bg_sigma:3"
VARIANT="union_488_560"
STRICT_BASE="union_488_560_strict_overlap"
PASS_COL="pass_bg_sigma_shape"
MASK_NAME="${STRICT_BASE}.tif"
OUT_NAME="${STRICT_BASE}_pass_bg_sigma_shape.tif"
CSV_REL="cell_qc_${VARIANT}/qc_features_filtered.csv"

LOG_DIR="${SAMPLE_DIR}/_lsf_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/bg_sigma_apply.log"

{
  echo "=========================================="
  echo "bg_sigma:3 — ${VARIANT}"
  echo "Sample:  ${SAMPLE_DIR}"
  echo "Started: $(date)"
  echo "Host:    $(hostname)"
  echo "=========================================="

  echo
  echo ">>> filter_by_intensity --method ${METHOD} ..."
  python "${PROJECT_ROOT}/qc/filter_by_intensity.py" \
    --from-volume \
    --method "${METHOD}" \
    --variant "${VARIANT}" \
    --output-dir "${SAMPLE_DIR}" \
    --force
  echo "    exit $?"

  echo
  echo ">>> apply_qc_pass_to_label_mask ..."
  python "${PROJECT_ROOT}/qc/apply_qc_pass_to_label_mask.py" \
    --output-dir "${SAMPLE_DIR}" \
    --variant "${VARIANT}" \
    --pass-column "${PASS_COL}" \
    --mask-name "${MASK_NAME}" \
    --output-name "${OUT_NAME}" \
    --csv-rel "${CSV_REL}" \
    --overwrite
  echo "    exit $?"

  echo
  echo "=========================================="
  echo "Finished: $(date)"
  echo "=========================================="
} 2>&1 | tee "${LOG}"
