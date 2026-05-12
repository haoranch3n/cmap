#!/bin/bash
# Per-sample worker: run bg_sigma:3 with 488+560 gating only, then apply mask.
# Usage (called by submit_bg_sigma_488560_batch.sh via bsub):
#   bash qc/_bg_sigma_488560_apply_one.sh <SAMPLE_DIR>
#
# Operates on the strict-overlap union (union_488_560_strict_overlap.tif).
#
# Outputs:
#   <SAMPLE_DIR>/cell_qc_union_488_560/qc_features_filtered.csv  (pass_bg_sigma_488560_shape col)
#   <SAMPLE_DIR>/union_488_560_strict_overlap_pass_bg_sigma_488560_shape.tif
# Log: <SAMPLE_DIR>/_lsf_logs/bg_sigma_488560_apply.log

set -uo pipefail

SAMPLE_DIR="${1:?ERROR: SAMPLE_DIR argument required}"
PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
METHOD="bg_sigma:3"
VARIANT="union_488_560"
STRICT_BASE="union_488_560_strict_overlap"
PASS_COL="pass_bg_sigma_488560_shape"
MASK_NAME="${STRICT_BASE}.tif"
OUT_NAME="${STRICT_BASE}_pass_bg_sigma_488560_shape.tif"
CSV_REL="cell_qc_${VARIANT}/qc_features_filtered.csv"

LOG_DIR="${SAMPLE_DIR}/_lsf_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/bg_sigma_488560_apply.log"

{
  echo "=========================================="
  echo "bg_sigma:3 (488+560 only) — ${VARIANT}"
  echo "Sample:  ${SAMPLE_DIR}"
  echo "Started: $(date)"
  echo "Host:    $(hostname)"
  echo "=========================================="

  echo
  echo ">>> filter_by_intensity --method ${METHOD} --filter-channels 488,560 ..."
  python "${PROJECT_ROOT}/qc/filter_by_intensity.py" \
    --from-volume \
    --method "${METHOD}" \
    --filter-channels "488,560" \
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
