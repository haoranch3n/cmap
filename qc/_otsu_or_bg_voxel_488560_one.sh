#!/bin/bash
# Per-sample worker: otsu_or_bg_voxel:3 (488+560 gating) from union volume, then join into union QC CSV.
# Usage (via bsub from submit_otsu_or_bg_voxel_488560_batch.sh):
#   bash qc/_otsu_or_bg_voxel_488560_one.sh <SAMPLE_DIR>
#
# Outputs:
#   <SAMPLE_DIR>/cell_qc_union_488_560_otsu_or_bg_voxel/qc_features_filtered.csv
#   <SAMPLE_DIR>/cell_qc_union_488_560_otsu_or_bg_voxel/bg_stats.csv
#   merges otsu_or_bg_voxel_* columns into:
#   <SAMPLE_DIR>/cell_qc_union_488_560/qc_features_filtered.csv
#
# Log: <SAMPLE_DIR>/_lsf_logs/otsu_or_bg_voxel_488560_apply.log

set -uo pipefail

SAMPLE_DIR="${1:?ERROR: SAMPLE_DIR argument required}"
PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
METHOD="${CMAP_OTSU_OR_BG_VOXEL_METHOD:-otsu_or_bg_voxel:3}"
# Optional Method A: trim 488 positive voxel pool before voxel log-Otsu (0 = off).
VOX488_PCT="${CMAP_VOXEL_488_POS_PCTILE_LO:-0}"
VARIANT="union_488_560"
CELL_QC_VOXEL="cell_qc_union_488_560_otsu_or_bg_voxel"

LOG_DIR="${SAMPLE_DIR}/_lsf_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/otsu_or_bg_voxel_488560_apply.log"

{
  echo "=========================================="
  echo "otsu_or_bg_voxel (488+560) — ${VARIANT}"
  echo "Sample:  ${SAMPLE_DIR}"
  echo "Method:  ${METHOD}"
  echo "voxel_488_positive_pctile_lo: ${VOX488_PCT}"
  echo "Started: $(date)"
  echo "Host:    $(hostname)"
  echo "=========================================="

  echo
  echo ">>> filter_by_intensity --from-volume --method ${METHOD} ..."
  python "${PROJECT_ROOT}/qc/filter_by_intensity.py" \
    --from-volume \
    --method "${METHOD}" \
    --filter-channels "488,560" \
    --variant "${VARIANT}" \
    --output-dir "${SAMPLE_DIR}" \
    --cell-qc-dir-name "${CELL_QC_VOXEL}" \
    --voxel-488-log-otsu-positive-pctile-lo "${VOX488_PCT}" \
    --force
  echo "    exit $?"

  echo
  echo ">>> join_qc_precrop_columns ..."
  python "${PROJECT_ROOT}/qc/join_qc_precrop_columns.py" \
    --sample-dir "${SAMPLE_DIR}"
  echo "    exit $?"

  echo
  echo "=========================================="
  echo "Finished: $(date)"
  echo "=========================================="
} 2>&1 | tee "${LOG}"
