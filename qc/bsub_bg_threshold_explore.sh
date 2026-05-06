#!/bin/bash
#BSUB -n 1
#BSUB -q standard
#BSUB -J bg_thresh_explore
#BSUB -W 180
#BSUB -M 32000
#BSUB -R "rusage[mem=32000]"

# Explore background-statistics QC thresholds on Sample7_Position7 (4_24_25 CGN)
# for both mask variants (union_488_560, filtered_642) so we can pick a
# winning method for the bg-sigma intensity filter.

set -uo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="${PROJECT_ROOT}/output/4_24_25_CGN_6_10_2/Sample7_Position7_decon_dsr"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "BG threshold exploration"
echo "Sample:  ${SAMPLE_DIR}"
echo "Started: $(date)"
echo "Host:    $(hostname)"
echo "=========================================="

run_variant() {
  local variant="$1"
  local extra="${2:-}"
  echo
  echo ">>> Variant: ${variant} ${extra}"
  python "${PROJECT_ROOT}/qc/bg_threshold_explore.py" \
    --output-dir "${SAMPLE_DIR}" \
    --variant "${variant}" \
    --track-cell 25 \
    ${extra} \
    2>&1
  local rc=$?
  echo "    (exit ${rc})"
  return 0
}

# Per user direction (2026-05-06): focus on union_488_560 only.
run_variant "union_488_560"
run_variant "union_488_560" "--no-clip-negatives --report-csv ${SAMPLE_DIR}/bg_threshold_report_union_488_560_noclip.csv"

echo
echo "=========================================="
echo "Finished: $(date)"
echo "Reports under: ${SAMPLE_DIR}/bg_threshold_report_*.csv"
echo "=========================================="
