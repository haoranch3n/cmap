#!/usr/bin/env bash
# Per-sample worker: crop + feature extract for the strict-overlap union after
# bg_sigma:3 (488+560) QC pass. Intended for LSF (standard queue, CPU).
#
# Usage:
#   bash scripts/_canonical_union_strict_crop_extract_one.sh <SAMPLE_DIR>
#
# Requires:
#   <SAMPLE_DIR>/union_488_560_strict_overlap_pass_bg_sigma_488560_shape.tif
#   <SAMPLE_DIR>/union_488_560_combined.tif
#
# Writes:
#   cell_box_bg_sigma_488560_shape/, cell_box_full_z_bg_sigma_488560_shape/
#   cell_qc_bg_sigma_488560_shape/qc_features.csv
#   cell_qc_bg_sigma_488560_shape/bg_stats.csv  (copy from cell_qc_union_488_560 when present)
#
# Log: <SAMPLE_DIR>/_lsf_logs/canonical_strict_crop_extract.log

set -euo pipefail

SAMPLE_DIR="${1:?ERROR: SAMPLE_DIR argument required}"
PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"

PASS_MASK="union_488_560_strict_overlap_pass_bg_sigma_488560_shape.tif"
QC_SRC="${SAMPLE_DIR}/cell_qc_union_488_560/bg_stats.csv"
QC_DST_DIR="${SAMPLE_DIR}/cell_qc_bg_sigma_488560_shape"
QC_DST="${QC_DST_DIR}/bg_stats.csv"

LOG_DIR="${SAMPLE_DIR}/_lsf_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/canonical_strict_crop_extract.log"

{
  echo "=========================================="
  echo "Canonical crop+extract (strict union, bg_sigma_488560)"
  echo "Sample:  ${SAMPLE_DIR}"
  echo "Started: $(date)"
  echo "Host:    $(hostname)"
  echo "=========================================="

  if [[ ! -f "${SAMPLE_DIR}/${PASS_MASK}" ]]; then
    echo "SKIP: missing ${PASS_MASK}"
    # Drop stale crops from earlier runs when this sample has no strict pass mask
    # (e.g. K_strict==0); otherwise Napari refresh counts orphan TIFFs vs master CSV.
    rm -rf "${SAMPLE_DIR}/cell_box_bg_sigma_488560_shape" \
           "${SAMPLE_DIR}/cell_box_full_z_bg_sigma_488560_shape"
    exit 0
  fi

  eval "$(conda shell.bash hook)"
  conda activate cmap
  cd "${PROJECT_ROOT}"

  # Remove stale crops from earlier union runs (different label sets leave
  # orphan cell_*.tif files that extract_features would otherwise still read).
  rm -rf "${SAMPLE_DIR}/cell_box_bg_sigma_488560_shape"
  rm -rf "${SAMPLE_DIR}/cell_box_full_z_bg_sigma_488560_shape"

  echo
  echo ">>> crop_cells.py (label mask = ${PASS_MASK})"
  python3 -u "${PROJECT_ROOT}/features/crop_cells.py" \
    --variant union_488_560 \
    --label-mask-name "${PASS_MASK}" \
    --output-subdir-key cell_box_bg_sigma_488560_shape \
    --also-full-z \
    --output-dir "${SAMPLE_DIR}" \
    --data-dir "${SAMPLE_DIR}" \
    --force
  echo "    crop exit $?"

  echo
  echo ">>> extract_features.py -> cell_qc_bg_sigma_488560_shape/"
  if python3 -u "${PROJECT_ROOT}/features/extract_features.py" \
    --variant union_488_560 \
    --cell-box-subdir-key cell_box_bg_sigma_488560_shape \
    --cell-qc-dir cell_qc_bg_sigma_488560_shape \
    --output-dir "${SAMPLE_DIR}" \
    --force; then
    echo "    extract exit 0"
  else
    ec=$?
    echo "    extract exit ${ec} (no crops — drop stale qc_features.csv if any)"
    mkdir -p "${QC_DST_DIR}"
    rm -f "${QC_DST_DIR}/qc_features.csv"
  fi

  if [[ -f "${QC_SRC}" ]]; then
    mkdir -p "${QC_DST_DIR}"
    cp -f "${QC_SRC}" "${QC_DST}"
    echo "Copied bg_stats -> ${QC_DST}"
  else
    echo "WARN: no ${QC_SRC} (canonical UMAP will skip bg columns for this sample)"
  fi

  echo
  echo "=========================================="
  echo "Finished: $(date)"
  echo "=========================================="
} 2>&1 | tee "${LOG}"
