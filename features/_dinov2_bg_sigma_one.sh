#!/usr/bin/env bash
# GPU per-sample worker: DINOv2 orthogonal_concat from cell_box_bg_sigma_shape crops.
# Usage (called by submit_dinov2_bg_sigma_batch.sh via bsub):
#   bash features/_dinov2_bg_sigma_one.sh <SAMPLE_DIR>
#
# Requires:
#   <SAMPLE_DIR>/cell_box_bg_sigma_shape/   (from crop_bg_sigma_one.sh)
#   <SAMPLE_DIR>/dinov2_volume_norm_bounds_union_488_560.csv
#
# Output: <SAMPLE_DIR>/cell_qc_bg_sigma_shape/dinov2_embeddings.{npy,csv}

set -euo pipefail

SAMPLE_DIR="${1:?ERROR: SAMPLE_DIR argument required}"
PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"

CELL_BOXING_DIR="cell_box_bg_sigma_shape"
CELL_QC_DIR="cell_qc_bg_sigma_shape"
BOUNDS_CSV="${SAMPLE_DIR}/dinov2_volume_norm_bounds_union_488_560.csv"
BOX_DIR="${SAMPLE_DIR}/${CELL_BOXING_DIR}"
OUT_CSV="${SAMPLE_DIR}/${CELL_QC_DIR}/dinov2_embeddings.csv"

if [[ ! -d "${BOX_DIR}" ]]; then
  echo "ERROR: missing cell boxing dir: ${BOX_DIR}" >&2
  exit 1
fi
shopt -s nullglob
cells=( "${BOX_DIR}"/cell_*.tif )
shopt -u nullglob
if [[ ${#cells[@]} -eq 0 ]]; then
  echo "ERROR: no cell_*.tif in ${BOX_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BOUNDS_CSV}" ]]; then
  echo "ERROR: missing volume norm bounds CSV: ${BOUNDS_CSV}" >&2
  exit 1
fi

if [[ -f "${OUT_CSV}" && "${FORCE_DINOV2:-0}" != "1" ]]; then
  echo "SKIP (exists): ${OUT_CSV}"
  exit 0
fi

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-cmap}"
fi

extra_force=()
if [[ "${FORCE_DINOV2:-0}" == "1" ]]; then
  extra_force=( --force )
fi

exec python features/extract_dinov2_embeddings.py \
  --output-dir "${SAMPLE_DIR}" \
  --variant union_488_560 \
  --cell-boxing-dir "${CELL_BOXING_DIR}" \
  --cell-qc-dir "${CELL_QC_DIR}" \
  --norm-scope volume \
  --extraction-mode orthogonal_concat \
  --empty-frac 0 \
  --device cuda \
  "${extra_force[@]}"
