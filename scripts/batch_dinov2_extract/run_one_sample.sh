#!/usr/bin/env bash
# GPU: DINOv2 orthogonal_concat + volume norm for one sample under cmap/output/.
#
# Usage:
#   export CMAP_ROOT=/abs/path/to/cmap
#   bash "$CMAP_ROOT/scripts/batch_dinov2_extract/run_one_sample.sh" \
#     4_18_25/CGNSample1_Position0_decon_dsr cell_boxing_filtered cell_qc_filtered
#
# Skip if dinov2_embeddings.csv already exists unless FORCE_DINOV2=1 or 4th arg "force".
#
# Optional 4th arg: word "force" (LSF does not forward env vars unless bsub -env all).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SAMPLE_REL="${1:?sample relpath under output/, e.g. 4_18_25/CGNSample1_Position0_decon_dsr}"
CELL_BOXING_DIR="${2:-cell_boxing}"
CELL_QC_DIR="${3:-cell_qc}"
if [[ "${4:-}" == "force" ]]; then
  FORCE_DINOV2=1
fi

OUT_DIR="${CMAP_ROOT}/output/${SAMPLE_REL}"
CSV="${OUT_DIR}/dinov2_volume_norm_bounds.csv"
BOX="${OUT_DIR}/${CELL_BOXING_DIR}"
OUT_CSV="${OUT_DIR}/${CELL_QC_DIR}/dinov2_embeddings.csv"

if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: not a directory: $OUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$CSV" ]]; then
  echo "ERROR: missing volume norm CSV: $CSV" >&2
  exit 1
fi
if [[ ! -d "$BOX" ]]; then
  echo "ERROR: missing cell boxing dir: $BOX" >&2
  exit 1
fi
shopt -s nullglob
cells=( "${BOX}"/cell_*.tif )
shopt -u nullglob
if [[ ${#cells[@]} -eq 0 ]]; then
  echo "ERROR: no cell_*.tif in $BOX" >&2
  exit 1
fi

if [[ -f "$OUT_CSV" && "${FORCE_DINOV2:-0}" != "1" ]]; then
  echo "SKIP (exists): $OUT_CSV"
  exit 0
fi

cd "$CMAP_ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-cmap}"
fi

extra_force=()
if [[ "${FORCE_DINOV2:-0}" == "1" ]]; then
  extra_force=( --force )
fi

exec python features/extract_dinov2_embeddings.py \
  --data-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR" \
  --cell-boxing-dir "$CELL_BOXING_DIR" \
  --cell-qc-dir "$CELL_QC_DIR" \
  --norm-scope volume \
  --extraction-mode orthogonal_concat \
  --empty-frac 0 \
  --device cuda \
  "${extra_force[@]}"
