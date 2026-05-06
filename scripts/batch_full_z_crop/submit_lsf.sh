#!/usr/bin/env bash
# Submit one LSF job per sample for full-Z cell cropping.
#
# Usage (from repo root):
#   bash scripts/batch_full_z_crop/submit_lsf.sh
#   bash scripts/batch_full_z_crop/submit_lsf.sh /path/to/output/root
#
# Environment:
#   OUTPUT_ROOT           root containing filtered_642_combined.tif per sample
#                         (default: PIPELINE_OUTPUT_DIR or <cmap>/output)
#   PATHS_FILE            pre-built list of absolute sample dirs (skip discovery)
#   LSF_RESOURCES         bsub resource string override
#   CROP_FORCE            set to 1 to pass --force
#   MARGIN_XY / MARGIN_Z override default margins

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

DEF_ROOT="${PIPELINE_OUTPUT_DIR:-$CMAP_ROOT/output}"
OUTPUT_ROOT="${1:-${OUTPUT_ROOT:-$DEF_ROOT}}"
LIST="${PATHS_FILE:-$CMAP_ROOT/logs/full_z_crop_paths.txt}"

LOGDIR="${LSF_LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOGDIR"
mkdir -p "$(dirname "$LIST")"

# Build sample list using the helper script
echo "Building sample list under: $OUTPUT_ROOT"
python "$CMAP_ROOT/scripts/batch_crop_cells_full_z.py" \
  --output-root "$OUTPUT_ROOT" \
  --write-list "$LIST"

N="$(wc -l < "$LIST")"
if [[ "$N" -lt 1 ]]; then
  echo "ERROR: no samples found under $OUTPUT_ROOT" >&2
  exit 1
fi
echo "Found $N sample(s) in $LIST"

# Propagate env vars into each job
ENV_EXPORTS="CMAP_ROOT=$CMAP_ROOT"
[[ "${CROP_FORCE:-0}" == "1" ]] && ENV_EXPORTS="$ENV_EXPORTS,CROP_FORCE=1"
[[ -n "${MARGIN_XY:-}" ]] && ENV_EXPORTS="$ENV_EXPORTS,MARGIN_XY=$MARGIN_XY"
[[ -n "${MARGIN_Z:-}" ]]  && ENV_EXPORTS="$ENV_EXPORTS,MARGIN_Z=$MARGIN_Z"
[[ -n "${CONDA_SH:-}" ]]  && ENV_EXPORTS="$ENV_EXPORTS,CONDA_SH=$CONDA_SH"
[[ -n "${CONDA_ENV:-}" ]] && ENV_EXPORTS="$ENV_EXPORTS,CONDA_ENV=$CONDA_ENV"
[[ -n "${VENV_ACTIVATE:-}" ]] && ENV_EXPORTS="$ENV_EXPORTS,VENV_ACTIVATE=$VENV_ACTIVATE"

IDX=0
while IFS= read -r SAMPLE_DIR || [[ -n "$SAMPLE_DIR" ]]; do
  [[ -z "${SAMPLE_DIR// }" ]] && continue
  IDX=$((IDX + 1))
  SAMPLE_NAME="$(basename "$SAMPLE_DIR")"

  echo "[$IDX/$N] bsub: $SAMPLE_NAME"

  bsub \
    -J "fzcrop_${SAMPLE_NAME}" \
    -oo "${LOGDIR}/fzcrop_${SAMPLE_NAME}.out" \
    -eo "${LOGDIR}/fzcrop_${SAMPLE_NAME}.err" \
    -env "$ENV_EXPORTS" \
    -u /dev/null \
    -q standard \
    -n 1 \
    -R "rusage[mem=16000] span[hosts=1]" \
    -W 4:00 \
    bash "$SCRIPT_DIR/run_one_sample.sh" "$SAMPLE_DIR"

done < "$LIST"

echo ""
echo "Submitted $IDX job(s)."
echo "Logs: $LOGDIR"
