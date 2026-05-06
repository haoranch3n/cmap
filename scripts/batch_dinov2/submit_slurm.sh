#!/usr/bin/env bash
# Submit a Slurm job array: one array task per non-comment line in samples.txt.
#
# Usage:
#   cd "$CMAP_ROOT"
#   bash scripts/batch_dinov2/discover_samples.sh > scripts/batch_dinov2/samples.txt
#   bash scripts/batch_dinov2/submit_slurm.sh
#
# Environment:
#   SAMPLES_FILE          path to samples list (default: scripts/batch_dinov2/samples.txt)
#   SLURM_ARRAY_THROTTLE  max concurrent tasks (default 20)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKDIR="$CMAP_ROOT"
LIST="${SAMPLES_FILE:-$SCRIPT_DIR/samples.txt}"

if [[ ! -f "$LIST" ]]; then
  echo "ERROR: samples file not found: $LIST" >&2
  exit 1
fi

N="$(grep -cve '^[[:space:]]*\(#\|$\)' "$LIST" || true)"
if [[ "$N" -lt 1 ]]; then
  echo "ERROR: no samples in $LIST" >&2
  exit 1
fi

THROTTLE="${SLURM_ARRAY_THROTTLE:-20}"
mkdir -p "$SCRIPT_DIR/logs"

ABS_LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"

echo "Submitting Slurm array 1-${N}%${THROTTLE} using SAMPLES_FILE=$ABS_LIST"
cd "$WORKDIR"
sbatch \
  --array="1-${N}%${THROTTLE}" \
  --export="ALL,SAMPLES_FILE=${ABS_LIST}" \
  "$SCRIPT_DIR/slurm_array.sbatch"
