#!/usr/bin/env bash
# Submit a Slurm job array: one array task per line in the sample-path list.
#
# Usage (from cmap repo root):
#   bash scripts/batch_full_z_crop/submit_slurm.sh
#   bash scripts/batch_full_z_crop/submit_slurm.sh /path/to/pipeline/output/root
#
# Environment:
#   OUTPUT_ROOT           root containing filtered_642_combined.tif per sample
#                         (default: segmentation_multiscale_cellpose_3D/output)
#   PATHS_FILE            where to write the list (default: logs/full_z_crop_paths.txt)
#   SLURM_ARRAY_THROTTLE  max concurrent array tasks (default: 16)
#   CROP_FORCE            set to 1 to pass --force to crop_cells
#
# If a sequential ``batch_crop_cells_full_z.py`` is still running, cancel it
# first to avoid duplicate I/O:  pkill -f batch_crop_cells_full_z.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKDIR="$CMAP_ROOT"
DEF_ROOT="$CMAP_ROOT/segmentation_multiscale_cellpose_3D/output"
OUTPUT_ROOT="${1:-${OUTPUT_ROOT:-$DEF_ROOT}}"
LIST="${PATHS_FILE:-$CMAP_ROOT/logs/full_z_crop_paths.txt}"
THROTTLE="${SLURM_ARRAY_THROTTLE:-16}"

mkdir -p "$(dirname "$LIST")"
mkdir -p "$SCRIPT_DIR/logs"

echo "Building sample list under: $OUTPUT_ROOT"
python "$CMAP_ROOT/scripts/batch_crop_cells_full_z.py" \
  --output-root "$OUTPUT_ROOT" \
  --write-list "$LIST"

N="$(wc -l < "$LIST")"
if [[ "$N" -lt 1 ]]; then
  echo "ERROR: empty list $LIST" >&2
  exit 1
fi

ABS_LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"

EXPORT="ALL,PATHS_FILE=${ABS_LIST},CMAP_ROOT=${CMAP_ROOT}"
if [[ "${CROP_FORCE:-0}" == "1" ]]; then
  EXPORT="${EXPORT},CROP_FORCE=1"
fi
if [[ -n "${MARGIN_XY:-}" ]]; then
  EXPORT="${EXPORT},MARGIN_XY=${MARGIN_XY}"
fi
if [[ -n "${MARGIN_Z:-}" ]]; then
  EXPORT="${EXPORT},MARGIN_Z=${MARGIN_Z}"
fi

echo "Submitting Slurm array 1-${N}%${THROTTLE} with PATHS_FILE=$ABS_LIST"
cd "$WORKDIR"
sbatch \
  --array="1-${N}%${THROTTLE}" \
  --export="${EXPORT}" \
  "$SCRIPT_DIR/slurm_array.sbatch"
