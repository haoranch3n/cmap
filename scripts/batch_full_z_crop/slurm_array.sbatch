#!/bin/bash
#SBATCH --job-name=cmap_fzcrop
# Paths are relative to the submission directory (run from cmap repo root).
#SBATCH --output=scripts/batch_full_z_crop/logs/fz_%A_%a.out
#SBATCH --error=scripts/batch_full_z_crop/logs/fz_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=standard
##SBATCH --account=YOUR_ACCOUNT
#
# Array range is passed on the sbatch CLI (see submit_slurm.sh), e.g.:
#   sbatch --array=1-97%16 --export=ALL,PATHS_FILE=$PWD/logs/full_z_crop_paths.txt \\
#     scripts/batch_full_z_crop/slurm_array.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT
cd "$CMAP_ROOT"
export PYTHONUNBUFFERED=1

PATHS_FILE="${PATHS_FILE:?PATHS_FILE must point to the sample list (one absolute dir per line)}"
if [[ ! -f "$PATHS_FILE" ]]; then
  echo "ERROR: PATHS_FILE not found: $PATHS_FILE" >&2
  exit 1
fi

OUTDIR="$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" "$PATHS_FILE")"
if [[ -z "${OUTDIR}" ]]; then
  echo "Task ${SLURM_ARRAY_TASK_ID}: empty line (past end of list); exiting 0"
  exit 0
fi

echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${OUTDIR}"

EXTRA=()
if [[ "${CROP_FORCE:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi
if [[ -n "${MARGIN_XY:-}" ]]; then
  EXTRA+=(--margin-xy "$MARGIN_XY")
fi
if [[ -n "${MARGIN_Z:-}" ]]; then
  EXTRA+=(--margin-z "$MARGIN_Z")
fi

exec python features/crop_cells.py \
  --data-dir "$OUTDIR" \
  --output-dir "$OUTDIR" \
  --also-full-z \
  "${EXTRA[@]}"
