#!/bin/bash
#SBATCH --job-name=dv2_emb
# Paths are relative to the submission directory (run from repo root).
#SBATCH --output=scripts/batch_dinov2/logs/dv2_%A_%a.out
#SBATCH --error=scripts/batch_dinov2/logs/dv2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
#
# Array range is NOT set in this file on purpose: pass it on the sbatch CLI
# (see submit_slurm.sh) so the tracked file is never rewritten, e.g.:
#   sbatch --array=1-50%20 --export=ALL,SAMPLES_FILE=$PWD/scripts/batch_dinov2/samples.txt \\
#     scripts/batch_dinov2/slurm_array.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT

SAMPLES_FILE="${SAMPLES_FILE:-$SCRIPT_DIR/samples.txt}"
if [[ ! -f "$SAMPLES_FILE" ]]; then
  echo "ERROR: SAMPLES_FILE not found: $SAMPLES_FILE" >&2
  exit 1
fi

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

LINE="${SLURM_ARRAY_TASK_ID:?}"
SAMPLE="$(awk -v "l=$LINE" 'NF && !/^[[:space:]]*#/ { c++; if (c == l) { print $1; exit } }' "$SAMPLES_FILE")"
if [[ -z "${SAMPLE:-}" ]]; then
  echo "ERROR: no sample on line index $LINE in $SAMPLES_FILE" >&2
  exit 1
fi

echo "Array task $LINE -> sample '$SAMPLE'"
exec bash "$SCRIPT_DIR/run_one_sample.sh" "$SAMPLE"
