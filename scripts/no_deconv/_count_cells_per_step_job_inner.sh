#!/usr/bin/env bash
# LSF worker: run count_cells_per_pipeline_step.py (conda cmap, repo root on PYTHONPATH).
# Args:
#   $1 = PROJECT_ROOT  (cmap repo, absolute)
#   $2 = OUTPUT_ROOT    (e.g. $PROJECT_ROOT/output_no_deconv, absolute)
#   $3 = OUT_CSV        (absolute path to output CSV)

set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="${1:?PROJECT_ROOT required}"
OUTPUT_ROOT="${2:?OUTPUT_ROOT required}"
OUT_CSV="${3:?OUT_CSV required}"

eval "$(conda shell.bash hook)"
conda activate cmap
cd "$PROJECT_ROOT"

exec "${CONDA_PREFIX}/bin/python" scripts/no_deconv/count_cells_per_pipeline_step.py \
  --output-root "$OUTPUT_ROOT" \
  --out-csv "$OUT_CSV"
