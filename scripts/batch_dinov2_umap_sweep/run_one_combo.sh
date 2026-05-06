#!/usr/bin/env bash
# Run one UMAP grid cell (CPU). Args: n_neighbors min_dist metric
#
#   export CMAP_ROOT=/abs/path/to/cmap
#   bash "$CMAP_ROOT/scripts/batch_dinov2_umap_sweep/run_one_combo.sh" 5 0.0 cosine

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
NN="${1:?n_neighbors}"
MD="${2:?min_dist}"
MET="${3:?metric}"

cd "$CMAP_ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-cmap}" 2>/dev/null || true
fi

OUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-$CMAP_ROOT/output}"

exec python visualization/dinov2_umap_grid_search.py \
  --output-root "$OUT_ROOT" \
  --n-neighbors "$NN" \
  --min-dist "$MD" \
  --metric "$MET"
