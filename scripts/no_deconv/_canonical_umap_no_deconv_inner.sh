#!/usr/bin/env bash
# LSF worker: canonical UMAP grid search + publish-best for no-deconv tree.
# Args:
#   $1 = PROJECT_ROOT (cmap repo)
#   $2 = OUTPUT_ROOT (e.g. .../output_no_deconv)
# Env: CMAP_SKIP_PUBLISH=1 to skip --publish-best

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${1:?PROJECT_ROOT required as \$1}"
OUTPUT_ROOT="${2:?OUTPUT_ROOT required as \$2}"
VARIANT="${CMAP_CANONICAL_UMAP_VARIANT:-no_deconv_bg_sigma_488560_shape}"
SKIP_PUBLISH="${CMAP_SKIP_PUBLISH:-0}"

eval "$(conda shell.bash hook)"
conda activate cmap
cd "$PROJECT_ROOT"

UMAP_PY="$PROJECT_ROOT/visualization/canonical_umap_grid_search.py"

echo "=== $(date -Is) canonical UMAP sweep variant=$VARIANT output-root=$OUTPUT_ROOT ==="

echo ">>> run-all-local"
python3 -u "$UMAP_PY" \
  --variant "$VARIANT" \
  --output-root "$OUTPUT_ROOT" \
  --run-all-local

if [[ "$SKIP_PUBLISH" != 1 ]]; then
  echo ">>> publish-best"
  python3 -u "$UMAP_PY" \
    --variant "$VARIANT" \
    --output-root "$OUTPUT_ROOT" \
    --publish-best
else
  echo "SKIP publish-best (CMAP_SKIP_PUBLISH=1)"
fi

echo "=== $(date -Is) DONE ==="
