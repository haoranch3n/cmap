#!/usr/bin/env bash
# Inner GPU smoke (invoked by bsub_smoke_reorder_union_one_cell.sh). Do not run directly.
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate cmap

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
SMOKE_DIR="$PROJECT_ROOT/output/_smoke_reorder_one_cell/Sample10_Position4_decon_dsr"

cd "$PROJECT_ROOT"
echo "=== GPU 1/2 compute_dinov2_volume_norm_csv ==="
python3 -u features/compute_dinov2_volume_norm_csv.py --variant union_488_560 --output-dir "$SMOKE_DIR"

echo "=== GPU 2/2 extract_dinov2_embeddings ==="
python3 -u features/extract_dinov2_embeddings.py \
  --variant union_488_560 \
  --output-dir "$SMOKE_DIR" \
  --cell-boxing-dir cell_box_otsu_shape_filtered \
  --cell-id 37 \
  --norm-scope volume \
  --extraction-mode orthogonal_concat \
  --empty-frac 0 \
  --device cuda \
  --force

echo "GPU DINOv2 smoke completed."
