#!/usr/bin/env bash
# Inner CPU smoke (invoked by bsub_smoke_reorder_union_one_cell.sh). Do not run directly.
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate cmap

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
SRC_SAMPLE="${CMAP_SMOKE_SRC_SAMPLE:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2/Sample10_Position4_decon_dsr}"
SMOKE_DIR="$PROJECT_ROOT/output/_smoke_reorder_one_cell/Sample10_Position4_decon_dsr"

cd "$PROJECT_ROOT"
mkdir -p "$SMOKE_DIR"
for f in union_488_560.tif union_488_560_combined.tif; do
  ln -sfn "$SRC_SAMPLE/$f" "$SMOKE_DIR/$f"
done

echo "=== 1/5 filter_by_intensity --from-volume ==="
python3 -u qc/filter_by_intensity.py --variant union_488_560 --from-volume --output-dir "$SMOKE_DIR" --force

echo "=== 2/5 apply_qc_pass ==="
python3 -u qc/apply_qc_pass_to_label_mask.py --variant union_488_560 --output-dir "$SMOKE_DIR" --overwrite

echo "=== 3/5 crop_cells filtered cell 37 ==="
python3 -u features/crop_cells.py --variant union_488_560 --output-dir "$SMOKE_DIR" --data-dir "$SMOKE_DIR" \
  --label-mask-name union_488_560_otsu_shape.tif --output-subdir-key cell_box_filtered \
  --cell-id 37 --also-full-z --force

echo "=== 4/5 extract_features filtered cell 37 ==="
python3 -u features/extract_features.py --variant union_488_560 --output-dir "$SMOKE_DIR" \
  --cell-box-subdir-key cell_box_filtered --cell-id 37 --force

echo "=== 5/5 merge_qc_features_filtered ==="
python3 -u qc/merge_qc_features_filtered.py --variant union_488_560 --output-dir "$SMOKE_DIR"

echo "CPU smoke steps completed (DINOv2 prep runs in GPU inner script)."
