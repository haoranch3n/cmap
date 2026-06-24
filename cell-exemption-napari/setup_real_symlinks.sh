#!/usr/bin/env bash
# Build a lightweight symlink farm of the real CMAP samples for the Cell
# Exemption plugin. Symlinks (not copies) -> no storage duplication, same read
# speed, and the plugin's discovery only walks this clean tree.
#
# For each sample with both required files it creates:
#   <DEST>/<batch>/<sample>/cell_box_bg_sigma_488560_shape_combined.tif        -> real TIFF
#   <DEST>/<batch>/<sample>/cell_box_bg_sigma_488560_shape/cell_coordinates.csv -> real CSV
#
# Usage (from login node, read-only on source):
#   bash cell-exemption-napari/setup_real_symlinks.sh
# Env overrides:
#   CMAP_SRC_ROOT   source data root   (default: <repo>/output_no_deconv)
#   CMAP_LINK_ROOT  symlink farm root  (default: <plugin>/real_data)
#   CMAP_BATCHES    space-separated batch names to include
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$PLUGIN_DIR/.." && pwd)"

SRC_ROOT="${CMAP_SRC_ROOT:-$REPO_ROOT/output_no_deconv}"
LINK_ROOT="${CMAP_LINK_ROOT:-$PLUGIN_DIR/real_data}"
BATCHES="${CMAP_BATCHES:-4_18_25 4_24_25_CGN_6_10_2}"

COMBINED="cell_box_bg_sigma_488560_shape_combined.tif"
SUBDIR="cell_box_bg_sigma_488560_shape"
COORD="cell_coordinates.csv"

echo "Source : $SRC_ROOT"
echo "Links  : $LINK_ROOT"
echo "Batches: $BATCHES"
echo

n_ok=0
n_skip=0
for batch in $BATCHES; do
  bdir="$SRC_ROOT/$batch"
  [[ -d "$bdir" ]] || { echo "MISSING batch dir: $bdir"; continue; }
  for sdir in "$bdir"/*/; do
    [[ -d "$sdir" ]] || continue
    sample="$(basename "$sdir")"
    src_tif="${sdir}${COMBINED}"
    src_csv="${sdir}${SUBDIR}/${COORD}"
    if [[ ! -f "$src_tif" || ! -f "$src_csv" ]]; then
      n_skip=$((n_skip + 1))
      continue
    fi
    dest="$LINK_ROOT/$batch/$sample"
    mkdir -p "$dest/$SUBDIR"
    ln -sfn "$src_tif" "$dest/$COMBINED"
    ln -sfn "$src_csv" "$dest/$SUBDIR/$COORD"
    n_ok=$((n_ok + 1))
  done
done

echo "Linked samples : $n_ok"
echo "Skipped (missing files) : $n_skip"
echo "Symlink farm root: $LINK_ROOT"
