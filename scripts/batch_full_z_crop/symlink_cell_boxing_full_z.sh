#!/usr/bin/env bash
# Create symlinks under cmap/output/ for every file under .../cell_boxing_full_z/
# in the segmentation output tree. Each link points to the real file on
# /research_jude/rgs01_jude/dept/DNB/... (absolute paths).
#
# Usage (from repo root):
#   bash scripts/batch_full_z_crop/symlink_cell_boxing_full_z.sh
#
# Environment:
#   SEG_OUTPUT_ROOT   source root (default: <CMAP_ROOT>/segmentation_multiscale_cellpose_3D/output)
#   CMAP_OUTPUT_ROOT  destination root (default: <CMAP_ROOT>/output)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SEG_ROOT="${SEG_OUTPUT_ROOT:-$CMAP_ROOT/segmentation_multiscale_cellpose_3D/output}"
OUT_ROOT="${CMAP_OUTPUT_ROOT:-$CMAP_ROOT/output}"

if [[ ! -d "$SEG_ROOT" ]]; then
  echo "ERROR: SEG_ROOT not a directory: $SEG_ROOT" >&2
  exit 1
fi

SEG_ROOT="$(cd "$SEG_ROOT" && pwd)"
OUT_ROOT="$(mkdir -p "$OUT_ROOT" && cd "$OUT_ROOT" && pwd)"

echo "Source:      $SEG_ROOT"
echo "Destination: $OUT_ROOT"
echo ""

linked=0
skipped=0
failed=0

while IFS= read -r -d '' fz_dir; do
  rel="${fz_dir#"$SEG_ROOT"/}"
  batch_sample="${rel%/cell_boxing_full_z}"
  [[ "$batch_sample" == "$rel" ]] && continue

  while IFS= read -r -d '' src; do
    base="$(basename "$src")"
    dst_dir="$OUT_ROOT/$batch_sample/cell_boxing_full_z"
    dst="$dst_dir/$base"

    mkdir -p "$dst_dir"

    if [[ -L "$dst" ]]; then
      cur="$(readlink "$dst" || true)"
      if [[ "$cur" == "$src" ]]; then
        skipped=$((skipped + 1))
        continue
      fi
      rm -f "$dst"
    elif [[ -e "$dst" ]]; then
      echo "SKIP (not symlink): $dst" >&2
      skipped=$((skipped + 1))
      continue
    fi

    if ln -s "$src" "$dst"; then
      linked=$((linked + 1))
    else
      echo "FAIL: ln -s $src $dst" >&2
      failed=$((failed + 1))
    fi
  done < <(find "$fz_dir" -maxdepth 1 -type f -print0 2>/dev/null)
done < <(find "$SEG_ROOT" -type d -name cell_boxing_full_z -print0)

echo "Symlinks created: $linked"
echo "Skipped (already OK or blocked): $skipped"
echo "Failed: $failed"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
