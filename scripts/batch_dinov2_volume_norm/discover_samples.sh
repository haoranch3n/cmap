#!/usr/bin/env bash
# Print one sample relpath per line (under output/), same discovery order as
# scripts/compute_dinov2_volume_norm_batch.py — suitable for LSF lists.
#
# Usage:
#   cd "$CMAP_ROOT"
#   bash scripts/batch_dinov2_volume_norm/discover_samples.sh > scripts/batch_dinov2_volume_norm/samples.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUT="${CMAP_ROOT}/output"

python3 - "$OUT" <<'PY'
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
paths: list[Path] = []
skip = {"cell_qc_all", "tif_planes"}
for child in sorted(output_root.iterdir(), key=lambda p: p.name.lower()):
    if child.name in skip:
        continue
    if not child.is_dir() and not (child.is_symlink() and child.resolve().is_dir()):
        continue
    for p in child.rglob("filtered_642_combined.tif"):
        # Do not resolve(): symlinks under output/ point at real data elsewhere
        # and would break relative_to(output/).
        paths.append(p)

paths.sort(key=lambda x: str(x).lower())
for p in paths:
    rel = p.relative_to(output_root.resolve())
    print(rel.parent.as_posix())
PY
