#!/usr/bin/env bash
# Print sample names (basename of each output child that has cell_boxing/).
# Usage:
#   PIPELINE_OUTPUT_DIR=/path/to/output ./discover_samples.sh
#   ./discover_samples.sh /path/to/output

set -euo pipefail
ROOT="${1:-${PIPELINE_OUTPUT_DIR:-}}"
if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
  echo "Usage: PIPELINE_OUTPUT_DIR=/path/to/output $0" >&2
  echo "   or: $0 /path/to/output" >&2
  exit 1
fi
ROOT="${ROOT%/}"

shopt -s nullglob
for d in "$ROOT"/*/; do
  [[ -d "${d}cell_boxing" ]] || continue
  n="$(find "${d}cell_boxing" -maxdepth 1 -name 'cell_*.tif' -print -quit 2>/dev/null || true)"
  [[ -n "$n" ]] || continue
  basename "$d"
done | sort -u
