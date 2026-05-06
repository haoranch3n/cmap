#!/usr/bin/env bash
# Compute dinov2_volume_norm_bounds.csv for one sample (CPU + RAM).
#
# Usage (from anywhere; uses absolute CMAP_ROOT):
#   export CMAP_ROOT=/path/to/cmap
#   bash "$CMAP_ROOT/scripts/batch_dinov2_volume_norm/run_one_sample.sh" 4_18_25/CGNSample1_Position0_decon_dsr
#
# If the CSV already exists, exits 0 immediately unless FORCE_VOLUME_NORM=1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SAMPLE_REL="${1:?sample relpath under output/, e.g. 4_18_25/CGNSample1_Position0_decon_dsr}"

OUT_DIR="${CMAP_ROOT}/output/${SAMPLE_REL}"
COMBINED="${OUT_DIR}/filtered_642_combined.tif"
CSV="${OUT_DIR}/dinov2_volume_norm_bounds.csv"

if [[ ! -f "$COMBINED" ]]; then
  echo "ERROR: missing $COMBINED" >&2
  exit 1
fi

if [[ -f "$CSV" && "${FORCE_VOLUME_NORM:-0}" != "1" ]]; then
  echo "SKIP (exists): $CSV"
  exit 0
fi

echo "OUT_DIR=$OUT_DIR"
cd "$CMAP_ROOT"
exec python features/compute_dinov2_volume_norm_csv.py --output-dir "$OUT_DIR"
