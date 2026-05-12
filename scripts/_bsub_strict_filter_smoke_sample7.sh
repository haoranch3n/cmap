#!/usr/bin/env bash
# Smoke test: wire the QC bg_sigma:3 (488+560 gating) filter to the new
# strict-overlap union mask on Sample7_Position7_decon_dsr (4_24_25_CGN_6_10_2).
#
# Writes to a side cell_qc directory and a *_smoke output TIFF so legacy
# artifacts under the sample are not overwritten. Prints before/after cell
# counts at the end of the worker log.
#
# Inner worker:
#   scripts/_strict_filter_smoke_sample7_inner.sh

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
INNER="$PROJECT_ROOT/scripts/_strict_filter_smoke_sample7_inner.sh"

LOG_DIR="$PROJECT_ROOT/logs/strict_filter_smoke"
mkdir -p "$LOG_DIR"

bsub \
  -J "strict_filter_smoke_S7P7" \
  -q standard \
  -n 1 \
  -R "rusage[mem=48000] span[hosts=1]" \
  -W 60 \
  -o "$LOG_DIR/sample7_%J.out" \
  -e "$LOG_DIR/sample7_%J.err" \
  bash "$INNER"

echo "Submitted strict-filter smoke job for Sample7_Position7_decon_dsr"
echo "Log dir: $LOG_DIR"
