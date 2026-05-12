#!/usr/bin/env bash
# Dry-run: strict-overlap merge on a single sample (Sample7_Position7) to
# preview gate counts before committing the rename/registry changes. Writes
# the new union_488_560_strict_overlap.tif alongside the legacy
# union_488_560.tif (no overwrite). Skips --combined output.
set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="$PROJECT_ROOT/output/4_24_25_CGN_6_10_2/Sample7_Position7_decon_dsr"
LOG_DIR="$PROJECT_ROOT/logs/strict_overlap_dryrun"
mkdir -p "$LOG_DIR"

UNION_PY="$PROJECT_ROOT/postprocess/union_488_560_mask.py"

bsub \
  -J "strict_dryrun_S7P7" \
  -q standard \
  -n 1 \
  -R "rusage[mem=48000] span[hosts=1]" \
  -W 60 \
  -o "$LOG_DIR/sample7_position7_%J.out" \
  -e "$LOG_DIR/sample7_position7_%J.err" \
  bash -lc "set -euo pipefail
   eval \"\$(conda shell.bash hook)\"
   conda activate cmap
   cd \"$PROJECT_ROOT\"
   exec python3 -u \"$UNION_PY\" \
     --output-dir \"$SAMPLE_DIR\" \
     --skip-combined \
     --tau 0.2 \
     --min-label-voxels 0 \
     --closure component \
     --force"

echo "Submitted dry-run."
echo "Log: $LOG_DIR/sample7_position7_<JOBID>.out"
echo "Monitor: bjobs -J strict_dryrun_S7P7"
