#!/usr/bin/env bash
# Submit one LSF job per sample (standard queue) to write filtered_488.tif and
# filtered_560.tif via postprocess/filter_642_mask.py --export-all-anchors.
# Uses --skip-combined so originals are not re-read for the large OME stack.
#
# Defaults target the 4_24_25_CGN_6_10_2 legacy output tree and Gutierrez
# CMAP_cropped_copies inputs (same layout as pipeline_full_run.log).
#
# Usage (from anywhere):
#   bash /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/scripts/submit_bsub_filter_all_anchors_4_24_25.sh
#
# Optional environment overrides:
#   CMAP_REPO_ROOT     — cmap repo (absolute)
#   CMAP_OUTPUT_BASE   — directory containing per-sample folders
#   CMAP_DATA_BASE     — parallel tree with same sample folder names (originals)
#   CMAP_SUBMIT_LIMIT  — if set (integer), only submit this many jobs (debug)
#
# No LSF email flags per site convention.

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"

OUT_BASE="${CMAP_OUTPUT_BASE:-$PROJECT_ROOT/segmentation_multiscale_cellpose_3D/output/4_24_25_CGN_6_10_2}"
DATA_BASE="${CMAP_DATA_BASE:-/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_cropped_copies/4_24_25_CGN_6_10_2}"

LOG_DIR="$PROJECT_ROOT/logs/filter_all_anchors_4_24_25"
mkdir -p "$LOG_DIR"

if [[ ! -d "$OUT_BASE" ]]; then
  echo "ERROR: output base not found: $OUT_BASE" >&2
  exit 1
fi
if [[ ! -d "$DATA_BASE" ]]; then
  echo "ERROR: data base not found: $DATA_BASE" >&2
  exit 1
fi

FILTER_PY="$PROJECT_ROOT/postprocess/filter_642_mask.py"
if [[ ! -f "$FILTER_PY" ]]; then
  echo "ERROR: missing $FILTER_PY" >&2
  exit 1
fi

n_sub=0
for od in "$OUT_BASE"/*; do
  [[ -d "$od" ]] || continue
  [[ -f "$od/filtered_642.tif" ]] || continue

  sam=$(basename "$od")
  dd="$DATA_BASE/$sam"
  if [[ ! -d "$dd" ]]; then
    echo "SKIP (no data dir): $sam"
    continue
  fi

  if [[ -n "${CMAP_SUBMIT_LIMIT:-}" ]] && [[ "$n_sub" -ge "${CMAP_SUBMIT_LIMIT}" ]]; then
    echo "Stopping after CMAP_SUBMIT_LIMIT=$CMAP_SUBMIT_LIMIT"
    break
  fi

  echo "Submit: $sam"
  bsub \
    -J "fa424_${sam}" \
    -q standard \
    -n 1 \
    -R "rusage[mem=32000] span[hosts=1]" \
    -W 240 \
    -o "$LOG_DIR/${sam}_%J.out" \
    -e "$LOG_DIR/${sam}_%J.err" \
    bash -lc \
    "set -euo pipefail
     eval \"\$(conda shell.bash hook)\"
     conda activate cmap
     cd \"$PROJECT_ROOT\"
     exec python3 -u \"$FILTER_PY\" \
       --data-dir \"$dd\" \
       --output-dir \"$od\" \
       --export-all-anchors \
       --skip-combined"

  n_sub=$((n_sub + 1))
done

echo "Submitted $n_sub job(s). Logs: $LOG_DIR"
echo "Monitor: bjobs -J 'fa424_*'"
