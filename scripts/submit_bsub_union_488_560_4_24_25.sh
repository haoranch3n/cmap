#!/usr/bin/env bash
# Submit one LSF job per sample (standard queue) to write union_488_560.tif and
# union_488_560_combined.tif via postprocess/union_488_560_mask.py.
#
# Defaults target the 4_24_25_CGN_6_10_2 output tree (under <repo>/output/) and
# Gutierrez CMAP_cropped_copies inputs (same layout used by the existing
# submit_bsub_filter_all_anchors_4_24_25.sh).
#
# Usage (from anywhere):
#   bash /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/scripts/submit_bsub_union_488_560_4_24_25.sh
#
# Optional environment overrides:
#   CMAP_REPO_ROOT          — cmap repo (absolute)
#   CMAP_OUTPUT_BASE        — directory containing per-sample folders (default: <repo>/output/4_24_25_CGN_6_10_2)
#   CMAP_DATA_BASE          — parallel tree with same sample folder names (originals)
#   CMAP_SUBMIT_LIMIT       — if set (integer), only submit this many jobs (debug)
#   CMAP_SAMPLE_FILTER      — substring filter on sample names (e.g. "Sample10_Position0")
#   CMAP_UNION_EXTRA_ARGS   — extra flags forwarded to union_488_560_mask.py
#                              (e.g. "--require-both-channels --skip-combined")
#   CMAP_FORCE_OVERWRITE    — set to 1 to add --force (default: skip samples that
#                              already have union_488_560.tif)
#
# No LSF email flags per site convention (no -B / -N).

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"

OUT_BASE="${CMAP_OUTPUT_BASE:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2}"
DATA_BASE="${CMAP_DATA_BASE:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_cropped_copies/4_24_25_CGN_6_10_2}"
DATASET="${CMAP_DATASET:-$(basename "$OUT_BASE")}"

LOG_DIR="$PROJECT_ROOT/logs/union_488_560_${DATASET}"
mkdir -p "$LOG_DIR"

if [[ ! -d "$OUT_BASE" ]]; then
  echo "ERROR: output base not found: $OUT_BASE" >&2
  exit 1
fi
if [[ ! -d "$DATA_BASE" ]]; then
  echo "WARNING: data base not found: $DATA_BASE (combined.tif will be skipped per sample)" >&2
fi

UNION_PY="$PROJECT_ROOT/postprocess/union_488_560_mask.py"
if [[ ! -f "$UNION_PY" ]]; then
  echo "ERROR: missing $UNION_PY" >&2
  exit 1
fi

EXTRA_ARGS="${CMAP_UNION_EXTRA_ARGS:-}"
FORCE_FLAG=""
if [[ "${CMAP_FORCE_OVERWRITE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

SAMPLE_FILTER="${CMAP_SAMPLE_FILTER:-}"

n_sub=0
n_skip_done=0
n_skip_inputs=0
for od in "$OUT_BASE"/*; do
  [[ -d "$od" ]] || continue

  sam=$(basename "$od")
  if [[ "$sam" == _* ]]; then
    continue
  fi

  if [[ -n "$SAMPLE_FILTER" && "$sam" != *"$SAMPLE_FILTER"* ]]; then
    continue
  fi

  # Need both per-channel indexed masks present.
  m488="$od/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif"
  m560="$od/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif"
  if [[ ! -f "$m488" || ! -f "$m560" ]]; then
    n_skip_inputs=$((n_skip_inputs + 1))
    continue
  fi

  # Skip if already produced (unless force).
  if [[ -z "$FORCE_FLAG" && -f "$od/union_488_560.tif" ]]; then
    n_skip_done=$((n_skip_done + 1))
    continue
  fi

  # Resolve data dir for combined.tif (optional; --skip-combined-friendly).
  dd="$DATA_BASE/$sam"
  data_arg=""
  if [[ -d "$dd" ]]; then
    data_arg="--data-dir \"$dd\""
  fi

  if [[ -n "${CMAP_SUBMIT_LIMIT:-}" ]] && [[ "$n_sub" -ge "${CMAP_SUBMIT_LIMIT}" ]]; then
    echo "Stopping after CMAP_SUBMIT_LIMIT=$CMAP_SUBMIT_LIMIT"
    break
  fi

  echo "Submit: $sam"
  bsub \
    -J "u488560_${sam}" \
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
     exec python3 -u \"$UNION_PY\" \
       --output-dir \"$od\" \
       $data_arg \
       $FORCE_FLAG \
       $EXTRA_ARGS"

  n_sub=$((n_sub + 1))
done

echo ""
echo "Submitted $n_sub job(s)."
echo "  Skipped (already have union_488_560.tif): $n_skip_done"
echo "  Skipped (missing 488/560 indexed masks):  $n_skip_inputs"
echo "Logs: $LOG_DIR"
echo "Monitor: bjobs -J 'u488560_*'"
