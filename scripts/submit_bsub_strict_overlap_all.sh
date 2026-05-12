#!/usr/bin/env bash
# Roll out the strict-overlap merge (gates 1+2+3, strict-1-1 default) to every
# sample under the configured output bases. Submits one CPU job per sample
# (standard queue). Writes:
#   <sample>/union_488_560_strict_overlap.tif
#   <sample>/union_488_560_strict_overlap_manifest.json
#   <sample>/union_488_560_strict_overlap_label_map.csv
#
# Inputs required per sample:
#   <sample>/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif
#   <sample>/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif
#
# Usage:
#   bash scripts/submit_bsub_strict_overlap_all.sh
#
# Optional environment overrides:
#   CMAP_REPO_ROOT          — cmap repo (absolute)
#   CMAP_STRICT_DATASETS    — space-separated dataset folder names under output/
#                             (default: "4_18_25 4_24_25_CGN_6_10_2")
#   CMAP_STRICT_TAU         — IoMin threshold (default: 0.2)
#   CMAP_STRICT_MIN_VOX     — min_label_voxels (default: 0)
#   CMAP_STRICT_EXTRA_ARGS  — extra flags forwarded to union_488_560_mask.py
#                              (e.g. "--allow-multimerge" or "--closure global")
#   CMAP_STRICT_FORCE       — set to 1 to add --force (default: skip samples that
#                              already have union_488_560_strict_overlap.tif)
#   CMAP_SUBMIT_LIMIT       — if set, only submit N jobs total (debug)
#   CMAP_SAMPLE_FILTER      — substring filter on sample names
#
# No email flags per site convention (no -B / -N).

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"

UNION_PY="$PROJECT_ROOT/postprocess/union_488_560_mask.py"
if [[ ! -f "$UNION_PY" ]]; then
  echo "ERROR: missing $UNION_PY" >&2
  exit 1
fi

DATASETS="${CMAP_STRICT_DATASETS:-4_18_25 4_24_25_CGN_6_10_2}"
TAU="${CMAP_STRICT_TAU:-0.2}"
MIN_VOX="${CMAP_STRICT_MIN_VOX:-0}"
EXTRA_ARGS="${CMAP_STRICT_EXTRA_ARGS:-}"
SAMPLE_FILTER="${CMAP_SAMPLE_FILTER:-}"

FORCE_FLAG=""
if [[ "${CMAP_STRICT_FORCE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

LOG_BASE="$PROJECT_ROOT/logs/strict_overlap_rollout"
mkdir -p "$LOG_BASE"

n_sub=0
n_skip_done=0
n_skip_inputs=0
n_skip_no_outdir=0

for dataset in $DATASETS; do
  OUT_BASE="$PROJECT_ROOT/output/$dataset"
  if [[ ! -d "$OUT_BASE" ]]; then
    echo "WARN: output base not found, skipping: $OUT_BASE" >&2
    n_skip_no_outdir=$((n_skip_no_outdir + 1))
    continue
  fi

  LOG_DIR="$LOG_BASE/$dataset"
  mkdir -p "$LOG_DIR"

  echo "==> Dataset: $dataset"

  for od in "$OUT_BASE"/*; do
    [[ -d "$od" ]] || continue

    sam=$(basename "$od")
    if [[ "$sam" == _* ]]; then
      continue
    fi
    if [[ -n "$SAMPLE_FILTER" && "$sam" != *"$SAMPLE_FILTER"* ]]; then
      continue
    fi

    m488="$od/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif"
    m560="$od/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif"
    if [[ ! -f "$m488" || ! -f "$m560" ]]; then
      n_skip_inputs=$((n_skip_inputs + 1))
      continue
    fi

    if [[ -z "$FORCE_FLAG" && -f "$od/union_488_560_strict_overlap.tif" ]]; then
      n_skip_done=$((n_skip_done + 1))
      continue
    fi

    if [[ -n "${CMAP_SUBMIT_LIMIT:-}" ]] && [[ "$n_sub" -ge "${CMAP_SUBMIT_LIMIT}" ]]; then
      echo "Stopping after CMAP_SUBMIT_LIMIT=$CMAP_SUBMIT_LIMIT"
      break 2
    fi

    echo "  Submit: $sam"
    bsub \
      -J "strict_${sam}" \
      -q standard \
      -n 1 \
      -R "rusage[mem=48000] span[hosts=1]" \
      -W 60 \
      -o "$LOG_DIR/${sam}_%J.out" \
      -e "$LOG_DIR/${sam}_%J.err" \
      bash -lc \
      "set -euo pipefail
       eval \"\$(conda shell.bash hook)\"
       conda activate cmap
       cd \"$PROJECT_ROOT\"
       exec python3 -u \"$UNION_PY\" \
         --output-dir \"$od\" \
         --skip-combined \
         --tau $TAU \
         --min-label-voxels $MIN_VOX \
         --closure component \
         $FORCE_FLAG \
         $EXTRA_ARGS"

    n_sub=$((n_sub + 1))
  done
done

echo ""
echo "Submitted: $n_sub job(s)."
echo "  Skipped (already have union_488_560_strict_overlap.tif): $n_skip_done"
echo "  Skipped (missing 488/560 indexed masks):                 $n_skip_inputs"
echo "  Skipped (missing output base dir):                       $n_skip_no_outdir"
echo "Log base: $LOG_BASE/<dataset>/"
echo "Monitor:  bjobs -J 'strict_*'  |  tail of *.out files in log dir"
