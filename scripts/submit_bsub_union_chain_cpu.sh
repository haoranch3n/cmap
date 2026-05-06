#!/usr/bin/env bash
# Submit one LSF job per sample (standard queue, CPU) to run the union_488_560
# CPU chain after union mask + combined exist on disk:
#
#   qc/filter_by_intensity.py           --variant union_488_560 --from-volume
#   qc/apply_qc_pass_to_label_mask.py   --variant union_488_560
#   features/crop_cells.py              --variant union_488_560
#     --label-mask-name union_488_560_otsu_shape.tif --output-subdir-key cell_box_filtered
#   features/extract_features.py        --variant union_488_560 --cell-box-subdir-key cell_box_filtered
#   qc/merge_qc_features_filtered.py    --variant union_488_560
#
# DINOv2 volume bounds + embedding extraction are **not** here — run
# scripts/submit_bsub_union_dinov2_gpu.sh (GPU) after this chain completes.
#
# Coexists with filtered_642 outputs; legacy cell_boxing_union_488_560/ is not
# written by this chain (crops go to cell_box_otsu_shape_filtered/).
#
# Usage (from anywhere):
#   bash <repo>/scripts/submit_bsub_union_chain_cpu.sh
#
# Optional environment overrides:
#   CMAP_REPO_ROOT          — cmap repo (absolute)
#   CMAP_OUTPUT_BASE        — directory containing per-sample folders
#                              (default: <repo>/output/4_24_25_CGN_6_10_2)
#   CMAP_DATASET            — dataset short name used in -J / log dir
#                              (default: derived from $CMAP_OUTPUT_BASE basename)
#   CMAP_SAMPLE_FILTER      — substring filter on sample names
#   CMAP_SUBMIT_LIMIT       — only submit this many jobs (debug)
#   CMAP_FORCE_OVERWRITE    — set to 1 to add --force / --overwrite (default: skip)
#   CMAP_QUEUE              — LSF queue (default: standard)
#   CMAP_WALLTIME           — bsub -W minutes (default: 240)
#   CMAP_MEM_MB             — rusage mem=<MB> (default: 32000)
#
# No LSF email flags per site convention (no -B / -N).

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"

OUT_BASE="${CMAP_OUTPUT_BASE:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2}"
DATASET="${CMAP_DATASET:-$(basename "$OUT_BASE")}"

LOG_DIR="$PROJECT_ROOT/logs/union_chain_cpu_${DATASET}"
mkdir -p "$LOG_DIR"

if [[ ! -d "$OUT_BASE" ]]; then
  echo "ERROR: output base not found: $OUT_BASE" >&2
  exit 1
fi

QUEUE="${CMAP_QUEUE:-standard}"
WALLTIME="${CMAP_WALLTIME:-240}"
MEM_MB="${CMAP_MEM_MB:-32000}"

FORCE_FLAG=""
OVERWRITE_FLAG=""
if [[ "${CMAP_FORCE_OVERWRITE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
  OVERWRITE_FLAG="--overwrite"
fi

SAMPLE_FILTER="${CMAP_SAMPLE_FILTER:-}"

for rel in \
  qc/filter_by_intensity.py \
  qc/apply_qc_pass_to_label_mask.py \
  qc/merge_qc_features_filtered.py \
  features/crop_cells.py \
  features/extract_features.py; do
  if [[ ! -f "$PROJECT_ROOT/$rel" ]]; then
    echo "ERROR: missing $PROJECT_ROOT/$rel" >&2
    exit 1
  fi
done

n_sub=0
n_skip_inputs=0
n_skip_done=0
for od in "$OUT_BASE"/*; do
  [[ -d "$od" ]] || continue

  sam=$(basename "$od")
  # Ignore housekeeping dirs under the dataset root (e.g. _lsf_logs_*).
  if [[ "$sam" == _* ]]; then
    continue
  fi

  if [[ -n "$SAMPLE_FILTER" && "$sam" != *"$SAMPLE_FILTER"* ]]; then
    continue
  fi

  if [[ ! -f "$od/union_488_560.tif" || ! -f "$od/union_488_560_combined.tif" ]]; then
    n_skip_inputs=$((n_skip_inputs + 1))
    continue
  fi

  if [[ -z "$FORCE_FLAG" \
        && -d "$od/cell_box_otsu_shape_filtered" \
        && -f "$od/cell_qc_union_488_560/qc_features_filtered.csv" \
        && -f "$od/union_488_560_otsu_shape.tif" ]]; then
    n_skip_done=$((n_skip_done + 1))
    continue
  fi

  if [[ -n "${CMAP_SUBMIT_LIMIT:-}" ]] && [[ "$n_sub" -ge "${CMAP_SUBMIT_LIMIT}" ]]; then
    echo "Stopping after CMAP_SUBMIT_LIMIT=$CMAP_SUBMIT_LIMIT"
    break
  fi

  echo "Submit: $sam"
  bsub \
    -J "u488560chain_${sam}" \
    -q "$QUEUE" \
    -n 1 \
    -R "rusage[mem=${MEM_MB}] span[hosts=1]" \
    -W "$WALLTIME" \
    -o "$LOG_DIR/${sam}_%J.out" \
    -e "$LOG_DIR/${sam}_%J.err" \
    bash -lc \
    "set -euo pipefail
     eval \"\$(conda shell.bash hook)\"
     conda activate cmap
     cd \"$PROJECT_ROOT\"

     echo === step 1/5 qc.filter_by_intensity --from-volume variant=union_488_560 ===
     python3 -u qc/filter_by_intensity.py \
       --variant union_488_560 \
       --from-volume \
       --output-dir \"$od\" \
       $FORCE_FLAG

     echo === step 2/5 qc.apply_qc_pass_to_label_mask variant=union_488_560 ===
     python3 -u qc/apply_qc_pass_to_label_mask.py \
       --variant union_488_560 \
       --output-dir \"$od\" \
       $OVERWRITE_FLAG

     echo === step 3/5 crop_cells variant=union_488_560 filtered ===
     python3 -u features/crop_cells.py \
       --variant union_488_560 \
       --output-dir \"$od\" \
       --data-dir \"$od\" \
       --label-mask-name union_488_560_otsu_shape.tif \
       --output-subdir-key cell_box_filtered \
       --also-full-z \
       $FORCE_FLAG

     echo === step 4/5 extract_features variant=union_488_560 filtered ===
     python3 -u features/extract_features.py \
       --variant union_488_560 \
       --output-dir \"$od\" \
       --cell-box-subdir-key cell_box_filtered \
       $FORCE_FLAG

     echo === step 5/5 qc.merge_qc_features_filtered variant=union_488_560 ===
     python3 -u qc/merge_qc_features_filtered.py \
       --variant union_488_560 \
       --output-dir \"$od\""

  n_sub=$((n_sub + 1))
done

echo ""
echo "Submitted $n_sub job(s)."
echo "  Skipped (chain artefacts already present): $n_skip_done"
echo "  Skipped (missing union_488_560*.tif):       $n_skip_inputs"
echo "Logs: $LOG_DIR"
echo "Monitor: bjobs -J 'u488560chain_*'"
