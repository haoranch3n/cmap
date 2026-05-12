#!/usr/bin/env bash
# Submit one GPU job per sample (95) for no-deconv Cellpose on registered
# 488 / 560 / 642 volumes. Writes ONLY under:
#   ${CMAP_NO_DECONV_OUTPUT_BASE}/<dataset>/<Sample_Position>/...
# and never touches the default output/ tree.
#
# Input layout (read-only):
#   ${CMAP_NO_DECONV_INPUT_BASE}/4_18_25/<sample>/registered/*.tif
#   ${CMAP_NO_DECONV_INPUT_BASE}/4_24_25_CGN_6_10_2/<sample>/registered/*.tif
#
# Usage (from login node, repo root):
#   bash scripts/no_deconv/submit_bsub_seg_registered_all.sh
#
# Environment overrides:
#   CMAP_REPO_ROOT              — cmap repo (absolute)
#   CMAP_NO_DECONV_INPUT_BASE   — rough_registration_batch parent (absolute)
#   CMAP_NO_DECONV_OUTPUT_BASE  — default: $CMAP_REPO_ROOT/output_no_deconv
#   CMAP_NO_DECONV_DATASETS     — space-separated (default: 4_18_25 4_24_25_CGN_6_10_2)
#   CMAP_NO_DECONV_FORCE        — set to 1 to resubmit even if 3D indexed masks exist
#   CMAP_SUBMIT_LIMIT           — max jobs to submit (debug)
#   CMAP_SAMPLE_FILTER          — substring filter on sample dir name
#
# Queue: rhel88_gpu (Cellpose). No email flags.

set -euo pipefail

PROJECT_ROOT="${CMAP_REPO_ROOT:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

INPUT_BASE="${CMAP_NO_DECONV_INPUT_BASE:-/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_general/No_decon_tests/outputs/rough_registration_batch}"
INPUT_BASE="$(cd "$INPUT_BASE" && pwd)"

OUTPUT_BASE="${CMAP_NO_DECONV_OUTPUT_BASE:-$PROJECT_ROOT/output_no_deconv}"
mkdir -p "$OUTPUT_BASE"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"

DATASETS="${CMAP_NO_DECONV_DATASETS:-4_18_25 4_24_25_CGN_6_10_2}"
SAMPLE_FILTER="${CMAP_SAMPLE_FILTER:-}"
FORCE="${CMAP_NO_DECONV_FORCE:-0}"

INNER="${PROJECT_ROOT}/scripts/no_deconv/_seg_registered_one_sample_inner.sh"
if [[ ! -f "$INNER" ]]; then
  echo "ERROR: missing $INNER" >&2
  exit 1
fi
chmod +x "$INNER" 2>/dev/null || true

LOG_BASE="${PROJECT_ROOT}/logs/no_deconv_seg_registered"
mkdir -p "$LOG_BASE"

n_sub=0
n_skip_done=0
n_skip_inputs=0

for dataset in $DATASETS; do
  IN_DS="${INPUT_BASE}/${dataset}"
  if [[ ! -d "$IN_DS" ]]; then
    echo "WARN: missing input dataset dir: $IN_DS" >&2
    continue
  fi
  LOG_DIR="${LOG_BASE}/${dataset}"
  mkdir -p "$LOG_DIR"
  echo "==> Dataset: $dataset"

  for pos in "${IN_DS}"/*; do
    [[ -d "$pos" ]] || continue
    sam=$(basename "$pos")
    if [[ "$sam" == _* ]]; then
      continue
    fi
    if [[ -n "$SAMPLE_FILTER" && "$sam" != *"$SAMPLE_FILTER"* ]]; then
      continue
    fi

    reg="${pos}/registered"
    if [[ ! -d "$reg" ]]; then
      echo "  SKIP $sam: no registered/"
      n_skip_inputs=$((n_skip_inputs + 1))
      continue
    fi
    if [[ ! -f "${reg}/488nm_registered.tif" || ! -f "${reg}/560nm_registered.tif" || ! -f "${reg}/642nm_reference.tif" ]]; then
      echo "  SKIP $sam: missing one of 488nm_registered.tif / 560nm_registered.tif / 642nm_reference.tif"
      n_skip_inputs=$((n_skip_inputs + 1))
      continue
    fi

    MERGE="${OUTPUT_BASE}/${dataset}/${sam}"
    if [[ "$FORCE" != "1" \
      && -f "${MERGE}/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif" \
      && -f "${MERGE}/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif" \
      && -f "${MERGE}/642nm_crop/segmentation_3D_masks/642nm_crop_3D_indexed.tif" ]]; then
      n_skip_done=$((n_skip_done + 1))
      continue
    fi

    if [[ -n "${CMAP_SUBMIT_LIMIT:-}" ]] && [[ "$n_sub" -ge "${CMAP_SUBMIT_LIMIT}" ]]; then
      echo "Stopping after CMAP_SUBMIT_LIMIT=${CMAP_SUBMIT_LIMIT}"
      break 2
    fi

    job="ndseg_${dataset}_${sam}"
    job="${job//[^A-Za-z0-9_]/_}"

    echo "  Submit: $dataset/$sam -> $MERGE"
    bsub \
      -J "$job" \
      -q rhel88_gpu \
      -gpu "num=1:j_exclusive=yes" \
      -n 1 \
      -R "rusage[mem=65536] span[hosts=1]" \
      -W 1440 \
      -o "${LOG_DIR}/${sam}_%J.out" \
      -e "${LOG_DIR}/${sam}_%J.err" \
      bash -lc "set -euo pipefail
       eval \"\$(conda shell.bash hook)\"
       conda activate cmap
       export PROJECT_ROOT=\"${PROJECT_ROOT}\"
       export NO_DECONV_REGISTERED=\"${reg}\"
       export NO_DECONV_MERGE=\"${MERGE}\"
       exec bash \"${INNER}\""

    n_sub=$((n_sub + 1))
  done
done

echo ""
echo "Submitted: $n_sub GPU job(s) (rhel88_gpu)."
echo "  Skipped (all three *_3D_indexed.tif already present): $n_skip_done"
echo "  Skipped (missing inputs):                        $n_skip_inputs"
echo "Output base: $OUTPUT_BASE"
echo "Logs:        $LOG_BASE/<dataset>/"
echo "Monitor:     bjobs -J 'ndseg_*'"
