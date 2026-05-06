#!/usr/bin/env bash
# Submit one LSF GPU job per sample for union_488_560 DINOv2:
#
#   1) features/compute_dinov2_volume_norm_csv.py --variant union_488_560
#      (skipped if bounds CSV already exists and CMAP_FORCE_OVERWRITE is not 1)
#   2) features/extract_dinov2_embeddings.py --variant union_488_560 \
#        --norm-scope volume --extraction-mode orthogonal_concat --empty-frac 0 \
#        --device cuda --output-dir <od>
#
# Requires crops under cell_box_otsu_shape_filtered/ (CPU union chain) and
# merged cell_qc_union_488_560/qc_features_filtered.csv. Writes / refreshes
# dinov2_volume_norm_bounds_union_488_560.csv then:
#   <od>/cell_qc_union_488_560/dinov2_embeddings.npy
#   <od>/cell_qc_union_488_560/dinov2_embeddings.csv
#
# Coexists with the existing cell_qc/dinov2_embeddings.* (filtered_642).
#
# Usage (from anywhere):
#   bash <repo>/scripts/submit_bsub_union_dinov2_gpu.sh
#
# Optional environment overrides:
#   CMAP_REPO_ROOT          — cmap repo (absolute)
#   CMAP_OUTPUT_BASE        — directory containing per-sample folders
#                              (default: <repo>/output/4_24_25_CGN_6_10_2)
#   CMAP_DATASET            — dataset short name used in -J / log dir
#                              (default: derived from $CMAP_OUTPUT_BASE basename)
#   CMAP_SAMPLE_FILTER      — substring filter on sample names
#   CMAP_SUBMIT_LIMIT       — only submit this many jobs (debug)
#   CMAP_FORCE_OVERWRITE    — set to 1 to add --force (default: skip samples that
#                              already have cell_qc_union_488_560/dinov2_embeddings.npy)
#   CMAP_QUEUE              — LSF GPU queue (default: rhel88_gpu)
#   CMAP_WALLTIME           — bsub -W H:MM (default: 4:00)
#   CMAP_MEM_MB             — rusage mem=<MB> (default: 32768)
#   CMAP_DINOV2_EXTRA       — extra flags forwarded to extract_dinov2_embeddings.py
#                              (e.g. "--apply-mask")
#
# No LSF email flags per site convention (no -B / -N).

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"

OUT_BASE="${CMAP_OUTPUT_BASE:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2}"
DATASET="${CMAP_DATASET:-$(basename "$OUT_BASE")}"

LOG_DIR="$PROJECT_ROOT/logs/union_dinov2_gpu_${DATASET}"
mkdir -p "$LOG_DIR"

if [[ ! -d "$OUT_BASE" ]]; then
  echo "ERROR: output base not found: $OUT_BASE" >&2
  exit 1
fi

EXTRACT_PY="$PROJECT_ROOT/features/extract_dinov2_embeddings.py"
VOLNORM_PY="$PROJECT_ROOT/features/compute_dinov2_volume_norm_csv.py"
for f in "$EXTRACT_PY" "$VOLNORM_PY"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

QUEUE="${CMAP_QUEUE:-rhel88_gpu}"
WALLTIME="${CMAP_WALLTIME:-4:00}"
MEM_MB="${CMAP_MEM_MB:-32768}"
DINOV2_EXTRA="${CMAP_DINOV2_EXTRA:-}"

FORCE_FLAG=""
if [[ "${CMAP_FORCE_OVERWRITE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

SAMPLE_FILTER="${CMAP_SAMPLE_FILTER:-}"

n_sub=0
n_skip_inputs=0
n_skip_done=0
for od in "$OUT_BASE"/*; do
  [[ -d "$od" ]] || continue

  sam=$(basename "$od")
  if [[ "$sam" == _* ]]; then
    continue
  fi

  if [[ -n "$SAMPLE_FILTER" && "$sam" != *"$SAMPLE_FILTER"* ]]; then
    continue
  fi

  bounds_csv="$od/dinov2_volume_norm_bounds_union_488_560.csv"
  box_dir="$od/cell_box_otsu_shape_filtered"
  qc_dir="$od/cell_qc_union_488_560"
  qc_csv="$qc_dir/qc_features_filtered.csv"
  if [[ ! -d "$box_dir" || ! -f "$qc_csv" ]]; then
    n_skip_inputs=$((n_skip_inputs + 1))
    continue
  fi

  if [[ -z "$FORCE_FLAG" && -f "$qc_dir/dinov2_embeddings.npy" ]]; then
    n_skip_done=$((n_skip_done + 1))
    continue
  fi

  if [[ -n "${CMAP_SUBMIT_LIMIT:-}" ]] && [[ "$n_sub" -ge "${CMAP_SUBMIT_LIMIT}" ]]; then
    echo "Stopping after CMAP_SUBMIT_LIMIT=$CMAP_SUBMIT_LIMIT"
    break
  fi

  echo "Submit: $sam"
  bsub \
    -J "u488560dinov2_${sam}" \
    -q "$QUEUE" \
    -n 1 \
    -R "rusage[mem=${MEM_MB}] span[hosts=1]" \
    -gpu 'num=1:j_exclusive=yes' \
    -W "$WALLTIME" \
    -o "$LOG_DIR/${sam}_%J.out" \
    -e "$LOG_DIR/${sam}_%J.err" \
    bash -lc \
    "set -euo pipefail
     eval \"\$(conda shell.bash hook)\"
     conda activate cmap
     cd \"$PROJECT_ROOT\"
     _CMAP_FO=\"${CMAP_FORCE_OVERWRITE:-0}\"
     if [[ ! -f \"$bounds_csv\" || \"\$_CMAP_FO\" == \"1\" ]]; then
       echo === dinov2 prep: compute_dinov2_volume_norm_csv variant=union_488_560 ===
       python3 -u \"$VOLNORM_PY\" --variant union_488_560 --output-dir \"$od\"
     fi
     echo === extract_dinov2_embeddings variant=union_488_560 ===
     exec python3 -u \"$EXTRACT_PY\" \
       --variant union_488_560 \
       --output-dir \"$od\" \
       --cell-boxing-dir cell_box_otsu_shape_filtered \
       --norm-scope volume \
       --extraction-mode orthogonal_concat \
       --empty-frac 0 \
       --device cuda \
       $FORCE_FLAG \
       $DINOV2_EXTRA"

  n_sub=$((n_sub + 1))
done

echo ""
echo "Submitted $n_sub job(s)."
echo "  Skipped (cell_qc_union_488_560/dinov2_embeddings.npy exists): $n_skip_done"
echo "  Skipped (missing cell_box_otsu_shape_filtered/ or qc_features_filtered.csv): $n_skip_inputs"
echo "Logs: $LOG_DIR"
echo "Monitor: bjobs -J 'u488560dinov2_*'"
