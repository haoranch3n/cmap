#!/usr/bin/env bash
# Submit a single LSF CPU job (queue=standard, no email) that pools the
# bg_sigma_488560_shape DINOv2 embeddings across all samples and runs the UMAP
# grid-search sweep:
#
#   visualization/dinov2_visualize.py        --variant union_488_560 \
#                                             --cell-qc-dir cell_qc_bg_sigma_488560_shape \
#                                             --master-npy-name dinov2_embeddings_bg_sigma_488560_shape_all.npy \
#                                             --master-csv-name dinov2_embedding_bg_sigma_488560_shape_all.csv
#   visualization/dinov2_umap_grid_search.py --variant bg_sigma_488560_shape --run-all-local
#   visualization/dinov2_umap_grid_search.py --variant bg_sigma_488560_shape --publish-best
#
# Output under <output>/cell_qc_all/:
#   dinov2_embedding_bg_sigma_488560_shape_all.csv
#   dinov2_embeddings_bg_sigma_488560_shape_all.npy
#   umap_dinov2_bg_sigma_488560_shape_sweep/
#
# Usage (from anywhere):
#   bash <repo>/scripts/submit_bsub_bg_sigma_488560_pool.sh
#
# Optional environment overrides:
#   CMAP_REPO_ROOT      — cmap repo (absolute)
#   CMAP_OUTPUT_ROOT    — pipeline output root (default: $CMAP_REPO_ROOT/output)
#   CMAP_QUEUE          — LSF queue (default: standard)
#   CMAP_WALLTIME       — bsub -W minutes (default: 720)
#   CMAP_MEM_MB         — rusage mem=<MB> (default: 65536)
#   CMAP_SKIP_VISUALIZE — set to 1 to skip step 1 (dinov2_visualize.py)
#   CMAP_SKIP_SWEEP     — set to 1 to skip step 2 (umap grid search)
#   CMAP_SKIP_PUBLISH   — set to 1 to skip step 3 (publish-best)

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"

OUTPUT_ROOT="${CMAP_OUTPUT_ROOT:-$PROJECT_ROOT/output}"

LOG_DIR="$PROJECT_ROOT/logs/bg_sigma_488560_pool"
mkdir -p "$LOG_DIR"

if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "ERROR: output root not found: $OUTPUT_ROOT" >&2
  exit 1
fi

VIS_PY="$PROJECT_ROOT/visualization/dinov2_visualize.py"
UMAP_PY="$PROJECT_ROOT/visualization/dinov2_umap_grid_search.py"
for f in "$VIS_PY" "$UMAP_PY"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

QUEUE="${CMAP_QUEUE:-standard}"
WALLTIME="${CMAP_WALLTIME:-720}"
MEM_MB="${CMAP_MEM_MB:-65536}"

SKIP_VISUALIZE="${CMAP_SKIP_VISUALIZE:-0}"
SKIP_SWEEP="${CMAP_SKIP_SWEEP:-0}"
SKIP_PUBLISH="${CMAP_SKIP_PUBLISH:-0}"

JOB_NAME="bgs488560pool"

echo "Pool job: queue=$QUEUE walltime=$WALLTIME mem=${MEM_MB}MB output_root=$OUTPUT_ROOT"
bsub \
  -J "$JOB_NAME" \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=${MEM_MB}] span[hosts=1]" \
  -W "$WALLTIME" \
  -o "$LOG_DIR/${JOB_NAME}_%J.out" \
  -e "$LOG_DIR/${JOB_NAME}_%J.err" \
  bash -lc \
  "set -euo pipefail
   eval \"\$(conda shell.bash hook)\"
   conda activate cmap
   cd \"$PROJECT_ROOT\"

   if [[ \"$SKIP_VISUALIZE\" != 1 ]]; then
     echo === step 1/3 dinov2_visualize variant=bg_sigma_488560_shape ===
     python3 -u \"$VIS_PY\" \
       --variant union_488_560 \
       --output-root \"$OUTPUT_ROOT\" \
       --cell-qc-dir cell_qc_bg_sigma_488560_shape \
       --master-npy-name dinov2_embeddings_bg_sigma_488560_shape_all.npy \
       --master-csv-name dinov2_embedding_bg_sigma_488560_shape_all.csv
   else
     echo SKIP step 1/3 dinov2_visualize CMAP_SKIP_VISUALIZE=1
   fi

   if [[ \"$SKIP_SWEEP\" != 1 ]]; then
     echo === step 2/3 dinov2_umap_grid_search --run-all-local variant=bg_sigma_488560_shape ===
     python3 -u \"$UMAP_PY\" \
       --variant bg_sigma_488560_shape \
       --output-root \"$OUTPUT_ROOT\" \
       --run-all-local
   else
     echo SKIP step 2/3 dinov2_umap_grid_search --run-all-local CMAP_SKIP_SWEEP=1
   fi

   if [[ \"$SKIP_PUBLISH\" != 1 ]]; then
     echo === step 3/3 dinov2_umap_grid_search --publish-best variant=bg_sigma_488560_shape ===
     python3 -u \"$UMAP_PY\" \
       --variant bg_sigma_488560_shape \
       --output-root \"$OUTPUT_ROOT\" \
       --publish-best
   else
     echo SKIP step 3/3 --publish-best CMAP_SKIP_PUBLISH=1
   fi"

echo ""
echo "Submitted pool job: $JOB_NAME"
echo "Logs: $LOG_DIR"
echo "Monitor: bjobs -J '$JOB_NAME'"
