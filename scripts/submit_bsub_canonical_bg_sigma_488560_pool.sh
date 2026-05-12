#!/usr/bin/env bash
# Submit a single LSF CPU job that pools canonical QC features (qc_features.csv)
# from cell_qc_bg_sigma_488560_shape/ across all samples, runs UMAP grid search,
# and publishes the best embedding to cell_qc_all/:
#
#   visualization/canonical_umap_grid_search.py --variant bg_sigma_488560_shape --run-all-local
#   visualization/canonical_umap_grid_search.py --variant bg_sigma_488560_shape --publish-best
#
# Output under <output>/cell_qc_all/:
#   canonical_embedding_bg_sigma_488560_shape_all.csv    ← Napari master table
#   umap_canonical_bg_sigma_488560_shape_sweep/          ← per-combo CSVs + scores
#
# Usage (from anywhere):
#   bash <repo>/scripts/submit_bsub_canonical_bg_sigma_488560_pool.sh
#
# Optional environment overrides:
#   CMAP_REPO_ROOT     — cmap repo (absolute)
#   CMAP_OUTPUT_ROOT   — pipeline output root (default: $CMAP_REPO_ROOT/output)
#   CMAP_QUEUE         — LSF queue (default: standard)
#   CMAP_WALLTIME      — bsub -W minutes (default: 120)
#   CMAP_MEM_MB        — rusage mem=<MB> (default: 16384)
#   CMAP_SKIP_PUBLISH  — set to 1 to skip --publish-best

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"
OUTPUT_ROOT="${CMAP_OUTPUT_ROOT:-$PROJECT_ROOT/output}"

LOG_DIR="$PROJECT_ROOT/logs/canonical_bg_sigma_488560_pool"
mkdir -p "$LOG_DIR"

UMAP_PY="$PROJECT_ROOT/visualization/canonical_umap_grid_search.py"
if [[ ! -f "$UMAP_PY" ]]; then
  echo "ERROR: missing $UMAP_PY" >&2
  exit 1
fi

QUEUE="${CMAP_QUEUE:-standard}"
WALLTIME="${CMAP_WALLTIME:-120}"
MEM_MB="${CMAP_MEM_MB:-16384}"
SKIP_PUBLISH="${CMAP_SKIP_PUBLISH:-0}"

JOB_NAME="can488560pool"

echo "Canonical UMAP pool: queue=$QUEUE walltime=$WALLTIME mem=${MEM_MB}MB"
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

   echo === step 1/2 canonical_umap_grid_search --run-all-local variant=bg_sigma_488560_shape ===
   python3 -u \"$UMAP_PY\" \
     --variant bg_sigma_488560_shape \
     --output-root \"$OUTPUT_ROOT\" \
     --run-all-local

   if [[ \"$SKIP_PUBLISH\" != 1 ]]; then
     echo === step 2/2 canonical_umap_grid_search --publish-best ===
     python3 -u \"$UMAP_PY\" \
       --variant bg_sigma_488560_shape \
       --output-root \"$OUTPUT_ROOT\" \
       --publish-best
   else
     echo SKIP step 2/2 CMAP_SKIP_PUBLISH=1
   fi"

echo ""
echo "Submitted: $JOB_NAME"
echo "Logs: $LOG_DIR"
echo "Monitor: bjobs -J '$JOB_NAME'"
echo "Output:  $OUTPUT_ROOT/cell_qc_all/canonical_embedding_bg_sigma_488560_shape_all.csv"
