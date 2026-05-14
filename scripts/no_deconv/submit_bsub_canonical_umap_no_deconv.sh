#!/usr/bin/env bash
# Submit one LSF CPU job: canonical UMAP hyperparameter sweep on pooled
# ``cell_qc/qc_features.csv`` from the no-deconv output tree, then
# ``--publish-best`` into ``cell_qc_all/``.
#
# Prereq: per-sample ``output_no_deconv/<batch>/<sample>/cell_qc/qc_features.csv``
#
# Usage:
#   bash scripts/no_deconv/submit_bsub_canonical_umap_no_deconv.sh
#
# Outputs (under OUTPUT_ROOT, default output_no_deconv):
#   cell_qc_all/umap_canonical_no_deconv_filtered_642_sweep/
#   cell_qc_all/canonical_embedding_no_deconv_filtered_642_all.csv
#
# Env overrides:
#   CMAP_REPO_ROOT, CMAP_NO_DECONV_OUT, CMAP_QUEUE, CMAP_WALLTIME, CMAP_MEM_MB
#   CMAP_SKIP_PUBLISH=1
#   CMAP_CANONICAL_UMAP_VARIANT=no_deconv_filtered_642

set -euo pipefail

PROJECT_ROOT="${CMAP_REPO_ROOT:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

OUTPUT_ROOT="${CMAP_NO_DECONV_OUT:-$PROJECT_ROOT/output_no_deconv}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

LOG_DIR="$PROJECT_ROOT/logs/no_deconv_canonical_umap"
mkdir -p "$LOG_DIR"

INNER="$PROJECT_ROOT/scripts/no_deconv/_canonical_umap_no_deconv_inner.sh"
chmod +x "$INNER" 2>/dev/null || true

QUEUE="${CMAP_QUEUE:-standard}"
WALLTIME="${CMAP_WALLTIME:-120}"
MEM_MB="${CMAP_MEM_MB:-16384}"
JOB_NAME="nd_canonical_umap"

bsub \
  -J "$JOB_NAME" \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=${MEM_MB}] span[hosts=1]" \
  -W "$WALLTIME" \
  -o "$LOG_DIR/${JOB_NAME}_%J.out" \
  -e "$LOG_DIR/${JOB_NAME}_%J.err" \
  bash -l "$INNER" "$PROJECT_ROOT" "$OUTPUT_ROOT"

echo "Submitted: $JOB_NAME"
echo "Logs: $LOG_DIR/${JOB_NAME}_%J.{out,err}"
echo "Master CSV: $OUTPUT_ROOT/cell_qc_all/canonical_embedding_no_deconv_filtered_642_all.csv"
