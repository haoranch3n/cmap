#!/usr/bin/env bash
# Submit one LSF (CPU) job to scan no-deconv samples and write a CSV of cell
# counts at each pipeline stage. Uses absolute paths for LSF spool cwd safety.
#
# Usage (login node, from anywhere):
#   bash /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/scripts/no_deconv/submit_bsub_count_cells_per_step.sh
#
# Environment overrides:
#   CMAP_REPO_ROOT       — cmap repo (absolute)
#   CMAP_NO_DECONV_OUT   — sample tree root (default: $CMAP_REPO_ROOT/output_no_deconv)
#   CMAP_COUNT_CELLS_QUEUE
#   CMAP_COUNT_CELLS_MEM_MB  — bsub rusage[mem=...] (default: 32768)
#   CMAP_COUNT_CELLS_OUT   — output CSV path (default: logs/no_deconv_cell_counts/cell_counts_by_pipeline_step.csv under repo)

set -euo pipefail

PROJECT_ROOT="${CMAP_REPO_ROOT:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

OUTPUT_ROOT="${CMAP_NO_DECONV_OUT:-$PROJECT_ROOT/output_no_deconv}"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

LOG_DIR="${PROJECT_ROOT}/logs/no_deconv_cell_counts"
mkdir -p "$LOG_DIR"

OUT_CSV="${CMAP_COUNT_CELLS_OUT:-$LOG_DIR/cell_counts_by_pipeline_step.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
OUT_CSV="$(readlink -f "$OUT_CSV")"

QUEUE="${CMAP_COUNT_CELLS_QUEUE:-standard}"
MEM="${CMAP_COUNT_CELLS_MEM_MB:-32768}"
INNER="${PROJECT_ROOT}/scripts/no_deconv/_count_cells_per_step_job_inner.sh"
chmod +x "$INNER" 2>/dev/null || true

bsub \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=${MEM}]" \
  -J nd_count_cells_steps \
  -o "${LOG_DIR}/count_cells_per_step_%J.out" \
  bash -l "$INNER" "$PROJECT_ROOT" "$OUTPUT_ROOT" "$OUT_CSV"

echo "Submitted nd_count_cells_steps — CSV will be: $OUT_CSV"
echo "Log: ${LOG_DIR}/count_cells_per_step_%J.out"
