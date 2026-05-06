#!/usr/bin/env bash
# Smoke test: reordered QC-before-crop union chain + DINOv2 on cell 37 only.
#
# 1) LSF standard: run scripts/run_smoke_reorder_union_one_cell_cpu_inner.sh
# 2) LSF rhel88_gpu: run scripts/run_smoke_reorder_union_one_cell_gpu_inner.sh after (1).
#
# Usage (from login node):
#   bash scripts/bsub_smoke_reorder_union_one_cell.sh
#
# Override source sample (absolute):
#   CMAP_SMOKE_SRC_SAMPLE=/path/to/Sample10_Position4_decon_dsr bash scripts/bsub_smoke_reorder_union_one_cell.sh

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

SRC_SAMPLE="${CMAP_SMOKE_SRC_SAMPLE:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2/Sample10_Position4_decon_dsr}"
LOG_DIR="$PROJECT_ROOT/logs/smoke_reorder_one_cell"
mkdir -p "$LOG_DIR"

CPU_INNER="$PROJECT_ROOT/scripts/run_smoke_reorder_union_one_cell_cpu_inner.sh"
GPU_INNER="$PROJECT_ROOT/scripts/run_smoke_reorder_union_one_cell_gpu_inner.sh"
for f in "$CPU_INNER" "$GPU_INNER"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

for f in union_488_560.tif union_488_560_combined.tif; do
  if [[ ! -f "$SRC_SAMPLE/$f" ]]; then
    echo "ERROR: missing $SRC_SAMPLE/$f" >&2
    exit 1
  fi
done

CPU_QUEUE="${CMAP_QUEUE_CPU:-standard}"
GPU_QUEUE="${CMAP_QUEUE_GPU:-rhel88_gpu}"
CPU_WALL="${CMAP_WALLTIME_CPU:-120}"
GPU_WALL="${CMAP_WALLTIME_GPU:-1:00}"
CPU_MEM="${CMAP_MEM_MB_CPU:-32000}"
GPU_MEM="${CMAP_MEM_MB_GPU:-32768}"

export CMAP_REPO_ROOT="$PROJECT_ROOT"
export CMAP_SMOKE_SRC_SAMPLE="$SRC_SAMPLE"

echo "Submitting CPU smoke (queue=$CPU_QUEUE) ..."
CPU_SUB="$(
  bsub -J "smoke_u560_cpu" -q "$CPU_QUEUE" -n 1 \
    -R "rusage[mem=${CPU_MEM}] span[hosts=1]" -W "$CPU_WALL" \
    -o "$LOG_DIR/smoke_cpu_%J.out" -e "$LOG_DIR/smoke_cpu_%J.err" \
    bash -lc "export CMAP_REPO_ROOT='$PROJECT_ROOT' CMAP_SMOKE_SRC_SAMPLE='$SRC_SAMPLE'; bash '$CPU_INNER'" 2>&1
)"
echo "$CPU_SUB"
CPU_JID=$(echo "$CPU_SUB" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p')
if [[ -z "$CPU_JID" ]]; then
  echo "ERROR: could not parse CPU job id from bsub output" >&2
  exit 1
fi

echo "Submitting GPU DINOv2 (queue=$GPU_QUEUE), wait done($CPU_JID) ..."
GPU_SUB="$(
  bsub -J "smoke_u560_gpu" -q "$GPU_QUEUE" -n 1 \
    -R "rusage[mem=${GPU_MEM}] span[hosts=1]" -gpu 'num=1:j_exclusive=yes' \
    -W "$GPU_WALL" -w "done($CPU_JID)" \
    -o "$LOG_DIR/smoke_gpu_%J.out" -e "$LOG_DIR/smoke_gpu_%J.err" \
    bash -lc "export CMAP_REPO_ROOT='$PROJECT_ROOT'; bash '$GPU_INNER'" 2>&1
)"
echo "$GPU_SUB"

echo "Logs: $LOG_DIR"
echo "Monitor: bjobs -J 'smoke_u560_*'"
