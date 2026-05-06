#!/usr/bin/env bash
# Full dataset rollout for union_488_560: gap-fill union volumes, forced CPU chain
# (union_488_560_otsu_shape.tif + otsu-shape-filtered crops + merge; no volume-norm here), then one LSF
# orchestrator waits for CPU completion and submits GPU DINOv2 per sample.
#
# From the repo login node (submits LSF only; no heavy local I/O):
#   bash scripts/submit_bsub_union_full_rollout.sh
#
# Env overrides:
#   CMAP_REPO_ROOT, CMAP_OUTPUT_BASE, CMAP_DATA_BASE (passed through to union script)
#   CMAP_ROLL_ORCH_QUEUE / CMAP_ROLL_ORCH_WALLTIME / CMAP_ROLL_ORCH_MEM_MB — orchestrator job
#   CMAP_ROLL_POOL_AFTER_GPU=1 — orchestrator also waits for GPU then runs
#                                  submit_bsub_union_dinov2_pool.sh
#   CMAP_ROLL_POLL_SEC, CMAP_ROLL_MAX_LOOPS — inner wait tuning
#
# Logs: $CMAP_REPO_ROOT/logs/union_full_rollout_<dataset>/

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"
OUT_BASE="${CMAP_OUTPUT_BASE:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2}"
DATASET="${CMAP_DATASET:-$(basename "$OUT_BASE")}"

LOG_DIR="$PROJECT_ROOT/logs/union_full_rollout_${DATASET}"
mkdir -p "$LOG_DIR"

INNER="$PROJECT_ROOT/scripts/run_union_rollout_orchestrate_inner.sh"
if [[ ! -f "$INNER" ]]; then
  echo "ERROR: missing $INNER" >&2
  exit 1
fi

ORCH_QUEUE="${CMAP_ROLL_ORCH_QUEUE:-standard}"
ORCH_WALL="${CMAP_ROLL_ORCH_WALLTIME:-2880}"
ORCH_MEM="${CMAP_ROLL_ORCH_MEM_MB:-8000}"

export CMAP_REPO_ROOT="$PROJECT_ROOT"
export CMAP_OUTPUT_BASE="$OUT_BASE"

echo "=== Phase 1: gap-fill union_488_560 (LSF per missing sample) ==="
bash "$PROJECT_ROOT/scripts/submit_bsub_union_488_560_4_24_25.sh"

echo "=== Phase 2: orchestrator LSF job (wait union → CPU submit+wait → GPU submit"
if [[ "${CMAP_ROLL_POOL_AFTER_GPU:-0}" == "1" ]]; then
  echo "         optional pool after GPU) ==="
else
  echo "         pool/UMAP skipped; export CMAP_ROLL_POOL_AFTER_GPU=1 to include) ==="
fi

out=$(bsub -q "$ORCH_QUEUE" -W "$ORCH_WALL" -R "rusage[mem=${ORCH_MEM}] span[hosts=1]" \
  -J "u488560_roll_orch_${DATASET}" \
  -o "$LOG_DIR/orchestrate_%J.out" \
  -e "$LOG_DIR/orchestrate_%J.err" \
  bash -lc \
  "set -euo pipefail
   export CMAP_REPO_ROOT=\"$PROJECT_ROOT\"
   export CMAP_OUTPUT_BASE=\"$OUT_BASE\"
   export CMAP_ROLL_POOL_AFTER_GPU=\"${CMAP_ROLL_POOL_AFTER_GPU:-0}\"
   export CMAP_ROLL_POLL_SEC=\"${CMAP_ROLL_POLL_SEC:-}\"
   export CMAP_ROLL_MAX_LOOPS=\"${CMAP_ROLL_MAX_LOOPS:-}\"
   eval \"\$(conda shell.bash hook)\"
   conda activate cmap
   exec bash \"$INNER\"")

echo "$out"
echo ""
echo "Orchestrator logs: $LOG_DIR/orchestrate_*.out / .err"
echo "CPU chain logs:    $PROJECT_ROOT/logs/union_chain_cpu_${DATASET}/"
echo "GPU logs:          $PROJECT_ROOT/logs/union_dinov2_gpu_${DATASET}/"
echo "Monitor:           bjobs -J 'u488560_roll_orch_${DATASET}'"
