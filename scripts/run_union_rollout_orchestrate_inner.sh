#!/usr/bin/env bash
# Run inside one long LSF standard job: wait for union gap-fill, run forced CPU
# chain for all samples, wait for CPU outputs, submit forced GPU DINOv2 jobs,
# optionally wait and submit the pooled UMAP job.
#
# Env (set by submit_bsub_union_full_rollout.sh or manually):
#   CMAP_REPO_ROOT, CMAP_OUTPUT_BASE
# Optional:
#   CMAP_ROLL_POOL_AFTER_GPU — set to 1 to run submit_bsub_union_dinov2_pool.sh
#                              after all per-sample dinov2_embeddings.npy exist.

set -euo pipefail

PROJECT_ROOT="${CMAP_REPO_ROOT:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap}"
OUT_BASE="${CMAP_OUTPUT_BASE:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2}"

poll_s="${CMAP_ROLL_POLL_SEC:-120}"
max_wait_loops="${CMAP_ROLL_MAX_LOOPS:-900}"

log() { echo "[$(date -Is)] $*"; }

count_mask_ready() {
  local n=0
  for od in "$OUT_BASE"/*; do
    [[ -d "$od" ]] || continue
    local sam
    sam=$(basename "$od")
    [[ "$sam" == _* ]] && continue
    local m488 m560
    m488="$od/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif"
    m560="$od/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif"
    if [[ -f "$m488" && -f "$m560" ]]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

count_union_complete() {
  local n=0
  for od in "$OUT_BASE"/*; do
    [[ -d "$od" ]] || continue
    local sam
    sam=$(basename "$od")
    [[ "$sam" == _* ]] && continue
    if [[ -f "$od/union_488_560.tif" && -f "$od/union_488_560_combined.tif" ]]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

count_cpu_chain_complete() {
  local n=0
  for od in "$OUT_BASE"/*; do
    [[ -d "$od" ]] || continue
    local sam
    sam=$(basename "$od")
    [[ "$sam" == _* ]] && continue
    if [[ ! -f "$od/union_488_560.tif" || ! -f "$od/union_488_560_combined.tif" ]]; then
      continue
    fi
    if [[ -f "$od/union_488_560_otsu_shape.tif" \
      && -d "$od/cell_box_otsu_shape_filtered" \
      && -f "$od/cell_qc_union_488_560/qc_features_filtered.csv" ]]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

count_union_inputs() {
  local n=0
  for od in "$OUT_BASE"/*; do
    [[ -d "$od" ]] || continue
    local sam
    sam=$(basename "$od")
    [[ "$sam" == _* ]] && continue
    if [[ -f "$od/union_488_560.tif" && -f "$od/union_488_560_combined.tif" ]]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

count_dinov2_complete() {
  local n=0
  for od in "$OUT_BASE"/*; do
    [[ -d "$od" ]] || continue
    local sam
    sam=$(basename "$od")
    [[ "$sam" == _* ]] && continue
    if [[ -f "$od/cell_qc_union_488_560/dinov2_embeddings.npy" ]]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

eval "$(conda shell.bash hook)"
conda activate cmap
cd "$PROJECT_ROOT"

need_masks="$(count_mask_ready)"
log "Union wait: need_mask_ready=$need_masks (488+560 indexed present)"
i=0
while [[ "$(count_union_complete)" -lt "$need_masks" ]]; do
  i=$((i + 1))
  if [[ "$i" -ge "$max_wait_loops" ]]; then
    log "ERROR: timeout waiting for union_488_560.tif+combined ($i polls)"
    exit 1
  fi
  log "  union_complete=$(count_union_complete) / $need_masks (poll $i)"
  sleep "$poll_s"
done
log "Union inputs ready for $(count_union_complete) sample(s)."

log "Submit CPU chain (CMAP_FORCE_OVERWRITE=1) …"
export CMAP_FORCE_OVERWRITE=1
bash "$PROJECT_ROOT/scripts/submit_bsub_union_chain_cpu.sh"

need_cpu="$(count_union_inputs)"
log "CPU wait: samples with union mask+combined=$need_cpu"
i=0
while [[ "$(count_cpu_chain_complete)" -lt "$need_cpu" ]]; do
  i=$((i + 1))
  if [[ "$i" -ge "$max_wait_loops" ]]; then
    log "ERROR: timeout waiting for CPU chain ($i polls); got $(count_cpu_chain_complete)/$need_cpu"
    exit 1
  fi
  log "  cpu_complete=$(count_cpu_chain_complete) / $need_cpu (poll $i)"
  sleep "$poll_s"
done
log "CPU chain artefacts present for $(count_cpu_chain_complete) sample(s)."

log "Submit GPU DINOv2 jobs (CMAP_FORCE_OVERWRITE=1) …"
bash "$PROJECT_ROOT/scripts/submit_bsub_union_dinov2_gpu.sh"

if [[ "${CMAP_ROLL_POOL_AFTER_GPU:-0}" == "1" ]]; then
  log "Pool wait: dinov2 .npy for $need_cpu sample(s)"
  i=0
  while [[ "$(count_dinov2_complete)" -lt "$need_cpu" ]]; do
    i=$((i + 1))
    if [[ "$i" -ge "$max_wait_loops" ]]; then
      log "ERROR: timeout waiting for dinov2_embeddings.npy"
      exit 1
    fi
    log "  dinov2_complete=$(count_dinov2_complete) / $need_cpu (poll $i)"
    sleep "$poll_s"
  done
  log "Submit pooled DINOv2 / UMAP job …"
  bash "$PROJECT_ROOT/scripts/submit_bsub_union_dinov2_pool.sh"
else
  log "Skip pool/UMAP (set CMAP_ROLL_POOL_AFTER_GPU=1 to enable after GPU completes)."
fi

log "Orchestrator finished."
