#!/bin/bash
# Submit one LSF GPU job per sample for DINOv2 extraction from cell_box_bg_sigma_shape.
# Run AFTER canonical features jobs (submit_extract_features_bg_sigma_batch.sh) complete.
#
# Usage:
#   bash features/submit_dinov2_bg_sigma_batch.sh [--dry-run]
#
# Override GPU queue or walltime:
#   CMAP_QUEUE=rhel88_gpu CMAP_WALLTIME=4:00 bash features/submit_dinov2_bg_sigma_batch.sh

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
WORKER="${PROJECT_ROOT}/features/_dinov2_bg_sigma_one.sh"
QUEUE="${CMAP_QUEUE:-rhel88_gpu}"
WALLTIME="${CMAP_WALLTIME:-4:00}"
MEM_MB="${CMAP_MEM_MB:-32768}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

LOGDIR="${PROJECT_ROOT}/logs/dinov2_bg_sigma"
mkdir -p "${LOGDIR}"

submit_count=0
skip_count=0

submit_one() {
  local sample_dir="$1"
  local sample_name
  sample_name=$(basename "${sample_dir}")

  if [ ! -d "${sample_dir}/cell_box_bg_sigma_shape" ]; then
    echo "  SKIP (no crops): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi
  if [ ! -f "${sample_dir}/dinov2_volume_norm_bounds_union_488_560.csv" ]; then
    echo "  SKIP (no bounds CSV): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi

  if (( DRY_RUN )); then
    echo "  [dry-run] bsub -J dino_bgs_${sample_name} ... ${sample_dir}"
  else
    bsub \
      -n 1 \
      -q "${QUEUE}" \
      -J "dino_bgs_${sample_name}" \
      -W "${WALLTIME}" \
      -M "${MEM_MB}" \
      -R "rusage[mem=${MEM_MB}] span[hosts=1]" \
      -gpu 'num=1:j_exclusive=yes' \
      -o "${LOGDIR}/${sample_name}_%J.out" \
      -e "${LOGDIR}/${sample_name}_%J.err" \
      bash "${WORKER}" "${sample_dir}"
    echo "  submitted: ${sample_name}"
  fi
  (( submit_count++ )) || true
}

echo "=== Submitting DINOv2 bg_sigma_shape batch jobs ==="
echo "Queue: ${QUEUE}  Walltime: ${WALLTIME}  Mem: ${MEM_MB} MB"
echo "Dry run: ${DRY_RUN}"
echo

echo "--- 4_24_25_CGN_6_10_2 ---"
for d in "${PROJECT_ROOT}/output/4_24_25_CGN_6_10_2/"*_decon_dsr; do
  [ -d "$d" ] && submit_one "$d"
done

echo
echo "--- 4_18_25 ---"
for d in "${PROJECT_ROOT}/output/4_18_25/"*_decon_dsr; do
  [ -d "$d" ] && submit_one "$d"
done

echo
echo "=== Done: submitted=${submit_count}  skipped=${skip_count} ==="
echo "Logs: ${LOGDIR}"
echo "Monitor: bjobs -J 'dino_bgs_*'"
