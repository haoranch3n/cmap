#!/bin/bash
# Submit one LSF job per sample to extract canonical features from cell_box_bg_sigma_488560_shape.
# Run AFTER submit_crop_bg_sigma_488560_batch.sh crops complete.
#
# Usage:
#   bash features/submit_extract_features_bg_sigma_488560_batch.sh [--dry-run]

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
WORKER="${PROJECT_ROOT}/features/_extract_features_bg_sigma_488560_one.sh"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

submit_count=0
skip_count=0

submit_one() {
  local sample_dir="$1"
  local sample_name
  sample_name=$(basename "${sample_dir}")

  if [ ! -d "${sample_dir}/cell_box_bg_sigma_488560_shape" ]; then
    echo "  SKIP (no crops dir): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi

  local log_dir="${sample_dir}/_lsf_logs"
  mkdir -p "${log_dir}"

  if (( DRY_RUN )); then
    echo "  [dry-run] bsub -J feat_488560_${sample_name} ... ${sample_dir}"
  else
    bsub \
      -n 1 \
      -q standard \
      -J "feat_488560_${sample_name}" \
      -W 60 \
      -M 8000 \
      -R "rusage[mem=8000]" \
      -o "${log_dir}/extract_features_bg_sigma_488560.lsf.out" \
      -e "${log_dir}/extract_features_bg_sigma_488560.lsf.err" \
      bash "${WORKER}" "${sample_dir}"
    echo "  submitted: ${sample_name}"
  fi
  (( submit_count++ )) || true
}

echo "=== Submitting extract_features_bg_sigma_488560 batch jobs ==="
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
