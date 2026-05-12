#!/bin/bash
# Submit one LSF job per sample for bg_sigma_shape cell cropping (union_488_560).
# Covers all 4_24_25_CGN_6_10_2 and 4_18_25 samples that have union_488_560_pass_bg_sigma_shape.tif.
#
# Usage:
#   bash features/submit_crop_bg_sigma_batch.sh [--dry-run]

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
WORKER="${PROJECT_ROOT}/features/_crop_bg_sigma_one.sh"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

submit_count=0
skip_count=0

submit_one() {
  local sample_dir="$1"
  local sample_name
  sample_name=$(basename "${sample_dir}")

  if [ ! -f "${sample_dir}/union_488_560_pass_bg_sigma_shape.tif" ]; then
    echo "  SKIP (no bg_sigma mask): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi

  local log_dir="${sample_dir}/_lsf_logs"
  mkdir -p "${log_dir}"

  if (( DRY_RUN )); then
    echo "  [dry-run] bsub -J crop_bgs_${sample_name} ... ${sample_dir}"
  else
    bsub \
      -n 1 \
      -q standard \
      -J "crop_bgs_${sample_name}" \
      -W 180 \
      -M 32000 \
      -R "rusage[mem=32000]" \
      -o "${log_dir}/crop_bg_sigma.lsf.out" \
      -e "${log_dir}/crop_bg_sigma.lsf.err" \
      bash "${WORKER}" "${sample_dir}"
    echo "  submitted: ${sample_name}"
  fi
  (( submit_count++ )) || true
}

echo "=== Submitting crop_bg_sigma_shape batch jobs ==="
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
