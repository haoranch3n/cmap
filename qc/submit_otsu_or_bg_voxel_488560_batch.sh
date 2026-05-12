#!/bin/bash
# Submit one LSF job per sample: otsu_or_bg_voxel:3 (488+560) + join into union QC CSV.
# Covers 4_24_25_CGN_6_10_2 and 4_18_25 samples with union_488_560 inputs and existing union QC CSV.
#
# Usage:
#   bash qc/submit_otsu_or_bg_voxel_488560_batch.sh [--dry-run]

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
WORKER="${PROJECT_ROOT}/qc/_otsu_or_bg_voxel_488560_one.sh"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

submit_count=0
skip_count=0

submit_one() {
  local sample_dir="$1"
  local sample_name
  sample_name=$(basename "${sample_dir}")

  # Strict-overlap mask is the canonical union variant input.
  if [ ! -f "${sample_dir}/union_488_560_strict_overlap.tif" ] || \
     [ ! -f "${sample_dir}/union_488_560_combined.tif" ]; then
    echo "  SKIP (missing union mask/combined): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi

  # otsu_or_bg_voxel joins back into the union QC CSV; that CSV must exist.
  if [ ! -f "${sample_dir}/cell_qc_union_488_560/qc_features_filtered.csv" ]; then
    echo "  SKIP (missing cell_qc_union_488_560/qc_features_filtered.csv): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi

  local log_dir="${sample_dir}/_lsf_logs"
  mkdir -p "${log_dir}"

  if (( DRY_RUN )); then
    echo "  [dry-run] bsub -J oobgv_${sample_name} ... ${sample_dir}"
  else
    bsub \
      -n 1 \
      -q standard \
      -J "oobgv_${sample_name}" \
      -W 180 \
      -M 32000 \
      -R "rusage[mem=32000]" \
      -o "${log_dir}/otsu_or_bg_voxel_488560_apply.lsf.out" \
      -e "${log_dir}/otsu_or_bg_voxel_488560_apply.lsf.err" \
      bash "${WORKER}" "${sample_dir}"
    echo "  submitted: ${sample_name}"
  fi
  (( submit_count++ )) || true
}

echo "=== Submitting otsu_or_bg_voxel_488560 batch jobs ==="
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
