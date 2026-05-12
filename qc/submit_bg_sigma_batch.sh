#!/bin/bash
# Submit one LSF job per sample for bg_sigma:3 filtering (union_488_560 variant).
# Covers all 4_24_25_CGN_6_10_2 and 4_18_25 samples that have union_488_560 inputs.
#
# Usage:
#   bash qc/submit_bg_sigma_batch.sh [--dry-run]
#
# With --dry-run: prints the bsub commands without submitting.

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
WORKER="${PROJECT_ROOT}/qc/_bg_sigma_apply_one.sh"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

submit_count=0
skip_count=0

submit_one() {
  local sample_dir="$1"
  local sample_name
  sample_name=$(basename "${sample_dir}")

  # Check required inputs exist
  # Strict-overlap mask is the canonical union variant input
  # (postprocess/__init__.py VARIANT_FILES['union_488_560']['mask']).
  # cell_qc_union_488_560/qc_features_filtered.csv is no longer required up-front:
  # filter_by_intensity.py --from-volume creates it from the mask + combined.
  if [ ! -f "${sample_dir}/union_488_560_strict_overlap.tif" ] || \
     [ ! -f "${sample_dir}/union_488_560_combined.tif" ]; then
    echo "  SKIP (missing inputs): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi

  local log_dir="${sample_dir}/_lsf_logs"
  mkdir -p "${log_dir}"

  if (( DRY_RUN )); then
    echo "  [dry-run] bsub -J bgs_${sample_name} ... ${sample_dir}"
  else
    bsub \
      -n 1 \
      -q standard \
      -J "bgs_${sample_name}" \
      -W 180 \
      -M 32000 \
      -R "rusage[mem=32000]" \
      -o "${log_dir}/bg_sigma_apply.lsf.out" \
      -e "${log_dir}/bg_sigma_apply.lsf.err" \
      bash "${WORKER}" "${sample_dir}"
    echo "  submitted: ${sample_name}"
  fi
  (( submit_count++ )) || true
}

echo "=== Submitting bg_sigma:3 batch jobs ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Dry run:      ${DRY_RUN}"
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
