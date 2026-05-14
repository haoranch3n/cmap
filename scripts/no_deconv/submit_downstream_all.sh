#!/usr/bin/env bash
# Submit the full no-deconv downstream pipeline for all segmented samples.
#
# Chain per sample (filtered_642 variant, CPU-only):
#   postprocess/filter_642_mask.py
#   postprocess/combine_with_mask.py
#   qc/filter_by_intensity.py --from-volume
#   qc/apply_qc_pass_to_label_mask.py
#   features/crop_cells.py
#   features/extract_features.py
#   qc/merge_qc_features_filtered.py
#
# Re-run safe: skips any sample that already has the final merged table
# cell_qc/qc_features_filtered.csv (not merely filtered_642.tif, which can
# exist after a partial run).
#
# Usage:
#   bash scripts/no_deconv/submit_downstream_all.sh
#   bash scripts/no_deconv/submit_downstream_all.sh --dry-run

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
OUTPUT_NO_DECONV="${PROJECT_ROOT}/output_no_deconv"
LOG_DIR="${PROJECT_ROOT}/logs/no_deconv_downstream"

[[ "$DRY_RUN" -eq 0 ]] && mkdir -p "$LOG_DIR"

BATCHES=("4_18_25" "4_24_25_CGN_6_10_2")

n_submitted=0
n_skipped=0
n_no_seg=0

for batch in "${BATCHES[@]}"; do
  batch_dir="${OUTPUT_NO_DECONV}/${batch}"
  if [[ ! -d "$batch_dir" ]]; then
    echo "WARNING: batch dir not found: $batch_dir" >&2
    continue
  fi

  for sample_dir in "${batch_dir}"/*/; do
    [[ -d "$sample_dir" ]] || continue
    sample="$(basename "$sample_dir")"
    sample_dir="${sample_dir%/}"

    # Skip if full downstream (through merge) is already done
    if [[ -f "${sample_dir}/cell_qc/qc_features_filtered.csv" ]]; then
      echo "SKIP (done): ${batch}/${sample}"
      (( n_skipped++ )) || true
      continue
    fi

    # Skip if segmentation is missing
    if [[ ! -f "${sample_dir}/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif" ]]; then
      echo "SKIP (no seg): ${batch}/${sample}"
      (( n_no_seg++ )) || true
      continue
    fi

    log_file="${LOG_DIR}/${batch}_${sample}_%J.out"
    job_name="nd_dn_${sample}"
    inner="${PROJECT_ROOT}/scripts/no_deconv/_downstream_one_sample_inner.sh"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "DRY-RUN bsub -q standard -n 4 -R rusage[mem=32768] -J ${job_name} -o ${log_file} bash -l ${inner} ${PROJECT_ROOT} ${sample_dir}"
    else
      bsub \
        -q standard \
        -n 4 \
        -R "rusage[mem=32768]" \
        -J "${job_name:0:60}" \
        -o "${log_file}" \
        bash -l "${inner}" "${PROJECT_ROOT}" "${sample_dir}"
      echo "SUBMITTED: ${batch}/${sample}"
    fi

    (( n_submitted++ )) || true
  done
done

echo ""
echo "Submitted: ${n_submitted}  Skipped (done): ${n_skipped}  Skipped (no seg): ${n_no_seg}"
