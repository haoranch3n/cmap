#!/usr/bin/env bash
# Submit one LSF (CPU) job per no-deconv sample to run the full union_488_560
# chain (strict mask → bg_sigma:3 QC → pass mask → crop → extract).
#
# Prereq per sample: 488nm_crop_3D_indexed.tif and 560nm_crop_3D_indexed.tif
#
# Usage (from login node):
#   bash scripts/no_deconv/submit_bsub_union_chain_no_deconv.sh
#   bash scripts/no_deconv/submit_bsub_union_chain_no_deconv.sh --dry-run
#   CMAP_FORCE_OVERWRITE=1 bash scripts/no_deconv/submit_bsub_union_chain_no_deconv.sh
#
# Outputs per sample (under output_no_deconv/<batch>/<sample>/):
#   union_488_560_strict_overlap.tif
#   union_488_560_strict_overlap_combined.tif
#   cell_qc_bg_sigma_488560_shape/qc_features_filtered.csv
#   union_488_560_strict_overlap_pass_bg_sigma_488560_shape.tif
#   cell_box_bg_sigma_488560_shape/   (cropped cells)
#   cell_box_full_z_bg_sigma_488560_shape/
#   cell_qc_bg_sigma_488560_shape/qc_features.csv  (post-crop)
#
# Env overrides:
#   CMAP_REPO_ROOT         -- cmap repo (absolute)
#   CMAP_NO_DECONV_OUT     -- sample tree root (default: $CMAP_REPO_ROOT/output_no_deconv)
#   CMAP_QUEUE             -- LSF queue (default: standard)
#   CMAP_WALLTIME          -- bsub -W minutes (default: 240)
#   CMAP_MEM_MB            -- rusage[mem=...] in MB (default: 32768)
#   CMAP_FORCE_OVERWRITE=1 -- rerun even if outputs already exist
#   CMAP_SAMPLE_FILTER     -- substring filter on sample name
#   CMAP_SUBMIT_LIMIT      -- max jobs to submit (debug)

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

PROJECT_ROOT="${CMAP_REPO_ROOT:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

OUTPUT_ROOT="${CMAP_NO_DECONV_OUT:-$PROJECT_ROOT/output_no_deconv}"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

LOG_DIR="$PROJECT_ROOT/logs/no_deconv_union_chain"
[[ "$DRY_RUN" -eq 0 ]] && mkdir -p "$LOG_DIR"

INNER="$PROJECT_ROOT/scripts/no_deconv/_union_chain_one_sample_inner.sh"
chmod +x "$INNER" 2>/dev/null || true

QUEUE="${CMAP_QUEUE:-standard}"
WALLTIME="${CMAP_WALLTIME:-240}"
MEM_MB="${CMAP_MEM_MB:-32768}"
FORCE="${CMAP_FORCE_OVERWRITE:-0}"
SAMPLE_FILTER="${CMAP_SAMPLE_FILTER:-}"
SUBMIT_LIMIT="${CMAP_SUBMIT_LIMIT:-}"

BATCHES=("4_18_25" "4_24_25_CGN_6_10_2")

n_submitted=0
n_skip_done=0
n_skip_no_seg=0

for batch in "${BATCHES[@]}"; do
  batch_dir="${OUTPUT_ROOT}/${batch}"
  [[ -d "$batch_dir" ]] || continue

  for sample_dir in "${batch_dir}"/*/; do
    [[ -d "$sample_dir" ]] || continue
    sample_dir="${sample_dir%/}"
    sample="$(basename "$sample_dir")"

    [[ -n "$SAMPLE_FILTER" && "$sample" != *"$SAMPLE_FILTER"* ]] && continue

    # Require segmentation inputs
    if [[ ! -f "${sample_dir}/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif" || \
          ! -f "${sample_dir}/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif" ]]; then
      echo "SKIP (no seg): ${batch}/${sample}"
      (( n_skip_no_seg++ )) || true
      continue
    fi

    # Skip if already done (unless force)
    done_csv="${sample_dir}/cell_qc_bg_sigma_488560_shape/qc_features.csv"
    if [[ "$FORCE" != "1" && -f "$done_csv" ]]; then
      echo "SKIP (done): ${batch}/${sample}"
      (( n_skip_done++ )) || true
      continue
    fi

    if [[ -n "$SUBMIT_LIMIT" && "$n_submitted" -ge "$SUBMIT_LIMIT" ]]; then
      echo "Stopping at CMAP_SUBMIT_LIMIT=$SUBMIT_LIMIT"
      break 2
    fi

    job_name="nd_union_${sample}"
    log_file="${LOG_DIR}/${batch}_${sample}_%J.out"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "DRY-RUN: bsub -q $QUEUE -J ${job_name} bash -l $INNER $PROJECT_ROOT $sample_dir"
    else
      CMAP_FORCE_OVERWRITE="$FORCE" \
      bsub \
        -q "$QUEUE" \
        -n 4 \
        -R "rusage[mem=${MEM_MB}] span[hosts=1]" \
        -W "$WALLTIME" \
        -J "${job_name:0:60}" \
        -o "$log_file" \
        -env "CMAP_FORCE_OVERWRITE=$FORCE" \
        bash -l "$INNER" "$PROJECT_ROOT" "$sample_dir"
      echo "SUBMITTED: ${batch}/${sample}"
    fi

    (( n_submitted++ )) || true
  done
done

echo ""
echo "Submitted=${n_submitted}  Skipped(done)=${n_skip_done}  Skipped(no_seg)=${n_skip_no_seg}"
echo "Logs: ${LOG_DIR}"
echo "Monitor: bjobs -J 'nd_union_*'"
