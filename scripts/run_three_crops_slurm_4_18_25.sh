#!/bin/bash
# SLURM array: one task per cropped TIFF (488 / 560 / 642 nm).
# Dataset: data/4_18_25/CGNSample1_Position0_decon_dsr
# Submit:  sbatch scripts/run_three_crops_slurm_4_18_25.sh

#SBATCH --job-name=cpose_4_18_25
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=ondemand_gpu
#SBATCH --time=48:00:00
#SBATCH --output=/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_multiscale_cellpose_3D/output/4_18_25/CGNSample1_Position0_decon_dsr/_slurm_logs/cpose_%A_%a.out

set -euo pipefail

BASE="/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_multiscale_cellpose_3D"
REL="4_18_25/CGNSample1_Position0_decon_dsr"
STEMS=(488nm_crop 560nm_crop 642nm_crop)

STEM="${STEMS[$SLURM_ARRAY_TASK_ID]}"
echo "Job ${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} stem=${STEM}  $(date -Is)"

mkdir -p "${BASE}/output/${REL}/_slurm_logs"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "${TMP}/${REL}"
ln -sf "${BASE}/data/${REL}/${STEM}.tif" "${TMP}/${REL}/${STEM}.tif"

export PIPELINE_DATA_DIR="$TMP"
export PIPELINE_OUTPUT_DIR="${BASE}/output/${REL}/${STEM}"
mkdir -p "$PIPELINE_OUTPUT_DIR"

cd "$BASE"
python3 run_pipeline.py 2>&1 | tee "${PIPELINE_OUTPUT_DIR}/pipeline_run.log"
echo "Finished ${STEM} $(date -Is)"
