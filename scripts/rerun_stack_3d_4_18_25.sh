#!/bin/bash
# Re-run steps 3+4 (stack 2D → 3D) for 4_18_25 dataset.
# Steps 1+2 already completed; only stacking + 3D matching failed due to path bug.

#SBATCH --job-name=cp_fix_4_18
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=ondemand_gpu
#SBATCH --time=12:00:00
#SBATCH --output=/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_multiscale_cellpose_3D/output/4_18_25/CGNSample1_Position0_decon_dsr/_slurm_logs/fix_%A_%a.out

set -euo pipefail

BASE="/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_multiscale_cellpose_3D"
REL="4_18_25/CGNSample1_Position0_decon_dsr"
STEMS=(488nm_crop 560nm_crop 642nm_crop)

STEM="${STEMS[$SLURM_ARRAY_TASK_ID]}"
echo "Job ${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} stem=${STEM}  $(date -Is)"

export PIPELINE_OUTPUT_DIR="${BASE}/output/${REL}/${STEM}"
export PIPELINE_DATA_DIR="${BASE}/data"

cd "$BASE"

echo ">>> Step 3: stack 2D planes"
python3 cellcomposor/stack_2D_planes.py

echo ">>> Step 4: create 3D cells"
python3 cellcomposor/create_3D_cells.py

echo "Finished ${STEM} $(date -Is)"
