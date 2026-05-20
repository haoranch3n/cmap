#!/usr/bin/env bash
# Submit segmentation_registered_crop pipeline as an LSF GPU job.
#
# Usage:
#   bsub < scripts/bsub_run_batch.sh
#   # or with a specific batch:
#   BATCH=4_24_25_CGN_6_10_2 bsub < scripts/bsub_run_batch.sh

#BSUB -J seg_reg_crop
#BSUB -q rhel88_gpu
#BSUB -n 8
#BSUB -R "rusage[mem=32000] span[hosts=1]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -W 48:00
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/seg_reg_crop_%J.out
#BSUB -e /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/seg_reg_crop_%J.err

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SCRIPT_DIR="$PROJECT_ROOT/segmentation_registered_crop"

eval "$(conda shell.bash hook)"
conda activate cmap

mkdir -p "$SCRIPT_DIR/logs"

cd "$PROJECT_ROOT"

BATCH_ARGS=""
if [ -n "${BATCH:-}" ]; then
    BATCH_ARGS="--batch $BATCH"
fi

echo "Starting segmentation_registered_crop pipeline at $(date)"
echo "BATCH_ARGS: $BATCH_ARGS"

python "$SCRIPT_DIR/run_pipeline.py" \
    --crops-root "/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_general/No_decon_tests/outputs/cell_crops/batch_processing_fullZ" \
    --output-root "$PROJECT_ROOT/output_registered_crop_seg" \
    $BATCH_ARGS

echo "Pipeline finished at $(date)"
