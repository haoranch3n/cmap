#!/usr/bin/env bash
# LSF GPU job: run the ortho-3d-expr experiment (XY + YZ orthogonal segmentation)
# on 5 representative registered cell crops.
#
# Usage:
#   bsub < segmentation_registered_crop/orthogonal_3d/bsub_ortho3d_expr.sh
#   # or run directly (CPU fallback):
#   bash segmentation_registered_crop/orthogonal_3d/bsub_ortho3d_expr.sh
#
# Env overrides:
#   ORTHO3D_OUTPUT_ROOT   default: <repo>/output_registered_crop_seg_ortho3d
#   ORTHO3D_FORCE         set to "1" to re-run even if outputs exist

#BSUB -J ortho3d_expr
#BSUB -q rhel88_gpu
#BSUB -n 4
#BSUB -R "rusage[mem=24000] span[hosts=1]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -W 12:00
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/ortho3d_expr_%J.out
#BSUB -e /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/ortho3d_expr_%J.err

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
ORTHO_SCRIPT="$PROJECT_ROOT/segmentation_registered_crop/orthogonal_3d/run_ortho3d_expr.py"

eval "$(conda shell.bash hook)"
conda activate cmap

mkdir -p "$PROJECT_ROOT/segmentation_registered_crop/logs"

cd "$PROJECT_ROOT"

FORCE_FLAG=""
if [ "${ORTHO3D_FORCE:-0}" = "1" ]; then
    FORCE_FLAG="--force"
fi

OUTPUT_ROOT="${ORTHO3D_OUTPUT_ROOT:-$PROJECT_ROOT/output_registered_crop_seg_ortho3d}"

echo "=========================================="
echo "ortho-3d-expr: XY + YZ orthogonal pipeline"
echo "Started: $(date)"
echo "Output:  $OUTPUT_ROOT"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "=========================================="

python "$ORTHO_SCRIPT" \
    --output-root "$OUTPUT_ROOT" \
    $FORCE_FLAG

echo "Completed: $(date)"
