#!/usr/bin/env bash
# Submit one LSF GPU job per registered cell crop.
# Usage:  bash scripts/submit_per_cell.sh [--force]
#
# Skips cells whose 3 *_segmented.tif outputs already exist (unless --force).

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
CROPS_ROOT="/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_general/No_decon_tests/outputs/cell_crops/batch_processing_fullZ"
OUTPUT_ROOT="$PROJECT_ROOT/output_registered_crop_seg"
LOG_DIR="$PROJECT_ROOT/segmentation_registered_crop/logs/per_cell"
SCRIPT_DIR="$PROJECT_ROOT/segmentation_registered_crop"

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

mkdir -p "$LOG_DIR"

submitted=0
skipped=0

# Discover all cells: batch / sample_position / cell_id
while IFS= read -r tif_path; do
    cell_id=$(basename "$(dirname "$tif_path")")
    # Get batch by stripping CROPS_ROOT prefix, then first path component
    rel="${tif_path#$CROPS_ROOT/}"
    batch=$(echo "$rel" | cut -d'/' -f1)

    # Skip if all 3 channel outputs already exist (and not forcing)
    if [[ -z "$FORCE_FLAG" ]]; then
        out_dir="$OUTPUT_ROOT/$batch"
        # Find the sample_position subdir
        sample_pos=$(echo "$rel" | cut -d'/' -f2)
        ch642="$out_dir/$sample_pos/$cell_id/${cell_id}_ch642_segmented.tif"
        ch488="$out_dir/$sample_pos/$cell_id/${cell_id}_ch488_segmented.tif"
        ch560="$out_dir/$sample_pos/$cell_id/${cell_id}_ch560_segmented.tif"
        if [[ -f "$ch642" && -f "$ch488" && -f "$ch560" ]]; then
            ((skipped++)) || true
            continue
        fi
    fi

    bsub \
        -J "seg_${cell_id}" \
        -q rhel88_gpu \
        -n 2 \
        -R "rusage[mem=8000] span[hosts=1]" \
        -gpu "num=1:mode=shared:mps=no" \
        -W 2:00 \
        -o "$LOG_DIR/${batch}_${cell_id}_%J.out" \
        -e "$LOG_DIR/${batch}_${cell_id}_%J.err" \
        bash -c "
            eval \"\$(conda shell.bash hook)\"
            conda activate cmap
            cd $PROJECT_ROOT
            python $SCRIPT_DIR/run_pipeline.py \
                --crops-root '$CROPS_ROOT' \
                --output-root '$OUTPUT_ROOT' \
                --batch '$batch' \
                --crop '$cell_id' \
                $FORCE_FLAG
        " 2>/dev/null

    ((submitted++)) || true
done < <(find "$CROPS_ROOT" \
    -path "*/flagged_rotation_increase_gt5deg/*" -prune \
    -o -name "*_registered.tif" -print | sort)

echo "Submitted: $submitted  Skipped (already done): $skipped"
