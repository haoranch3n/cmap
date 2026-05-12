#!/usr/bin/env bash
# Generate bg_stats.csv for every sample that has a combined TIFF but no bg_stats.csv.
#
# bg_stats.csv is ONE file per sample (per 3D volume), written to:
#   cell_qc_union_488_560/bg_stats.csv  (union_488_560 variant)
#   cell_qc/bg_stats.csv                (filtered_642 variant)
#
# It stores per-channel background mean, std, and pixel count computed from the
# image background (pixels outside all cell masks).  The napari plugin uses these
# with the interactive X spinbox to dim cells where mean < bg_mean + X·bg_std.
#
# Usage (run from the LSF login node where /research_jude/... is mounted):
#   bash scripts/submit_bsub_generate_bg_stats.sh
#
# Environment overrides:
#   CMAP_REPO_ROOT      — repo root (default: this script's parent)
#   CMAP_METHOD         — filter method (default: otsu_or_bg:3)
#   CMAP_FORCE          — set 1 to re-generate even if bg_stats.csv already exists
#   CMAP_SUBMIT_LIMIT   — max jobs to submit (default: unlimited)
#   CMAP_OUTPUT_BASES   — space-separated output base dirs (default: both datasets)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -L 2>/dev/null || pwd)"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
FILTER_PY="$PROJECT_ROOT/qc/filter_by_intensity.py"
METHOD="${CMAP_METHOD:-otsu_or_bg:3}"
FORCE="${CMAP_FORCE:-0}"
LIMIT="${CMAP_SUBMIT_LIMIT:-}"

# Default: both production datasets (use _jude path so jobs find the data)
DEFAULT_OUTPUT_BASES=(
    "/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/4_18_25"
    "/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/output/4_24_25_CGN_6_10_2"
)
if [[ -n "${CMAP_OUTPUT_BASES:-}" ]]; then
    read -ra OUTPUT_BASES <<< "$CMAP_OUTPUT_BASES"
else
    OUTPUT_BASES=("${DEFAULT_OUTPUT_BASES[@]}")
fi

LOG_DIR="$PROJECT_ROOT/logs/generate_bg_stats"
mkdir -p "$LOG_DIR"

if [[ ! -f "$FILTER_PY" ]]; then
    echo "ERROR: filter script not found: $FILTER_PY" >&2
    exit 1
fi

n_sub=0
n_skip=0
n_already=0

for out_base in "${OUTPUT_BASES[@]}"; do
    if [[ ! -d "$out_base" ]]; then
        echo "SKIP (not accessible): $out_base"
        continue
    fi

    dataset=$(basename "$out_base")
    echo ""
    echo "=== Dataset: $dataset ($out_base) ==="

    for sample_dir in "$out_base"/*/; do
        [[ -d "$sample_dir" ]] || continue
        sample=$(basename "$sample_dir")

        # Determine variant: prefer union_488_560, fall back to filtered_642
        if [[ -f "$sample_dir/union_488_560_combined.tif" ]]; then
            variant="union_488_560"
            bg_csv="$sample_dir/cell_qc_union_488_560/bg_stats.csv"
        elif [[ -f "$sample_dir/filtered_642_combined.tif" ]]; then
            variant="filtered_642"
            bg_csv="$sample_dir/cell_qc/bg_stats.csv"
        else
            echo "  SKIP (no combined TIFF): $sample"
            n_skip=$((n_skip + 1))
            continue
        fi

        if [[ -f "$bg_csv" ]] && [[ "$FORCE" != "1" ]]; then
            echo "  DONE (bg_stats.csv exists): $sample [$variant]"
            n_already=$((n_already + 1))
            continue
        fi

        if [[ -n "$LIMIT" ]] && [[ "$n_sub" -ge "$LIMIT" ]]; then
            echo "  Stopping at CMAP_SUBMIT_LIMIT=$LIMIT"
            break 2
        fi

        job_name="bgstats_${dataset:0:6}_${sample}"
        echo "  Submit: $sample [$variant]"

        bsub \
            -J "$job_name" \
            -q standard \
            -n 1 \
            -R "rusage[mem=32000] span[hosts=1]" \
            -W 30 \
            -o "$LOG_DIR/${dataset}_${sample}_%J.out" \
            -e "$LOG_DIR/${dataset}_${sample}_%J.err" \
            bash -lc \
            "set -euo pipefail
             eval \"\$(conda shell.bash hook)\"
             conda activate cmap
             cd \"$PROJECT_ROOT\"
             exec python3 -u \"$FILTER_PY\" \
               --output-dir \"$sample_dir\" \
               --from-volume \
               --variant \"$variant\" \
               --method \"$METHOD\" \
               --force"

        n_sub=$((n_sub + 1))
    done
done

echo ""
echo "========================================"
echo "Submitted : $n_sub job(s)"
echo "Already done : $n_already sample(s)"
echo "Skipped (no TIFF): $n_skip sample(s)"
echo "Logs: $LOG_DIR"
echo ""
echo "Monitor:  bjobs -J 'bgstats_*'"
echo "Check on finish:"
echo "  grep -r 'Wrote bg_stats' $LOG_DIR"
echo "  grep -r 'ERROR' $LOG_DIR"
