#!/usr/bin/env bash
# Submit the image-level annotation verification (reads label channel of the
# combined TIFFs) to LSF. Run from the login node:
#   bash cell-exemption-napari/_bsub_verify_annotations.sh
set -euo pipefail

PROJECT_ROOT="/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap"
PLUGIN_DIR="$PROJECT_ROOT/cell-exemption-napari"
LOG_DIR="$PLUGIN_DIR"
QUEUE="${CMAP_QUEUE:-standard}"

bsub \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=12288] span[hosts=1]" \
  -W 30 \
  -J "verify_ann" \
  -o "$LOG_DIR/_verify_annotations_image.%J.out" \
  bash -lc "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate cmap
    cd \"$PLUGIN_DIR\"
    python _verify_annotations_image.py
  "
echo "Submitted. Watch with: bjobs -J verify_ann ; tail -f $LOG_DIR/_verify_annotations_image.*.out"
