#!/bin/bash
#BSUB -J combine_masks
#BSUB -q standard
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/combine_masks_%J.out
#BSUB -e /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/combine_masks_%J.err

PROJECT_ROOT=/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap
python3 "$PROJECT_ROOT/segmentation_registered_crop/scripts/combine_channel_masks.py" "$@"
