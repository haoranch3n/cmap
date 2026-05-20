#!/bin/bash
#BSUB -J patch_seg_axes
#BSUB -q standard
#BSUB -n 1
#BSUB -R "rusage[mem=8000]"
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/patch_axes_%J.out
#BSUB -e /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/patch_axes_%J.err

PROJECT_ROOT=/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap
python3 "$PROJECT_ROOT/segmentation_registered_crop/scripts/patch_segmented_tif_axes.py"
