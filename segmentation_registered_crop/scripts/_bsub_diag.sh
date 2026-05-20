#!/usr/bin/env bash
#BSUB -J cpsam_diag
#BSUB -q rhel88_gpu
#BSUB -n 2
#BSUB -R "rusage[mem=16000] span[hosts=1]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -W 0:30
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/diag_%J.out
#BSUB -e /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_registered_crop/logs/diag_%J.err

eval "$(conda shell.bash hook)"
conda activate cmap
cd /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap
python segmentation_registered_crop/scripts/_diag_cpsam.py
