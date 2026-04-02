"""Shared paths and settings for the 2D/3D segmentation pipeline."""
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TIF_PLANES_DIR = OUTPUT_DIR / "tif_planes"
SEGMENTATION_2D_DIAMETERS_DIR = OUTPUT_DIR / "segmentation_2D_diameters"
SEGMENTATION_2D_DIR = OUTPUT_DIR / "segmentation_2D_planes"
SEGMENTATION_2D_STACKED_DIR = OUTPUT_DIR / "segmentation_2D_stacked"
SEGMENTATION_3D_DIR = OUTPUT_DIR / "segmentation_3D_masks"

# -- Cellpose 2D segmentation ------------------------------------------------
CELLPOSE_PRETRAINED_MODEL = "cpsam"
CELLPOSE_DIAMETERS = list(range(50, 110, 10))   # [50, 60, 70, 80, 90, 100]
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = 0.0

# -- Multiscale merge / filter -----------------------------------------------
AREA_THRESHOLD = int(np.pi * 20 ** 2)
SPLIT_COVERAGE_THRESHOLD = 0.9        # cumulative overlap to trigger cell splitting

# -- 3D matching --------------------------------------------------------------
JI_THRESHOLD = 0.2

# -- 3D post-processing -------------------------------------------------------
MIN_CELL_VOLUME_3D = 100              # voxels
MAX_AREA_CHANGE_RATIO = 3.0           # max consecutive-slice area fold-change
