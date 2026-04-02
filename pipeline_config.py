"""Shared paths and Cellpose settings for the DAPI-only 2D pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TIF_PLANES_DIR = OUTPUT_DIR / "tif_planes"
SEGMENTATION_2D_DIR = OUTPUT_DIR / "segmentation_2D_planes"
SEGMENTATION_2D_STACKED_DIR = OUTPUT_DIR / "segmentation_2D_stacked"
SEGMENTATION_3D_DIR = OUTPUT_DIR / "segmentation_3D_masks"

# Cellpose 4.x default 2D pretrained checkpoint (Cellpose-SAM).
CELLPOSE_PRETRAINED_MODEL = "cpsam"
