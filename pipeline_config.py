"""Shared paths and settings for the 2D/3D segmentation pipeline."""
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    if v:
        return Path(v).expanduser().resolve()
    return default


DATA_DIR = _path_from_env("PIPELINE_DATA_DIR", PROJECT_ROOT / "data")
OUTPUT_DIR = _path_from_env("PIPELINE_OUTPUT_DIR", PROJECT_ROOT / "output")

# Per-image runs: set PIPELINE_OUTPUT_DIR to output/<REL>/<image_stem>/ so you get e.g.
#   .../488nm_crop/tif_planes/
#   .../488nm_crop/segmentation_2D_stack/
#   .../488nm_crop/segmentation_3D_masks/


def strip_path_shared_with_output_mirror(rel: str) -> str:
    """
    Drop leading path components that match OUTPUT_DIR relative to output/, so
    stage folders (tif_planes, segmentation_2D_*, ...) are not nested again
    under the same <REL>/<stem> that OUTPUT_DIR already encodes.
    """
    if not rel or rel == ".":
        return ""
    try:
        mirror = Path(OUTPUT_DIR).relative_to(PROJECT_ROOT / "output").as_posix()
    except ValueError:
        return rel.replace("\\", "/")
    if not mirror or mirror == ".":
        return rel.replace("\\", "/")
    ra = [p for p in str(rel).replace("\\", "/").split("/") if p]
    mb = [p for p in mirror.split("/") if p]
    i = 0
    while i < len(ra) and i < len(mb) and ra[i] == mb[i]:
        i += 1
    return "/".join(ra[i:])


TIF_PLANES_DIR = OUTPUT_DIR / "tif_planes"
SEGMENTATION_2D_DIAMETERS_DIR = OUTPUT_DIR / "segmentation_2D_diameters"
SEGMENTATION_2D_DIR = OUTPUT_DIR / "segmentation_2D_planes"
SEGMENTATION_2D_STACKED_DIR = OUTPUT_DIR / "segmentation_2D_stack"
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
JI_THRESHOLD = 0.1

# -- 3D post-processing -------------------------------------------------------
MIN_CELL_Z_SPAN = 5                  # remove cells spanning fewer than N z-slices
MIN_CELL_VOLUME_3D = 100              # voxels
MAX_AREA_CHANGE_RATIO = 3.0           # max consecutive-slice area fold-change
