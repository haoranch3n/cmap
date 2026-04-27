"""Shared paths and settings for the segmentation module."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(key: str, default: Path) -> Path:
    value = os.environ.get(key)
    if value:
        return Path(value).expanduser().resolve()
    return default


DATA_DIR = _path_from_env("PIPELINE_DATA_DIR", PROJECT_ROOT / "data")
OUTPUT_DIR = _path_from_env("PIPELINE_OUTPUT_DIR", PROJECT_ROOT / "output")


def strip_path_shared_with_output_mirror(rel: str) -> str:
    """
    Drop leading path components that match OUTPUT_DIR relative to output/, so
    stage folders are not nested twice under the same prefix.
    """
    if not rel or rel == ".":
        return ""
    try:
        mirror = Path(OUTPUT_DIR).relative_to(PROJECT_ROOT / "output").as_posix()
    except ValueError:
        return rel.replace("\\", "/")
    if not mirror or mirror == ".":
        return rel.replace("\\", "/")
    rel_parts = [p for p in str(rel).replace("\\", "/").split("/") if p]
    mirror_parts = [p for p in mirror.split("/") if p]
    idx = 0
    while idx < len(rel_parts) and idx < len(mirror_parts) and rel_parts[idx] == mirror_parts[idx]:
        idx += 1
    return "/".join(rel_parts[idx:])


TIF_PLANES_DIR = OUTPUT_DIR / "tif_planes"
SEGMENTATION_2D_DIAMETERS_DIR = OUTPUT_DIR / "segmentation_2D_diameters"
SEGMENTATION_2D_DIR = OUTPUT_DIR / "segmentation_2D_planes"
SEGMENTATION_2D_STACKED_DIR = OUTPUT_DIR / "segmentation_2D_stack"
SEGMENTATION_3D_DIR = OUTPUT_DIR / "segmentation_3D_masks"

CELLPOSE_PRETRAINED_MODEL = "cpsam"
CELLPOSE_DIAMETERS = list(range(50, 110, 10))
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = 0.0

AREA_THRESHOLD = int(np.pi * 20 ** 2)
SPLIT_COVERAGE_THRESHOLD = 0.9

JI_THRESHOLD = 0.1
MIN_CELL_Z_SPAN = 5
MIN_CELL_VOLUME_3D = 100
MAX_AREA_CHANGE_RATIO = 3.0

