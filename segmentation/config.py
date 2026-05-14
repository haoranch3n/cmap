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

# For 4D (Z,C,Y,X) inputs: additional signal channels to export as normalized float32 planes.
# Each tuple is (source_channel_index, filename_tag).  C=0 is always DAPI.
SIGNAL_CHANNELS_EXPORT: tuple[tuple[int, str], ...] = ((1, "488"), (2, "560"), (3, "642"))
SIGNAL_VOLUME_PERCENTILES: tuple[float, float] = (0.0, 99.99)

SEGMENTATION_2D_DIAMETERS_DIR = OUTPUT_DIR / "segmentation_2D_diameters"
SEGMENTATION_2D_DIR = OUTPUT_DIR / "segmentation_2D_planes"
SEGMENTATION_2D_STACKED_DIR = OUTPUT_DIR / "segmentation_2D_stack"
SEGMENTATION_3D_DIR = OUTPUT_DIR / "segmentation_3D_masks"

CELLPOSE_PRETRAINED_MODEL = "cpsam"
CELLPOSE_DIAMETERS = list(range(50, 110, 10))
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = 0.0
# If False, Cellpose's built-in per-input normalization is skipped and
# SEGMENTATION_GLOBAL_VOLUME_PERCENTILES is applied as a global per-channel affine instead.
CELLPOSE_EVAL_NORMALIZE = False
# (lo_pct, hi_pct) for the global volume normalization applied before Cellpose.
SEGMENTATION_GLOBAL_VOLUME_PERCENTILES: tuple[float, float] = (0.0, 99.99)
# Written by preprocess/separate_channels.py inside each tif_planes volume subdirectory.
# Read by cellpose2d/segmentation_cellpose_2d.py to skip recomputing bounds.
# Format: one comment row, then header row, then one row per channel: channel_index,lo,hi
SEGMENTATION_NORM_BOUNDS_CSV = "segmentation_norm_bounds.csv"
# Suffixes used when naming per-Z plane TIFFs; discovery and stem-stripping uses this list.
# Single-channel volumes get a tag inferred from the source filename (e.g. "488", "560", "642").
# Multi-channel volumes use "DAPI" for channel 0.  "seg" is the generic fallback.
SEG_PLANE_TAGS: tuple[str, ...] = ("488", "560", "642", "DAPI", "seg")

AREA_THRESHOLD = int(np.pi * 20 ** 2)
SPLIT_COVERAGE_THRESHOLD = 0.9

JI_THRESHOLD = 0.1
MIN_CELL_Z_SPAN = 5
MIN_CELL_VOLUME_3D = 100
MAX_AREA_CHANGE_RATIO = 3.0
