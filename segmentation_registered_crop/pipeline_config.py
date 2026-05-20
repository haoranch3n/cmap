"""Shared paths and settings for the segmentation_registered_crop pipeline."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


def _path_from_env(key: str, default: Path) -> Path:
    value = os.environ.get(key)
    if value:
        return Path(value).expanduser().resolve()
    return default


# Root directory of registered cell crops produced by Jorge's registration pipeline.
CROPS_ROOT = _path_from_env(
    "CROPS_ROOT",
    Path(
        "/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/"
        "CMAP_general/No_decon_tests/outputs/cell_crops/batch_processing"
    ),
)

# Root directory where pipeline outputs are written.
OUTPUT_ROOT = _path_from_env(
    "SEG_CROP_OUTPUT_ROOT",
    REPO_ROOT / "output_registered_crop_seg",
)

# Root directory of the existing output_no_deconv tree, which contains the
# per-channel segmentation_norm_bounds.csv files computed from full FOV images.
NORM_BOUNDS_ROOT = _path_from_env(
    "NORM_BOUNDS_ROOT",
    REPO_ROOT / "output_no_deconv",
)

# ---------------------------------------------------------------------------
# Channel configuration
# Channel indices in the registered (Z, C, Y, X) TIF:
#   0 = 642 nm (structural reference)
#   1 = 488 nm
#   2 = 560 nm
#   3, 4 = derived (not segmented)
# ---------------------------------------------------------------------------
SEGMENTATION_CHANNELS: dict[str, int] = {
    "ch642": 0,
    "ch488": 1,
    "ch560": 2,
}

# Maps channel name → folder name used in NORM_BOUNDS_ROOT tree.
CHANNEL_NORM_FOLDER: dict[str, str] = {
    "ch642": "642nm_crop",
    "ch488": "488nm_crop",
    "ch560": "560nm_crop",
}

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
# (lo_pct, hi_pct) used as fallback when no pre-computed CSV is found.
SEGMENTATION_GLOBAL_VOLUME_PERCENTILES: tuple[float, float] = (0.0, 99.99)
# Written by preprocessing/extract_channel.py and read by Cellpose 2D step.
SEGMENTATION_NORM_BOUNDS_CSV = "segmentation_norm_bounds.csv"
# Cellpose normalize=False because we pre-normalize from full FOV bounds.
CELLPOSE_EVAL_NORMALIZE = False

# ---------------------------------------------------------------------------
# Cellpose
# ---------------------------------------------------------------------------
CELLPOSE_PRETRAINED_MODEL = "cpsam"
CELLPOSE_DIAMETERS = list(range(50, 110, 10))  # [50, 60, 70, 80, 90, 100]
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = 0.0

# Plane filename tags used for discovery by the Cellpose 2D step.
SEG_PLANE_TAGS: tuple[str, ...] = ("ch642", "ch488", "ch560", "488", "560", "642", "DAPI", "seg")

# ---------------------------------------------------------------------------
# 2D segmentation merge / filter
# ---------------------------------------------------------------------------
AREA_THRESHOLD = int(np.pi * 20 ** 2)
SPLIT_COVERAGE_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# 3D assembly
# ---------------------------------------------------------------------------
JI_THRESHOLD = 0.1
MIN_CELL_Z_SPAN = 5
MIN_CELL_VOLUME_3D = 100
MAX_AREA_CHANGE_RATIO = 3.0


def strip_path_shared_with_output_mirror(rel: str) -> str:
    """No-op pass-through (flat output layout — no OUTPUT_DIR nesting to strip)."""
    return rel.replace("\\", "/") if rel else ""
