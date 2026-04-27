"""
iGluSnFR Analysis Toolbox v3 - Configuration

All tunable defaults now live in ``config_manager.DEFAULT_CONFIG``.
This file re-exports them as module-level constants for backward
compatibility and adds path / environment helpers.
"""

import os

# Pull defaults from the single source of truth
from config_manager import DEFAULT_CONFIG as _DC

# =============================================================================
# Paths
# =============================================================================

TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
CONDA_ENVS_DIR = os.path.join(HOME_DIR, ".conda", "envs")

# =============================================================================
# Default Processing Parameters (from config_manager)
# =============================================================================

_proc = _DC["processing"]
DEFAULT_BKG_PERCENTILE = _proc["bkg_percentile"]
DEFAULT_MAX_DISTANCE = _proc["max_distance"]
DEFAULT_MIN_AREA = _proc["min_area"]
DEFAULT_MAX_AREA = _proc["max_area"]
DEFAULT_SNR_THRESHOLD = _proc["snr_threshold"]
DEFAULT_IOU_THRESHOLD = _proc["iou_threshold"]
DEFAULT_SNR_START_FRAME = _proc["snr_start_frame"]
DEFAULT_SNR_END_FRAME = _proc["snr_end_frame"]

# =============================================================================
# Parallel Processing
# =============================================================================

DEFAULT_N_JOBS = _proc["n_jobs"]
PARALLEL_BACKEND = _proc["parallel_backend"]

# =============================================================================
# UI Settings (from config_manager)
# =============================================================================

_ui = _DC["ui"]
FONT_SIZE_NORMAL = _ui["font_size"]
FONT_SIZE_LARGE = _ui["font_size_large"]
FONT_SIZE_TITLE = _ui["font_size_title"]
MIN_PANEL_WIDTH = _ui["min_panel_width"]

# =============================================================================
# Conda Environment Names
# =============================================================================

# Primary environments (created by setup.sh)
VIEWER_ENV = "iglusnfr_viewer"
PROCESSING_ENV = "iglusnfr_processing"

# Fallback environments (if user has existing ones)
VIEWER_ENV_FALLBACK = "napari_viewer"
PROCESSING_ENV_FALLBACK = "ilastik"

# All processing environments to check (in order of preference)
PROCESSING_ENVS = [PROCESSING_ENV, PROCESSING_ENV_FALLBACK]
VIEWER_ENVS = [VIEWER_ENV, VIEWER_ENV_FALLBACK]


def get_env_python(env_name):
    """Get python executable path for a conda environment."""
    python_path = os.path.join(CONDA_ENVS_DIR, env_name, "bin", "python")
    if os.path.exists(python_path):
        return python_path
    return None


def get_best_viewer_env():
    """Get the best available viewer environment."""
    for env in VIEWER_ENVS:
        if get_env_python(env):
            return env
    return None


def get_best_processing_env():
    """Get the best available processing environment."""
    for env in PROCESSING_ENVS:
        if get_env_python(env):
            return env
    return None
