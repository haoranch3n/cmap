"""Re-export project paths for legacy imports."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline_config import (  # noqa: E402
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    TIF_PLANES_DIR,
    SEGMENTATION_2D_DIR,
    CELLPOSE_PRETRAINED_MODEL,
    AREA_THRESHOLD,
)

BASE_DIR = OUTPUT_DIR
