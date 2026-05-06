"""Cell-crop loading for the DINOv2 pipeline.

The active producer of these files is ``features/crop_cells.py`` which writes
``output/<sample>/cell_boxing/cell_<NNNN>.tif`` of shape ``(Z, 5, Y, X)`` with
channels ``[642, 488, 560, Primary_Cell_Mask, Mask_Boundary]``.  We tolerate
older 4-channel crops (no mask boundary) and raw 3-channel volumes for
robustness, but always return only the 3 intensity channels plus an optional
binary primary-cell mask for downstream masking.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

INTENSITY_CHANNELS = 3
_CELL_ID_RE = re.compile(r"cell_(\d+)")


def parse_cell_id(filename: str) -> int | None:
    """Return the integer cell id from ``cell_0007.tif`` style names."""
    m = _CELL_ID_RE.match(filename)
    return int(m.group(1)) if m else None


def list_cell_crops(box_dir: Path) -> list[Path]:
    """Sorted list of ``cell_*.tif`` files under ``cell_boxing/``."""
    return sorted(Path(box_dir).glob("cell_*.tif"))


def read_cell_crop(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a 4-D cell crop and split into intensities + optional mask.

    Args:
        path: Path to ``cell_<NNNN>.tif``.

    Returns:
        Tuple ``(intensity, mask)`` where:
            - ``intensity`` is float32 of shape ``(Z, 3, Y, X)`` (channels
              ``[642, 488, 560]``).
            - ``mask`` is bool of shape ``(Z, Y, X)`` if a primary-cell mask
              channel was present, else ``None``.

    Raises:
        ValueError: If the file does not look like a cell crop produced by
            ``features/crop_cells.py``.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4-D (Z, C, Y, X), got shape {arr.shape} from {path}")
    n_channels = arr.shape[1]
    if n_channels not in (3, 4, 5):
        raise ValueError(
            f"Expected channel count in {{3, 4, 5}}, got C={n_channels} from {path}"
        )
    intensity = arr[:, :INTENSITY_CHANNELS, :, :].astype(np.float32, copy=False)
    mask: np.ndarray | None = None
    if n_channels >= 4:
        mask = arr[:, 3, :, :] > 0
    return intensity, mask
