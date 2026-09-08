"""Geometry helpers for Napari shapes and CMAP full-volume coordinates."""

from __future__ import annotations

import numpy as np

from .models import CellBoxRecord


def box_center_size_to_corners_yx(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> np.ndarray:
    """Return 2D rectangle vertices in Napari `(y, x)` order."""

    half_w = width / 2.0
    half_h = height / 2.0
    return np.array(
        [
            [center_y - half_h, center_x - half_w],
            [center_y - half_h, center_x + half_w],
            [center_y + half_h, center_x + half_w],
            [center_y + half_h, center_x - half_w],
        ],
        dtype=np.float64,
    )


def corners_yx_to_box(corners_yx: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 2D Napari rectangle vertices to `(center_x, center_y, width, height)`."""

    corners = np.asarray(corners_yx, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f"Expected 2D rectangle with shape (4, 2), got {corners.shape}")
    ys = corners[:, 0]
    xs = corners[:, 1]
    min_y, max_y = float(ys.min()), float(ys.max())
    min_x, max_x = float(xs.min()), float(xs.max())
    return (
        (min_x + max_x) / 2.0,
        (min_y + max_y) / 2.0,
        max_x - min_x,
        max_y - min_y,
    )


def corners_zyx_to_missing_box(corners_zyx: np.ndarray) -> dict[str, float | int]:
    """Convert a 3D Napari rectangle on one Z slice to CSV-ready box values."""

    corners = np.asarray(corners_zyx, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError(f"Expected 3D rectangle with shape (4, 3), got {corners.shape}")
    z_values = corners[:, 0]
    yx = corners[:, 1:3]
    center_x, center_y, width, height = corners_yx_to_box(yx)
    return {
        "z_index": int(round(float(np.median(z_values)))),
        "x0": float(yx[:, 1].min()),
        "x1": float(yx[:, 1].max()),
        "y0": float(yx[:, 0].min()),
        "y1": float(yx[:, 0].max()),
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }


def corners_zyx_to_full_z_rectangle_yx(corners_zyx: np.ndarray) -> np.ndarray:
    """Return the axis-aligned `(y, x)` footprint of a 3D rectangle.

    Dropping the Z column turns a single-slice missing-cell box into a 2D
    shape. Placed on a 2D Shapes layer inside the 3D viewer, napari broadcasts
    it across every Z slice, so the reviewer keeps seeing where they drew after
    scrolling away from the slice they drew on.
    """

    corners = np.asarray(corners_zyx, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError(f"Expected 3D rectangle with shape (4, 3), got {corners.shape}")
    ys = corners[:, 1]
    xs = corners[:, 2]
    min_y, max_y = float(ys.min()), float(ys.max())
    min_x, max_x = float(xs.min()), float(xs.max())
    return np.array(
        [
            [min_y, min_x],
            [min_y, max_x],
            [max_y, max_x],
            [max_y, min_x],
        ],
        dtype=np.float64,
    )


def cell_box_to_point(box: CellBoxRecord) -> tuple[float, float, float]:
    """Return the cell centroid in Napari layer coordinate order `(z, y, x)`."""

    return box.center_zyx


def cell_box_to_slice_rectangle(box: CellBoxRecord, z_index: int | None = None) -> np.ndarray:
    """Represent a 3D cell crop as a rectangle on one Z slice in `(z, y, x)` order."""

    z = float(z_index if z_index is not None else round(box.center_zyx[0]))
    return np.array(
        [
            [z, box.y0, box.x0],
            [z, box.y0, box.x1],
            [z, box.y1, box.x1],
            [z, box.y1, box.x0],
        ],
        dtype=np.float64,
    )
