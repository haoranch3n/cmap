from __future__ import annotations

import numpy as np
import pytest

from cmap_cell_exemption_plugin.geometry import (
    box_center_size_to_corners_yx,
    corners_yx_to_box,
    corners_zyx_to_full_z_rectangle_yx,
    corners_zyx_to_missing_box,
)


def test_2d_box_round_trip() -> None:
    corners = box_center_size_to_corners_yx(10.0, 20.0, 6.0, 8.0)
    center_x, center_y, width, height = corners_yx_to_box(corners)
    assert center_x == pytest.approx(10.0)
    assert center_y == pytest.approx(20.0)
    assert width == pytest.approx(6.0)
    assert height == pytest.approx(8.0)


def test_3d_missing_box_uses_z_slice() -> None:
    corners = np.asarray(
        [
            [5.0, 10.0, 20.0],
            [5.0, 10.0, 30.0],
            [5.0, 18.0, 30.0],
            [5.0, 18.0, 20.0],
        ]
    )
    values = corners_zyx_to_missing_box(corners)
    assert values["z_index"] == 5
    assert values["x0"] == pytest.approx(20.0)
    assert values["x1"] == pytest.approx(30.0)
    assert values["y0"] == pytest.approx(10.0)
    assert values["y1"] == pytest.approx(18.0)
    assert values["center_x"] == pytest.approx(25.0)
    assert values["center_y"] == pytest.approx(14.0)


def test_bad_missing_box_shape_raises() -> None:
    with pytest.raises(ValueError):
        corners_zyx_to_missing_box(np.zeros((4, 2)))


def test_full_z_rectangle_drops_z_and_keeps_xy_footprint() -> None:
    corners = np.asarray(
        [
            [5.0, 10.0, 20.0],
            [5.0, 10.0, 30.0],
            [5.0, 18.0, 30.0],
            [5.0, 18.0, 20.0],
        ]
    )
    rect = corners_zyx_to_full_z_rectangle_yx(corners)
    assert rect.shape == (4, 2)
    assert rect[:, 0].min() == pytest.approx(10.0)
    assert rect[:, 0].max() == pytest.approx(18.0)
    assert rect[:, 1].min() == pytest.approx(20.0)
    assert rect[:, 1].max() == pytest.approx(30.0)


def test_full_z_rectangle_is_axis_aligned_whatever_the_draw_order() -> None:
    # napari stores vertices in draw order, so a box dragged from its
    # bottom-right corner arrives reversed; the guide must still be the same
    # axis-aligned footprint.
    corners = np.asarray(
        [
            [7.0, 18.0, 30.0],
            [7.0, 18.0, 20.0],
            [7.0, 10.0, 20.0],
            [7.0, 10.0, 30.0],
        ]
    )
    rect = corners_zyx_to_full_z_rectangle_yx(corners)
    expected = np.asarray([[10.0, 20.0], [10.0, 30.0], [18.0, 30.0], [18.0, 20.0]])
    assert np.allclose(rect, expected)


def test_full_z_rectangle_rejects_2d_input() -> None:
    with pytest.raises(ValueError):
        corners_zyx_to_full_z_rectangle_yx(np.zeros((4, 2)))
