from __future__ import annotations

import numpy as np
import pytest

from cmap_cell_exemption_plugin.geometry import (
    box_center_size_to_corners_yx,
    corners_yx_to_box,
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
