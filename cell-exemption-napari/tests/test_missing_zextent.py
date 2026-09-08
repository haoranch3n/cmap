"""Full-Z guide for missing-cell boxes.

These tests drive real napari ``Shapes`` layers but never build a viewer or a
Qt application, so they run headless on a login node.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

from napari.components.dims import Dims  # noqa: E402
from napari.layers import Shapes  # noqa: E402

from cmap_cell_exemption_plugin.widget import CellExemptionWidget  # noqa: E402


def _rect_zyx(z: float, y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    return np.asarray(
        [[z, y0, x0], [z, y0, x1], [z, y1, x1], [z, y1, x0]], dtype=np.float64
    )


def _widget_with_layers(rects: list[np.ndarray]) -> CellExemptionWidget:
    """Build a widget shell without running Qt's ``__init__``."""
    widget = CellExemptionWidget.__new__(CellExemptionWidget)
    widget.label_data = np.zeros((50, 200, 200), dtype=np.int32)
    widget._missing_layer = (
        Shapes(rects, shape_type="rectangle", ndim=3) if rects else Shapes(ndim=3)
    )
    widget._missing_zextent_layer = Shapes(ndim=2)
    return widget


def test_guide_mirrors_each_missing_box_as_one_2d_shape() -> None:
    widget = _widget_with_layers(
        [_rect_zyx(5, 10, 30, 20, 40), _rect_zyx(41, 100, 120, 60, 90)]
    )
    widget._refresh_missing_zextent()

    guide = widget._missing_zextent_layer
    assert len(guide.data) == 2
    assert all(shape.shape == (4, 2) for shape in guide.data)
    assert np.allclose(guide.data[0][:, 0].min(), 10.0)
    assert np.allclose(guide.data[0][:, 1].max(), 40.0)


def test_guide_is_visible_on_every_z_slice() -> None:
    widget = _widget_with_layers([_rect_zyx(5, 10, 30, 20, 40)])
    widget._refresh_missing_zextent()
    guide = widget._missing_zextent_layer

    for z in (0, 5, 23, 49):
        dims = Dims(
            ndim=3,
            ndisplay=2,
            range=((0, 50, 1), (0, 200, 1), (0, 200, 1)),
            point=(z, 0, 0),
        )
        guide._slice_dims(dims)
        assert list(guide._indices_view) == [0], f"guide hidden at z={z}"


def test_guide_outline_has_no_fill() -> None:
    widget = _widget_with_layers([_rect_zyx(5, 10, 30, 20, 40)])
    widget._refresh_missing_zextent()
    guide = widget._missing_zextent_layer

    assert guide.face_color[0][3] == pytest.approx(0.0)
    assert 0.0 < guide.edge_color[0][3] < 1.0


def test_guide_clears_when_all_boxes_are_deleted() -> None:
    widget = _widget_with_layers([_rect_zyx(5, 10, 30, 20, 40)])
    widget._refresh_missing_zextent()
    assert len(widget._missing_zextent_layer.data) == 1

    widget._missing_layer.data = []
    widget._refresh_missing_zextent()
    assert len(widget._missing_zextent_layer.data) == 0


def test_guide_refresh_is_a_no_op_without_a_layer() -> None:
    widget = _widget_with_layers([_rect_zyx(5, 10, 30, 20, 40)])
    widget._missing_zextent_layer = None
    widget._refresh_missing_zextent()  # must not raise
