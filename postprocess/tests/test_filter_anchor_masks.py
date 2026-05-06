"""Synthetic ZYX tests for triple-overlap mask filtering.

Run from repo root::

    pytest postprocess/tests/test_filter_anchor_masks.py -q
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from postprocess.filter_642_mask import (  # noqa: E402
    filter_642_mask,
    filter_anchor_by_other_two,
)


def _silent_filter(fn, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return fn(*args, **kwargs)


def test_anchor_kept_only_if_overlaps_both_others():
    z, y, x = 4, 8, 8
    anchor = np.zeros((z, y, x), dtype=np.uint16)
    oa = np.zeros((z, y, x), dtype=np.uint16)
    ob = np.zeros((z, y, x), dtype=np.uint16)
    # Label 1: overlaps both -> keep
    anchor[0, 0:2, 0:2] = 1
    oa[0, 1, 1] = 1
    ob[0, 0, 0] = 1
    # Label 2: overlaps only oa -> remove
    anchor[1, 2:4, 2:4] = 2
    oa[1, 3, 3] = 1
    # Label 3: overlaps only ob -> remove
    anchor[2, 4:6, 4:6] = 3
    ob[2, 5, 5] = 1

    out = _silent_filter(
        filter_anchor_by_other_two,
        anchor,
        oa,
        ob,
        anchor_label="A",
        other_a_label="B",
        other_b_label="C",
    )
    assert set(np.unique(out)) == {0, 1}
    assert np.all(out[anchor == 2] == 0)
    assert np.all(out[anchor == 3] == 0)


def test_filter_642_mask_matches_permutation_of_anchors():
    z, y, x = 3, 6, 6
    m642 = np.zeros((z, y, x), dtype=np.uint16)
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m642[0, 0:2, 0:2] = 10
    m488[0, 1, 1] = 1
    m560[0, 0, 0] = 1
    m642[1, 3:5, 3:5] = 20
    m488[1, 4, 4] = 1

    f642 = _silent_filter(filter_642_mask, m642, m488, m560)
    f642_b = _silent_filter(
        filter_anchor_by_other_two,
        m642,
        m488,
        m560,
        anchor_label="642",
        other_a_label="488",
        other_b_label="560",
    )
    np.testing.assert_array_equal(f642, f642_b)

    # 560 anchor: label 1 overlaps 488 and 642; label 2 overlaps 488 only -> drop label 2
    m560_alt = np.zeros_like(m560)
    m560_alt[0, 1, 1] = 1  # overlaps 488 at (0,1,1) and 642 label 10 region
    m560_alt[2, 0, 0] = 2  # overlaps 488 at z=2; m642 is zero on this slice -> drop label 2
    m488[2, 0, 0] = 1
    f560_alt = _silent_filter(
        filter_anchor_by_other_two,
        m560_alt,
        m488,
        m642,
        anchor_label="560",
        other_a_label="488",
        other_b_label="642",
    )
    assert set(np.unique(f560_alt)) <= {0, 1}
    assert np.all(f560_alt[m560_alt == 2] == 0)


def test_488_anchor_symmetric():
    z, y, x = 2, 5, 5
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m642 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 1:3, 1:3] = 7
    m560[0, 2, 2] = 1
    m642[0, 1, 1] = 1
    m488[1, 0:2, 0:2] = 8
    m560[1, 0, 0] = 1

    out = _silent_filter(
        filter_anchor_by_other_two,
        m488,
        m560,
        m642,
        anchor_label="488",
        other_a_label="560",
        other_b_label="642",
    )
    assert set(np.unique(out)) == {0, 7}
    assert np.all(out[m488 == 8] == 0)
