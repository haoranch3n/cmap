"""Synthetic ZYX tests for IoMin label stitching in
``postprocess/union_488_560_mask.py``.

Run from repo root::

    pytest postprocess/tests/test_union_488_560_iomin.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from postprocess.union_488_560_mask import union_488_560_labels  # noqa: E402


def _component_ids(out: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(out) if v != 0}


def test_full_inclusion_fuses_into_one_id():
    """488 label fully inside a much larger 560 label -> one fused ID
    covering both the 488 region and the surrounding 560 region.
    IoU would fail here (low value); IoMin = 1.0.
    """
    z, y, x = 4, 12, 12
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    # 488 cell: 1x2x2 = 4 voxels
    m488[1, 4:6, 4:6] = 1
    # 560 cell wraps around it: 4x10x10 = 400 voxels, including the 488 region
    m560[:, 1:11, 1:11] = 7

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 1, diag
    assert diag["paired_1_1"] == 1
    assert _component_ids(out) == {1}
    # Both regions painted with the same ID
    assert int(out[1, 4, 4]) == 1
    assert int(out[0, 1, 1]) == 1


def test_disjoint_keeps_separate_ids():
    """488 and 560 labels with no spatial overlap -> two distinct IDs."""
    z, y, x = 3, 10, 10
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0:2, 0:2] = 5
    m560[2, 7:9, 7:9] = 11

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 2, diag
    assert diag["488_only"] == 1
    assert diag["560_only"] == 1
    assert diag["paired_1_1"] == 0
    assert _component_ids(out) == {1, 2}


def test_488_oversegmented_into_one_560_cell_fuses_to_one_id():
    """Two adjacent 488 labels both contained in one 560 label -> one fused ID."""
    z, y, x = 2, 8, 8
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 1:3, 1:3] = 1   # 4 voxels
    m488[0, 1:3, 5:7] = 2   # 4 voxels (NOT touching label 1)
    m560[0, 1:3, 1:7] = 9   # 12 voxels covering both 488 fragments
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 1, diag
    assert diag["n_to_m"] == 1
    assert _component_ids(out) == {1}


def test_two_488_cells_with_separate_560_partners_stay_separate():
    """Two 488 cells touching/adjacent in space, each with its own clean 560
    partner. The old pure-OR + 26-conn algo would over-merge into 1 component;
    IoMin stitching must produce 2 distinct IDs.
    """
    z, y, x = 1, 8, 8
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    # Cell A: 488 label 1 + 560 label 1 share voxels.
    m488[0, 0:3, 0:3] = 1
    m560[0, 0:3, 0:3] = 1
    # Cell B: 488 label 2 + 560 label 2, adjacent to A but its own match.
    m488[0, 0:3, 3:6] = 2
    m560[0, 0:3, 3:6] = 2

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 2, diag
    assert diag["paired_1_1"] == 2
    assert diag["n_to_m"] == 0
    assert _component_ids(out) == {1, 2}
    assert int(out[0, 0, 0]) != int(out[0, 0, 4])


def test_sub_tau_overlap_does_not_fuse():
    """A tiny shared corner between large 488 and 560 labels (IoMin << tau):
    they remain separate components and contribute conflict_voxels.
    """
    z, y, x = 1, 20, 20
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    # 488 label: 200 voxels
    m488[0, 0:10, 0:20] = 4
    # 560 label: 200 voxels, overlapping only 1 voxel of 488
    m560[0, 9:19, 0:20] = 6
    # Shared overlap is m488==4 AND m560==6 at row 9 -> 20 voxels.
    # IoMin = 20/200 = 0.1 < 0.2 -> no edge.

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 2
    assert diag["edges"] == 0
    assert diag["paired_1_1"] == 0
    assert diag["conflict_voxels"] > 0


def test_min_label_voxels_drops_tiny_labels():
    """Small labels below min_label_voxels are dropped (their voxels become 0)."""
    z, y, x = 1, 6, 6
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0, 0] = 1                  # 1 voxel - tiny
    m488[0, 1:4, 1:4] = 2               # 9 voxels
    m560[0, 1:4, 1:4] = 5               # 9 voxels (matches label 2)

    out, diag = union_488_560_labels(m488, m560, tau=0.2, min_label_voxels=5)
    assert diag["N488"] == 1   # the 1-voxel label is dropped from 488
    assert diag["N560"] == 1
    assert diag["K"] == 1      # only one fused cell remains
    # The 1-voxel 488 label has been dropped -> that voxel is background
    assert int(out[0, 0, 0]) == 0


def test_dtype_escalates_to_uint32_when_K_exceeds_uint16_max():
    """Synthetic many-label case to validate uint32 escalation path."""
    z, y, x = 1, 1, 70000
    m488 = np.arange(1, x + 1, dtype=np.int64).reshape(z, y, x)
    m560 = np.zeros((z, y, x), dtype=np.int64)
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 70000
    assert out.dtype == np.uint32


def test_contiguous_relabel():
    """Output IDs are always contiguous 1..K (no gaps)."""
    z, y, x = 2, 6, 6
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    # Use non-contiguous Cellpose-style IDs
    m488[0, 0:2, 0:2] = 17
    m488[1, 4:6, 4:6] = 42
    m560[0, 0:2, 0:2] = 99   # matches 488 label 17
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    used = sorted(_component_ids(out))
    assert used == list(range(1, diag["K"] + 1))


def test_background_stays_zero():
    z, y, x = 2, 5, 5
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 1, 1] = 1
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    bg = (m488 == 0) & (m560 == 0)
    assert int(out[bg].max()) == 0
