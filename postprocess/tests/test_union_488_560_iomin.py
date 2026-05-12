"""Synthetic ZYX tests for the strict-overlap IoMin label-stitching merge in
``postprocess/union_488_560_mask.py``.

The merge is strict-by-construction: every emitted cell satisfies

  - gate-1 (mixed-channel): n_488 >= 1 AND n_560 >= 1
  - gate-2 (pairwise IoMin closure): every internal cross-pair with raw
    voxel intersection > 0 has IoMin >= tau

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

from postprocess.union_488_560_mask import (  # noqa: E402
    _compute_label_map,
    union_488_560_labels,
)


def _component_ids(out: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(out) if v != 0}


def _assert_strict_invariants(out: np.ndarray, diag: dict, m488: np.ndarray, m560: np.ndarray, tau: float) -> None:
    """Regression: every emitted cell must satisfy gate-1 + gate-2."""
    label_map = diag.get("label_map", {})
    for cid, entry in label_map.items():
        assert entry["n_488"] >= 1, f"cell {cid} fails gate-1 (no 488): {entry}"
        assert entry["n_560"] >= 1, f"cell {cid} fails gate-1 (no 560): {entry}"
    wik = diag.get("worst_internal_iomin_kept", float("nan"))
    if diag["K"] > 0 and not np.isnan(wik):
        assert wik >= tau - 1e-9, (
            f"gate-2 violated: worst_internal_iomin_kept={wik} < tau={tau}"
        )


def test_full_inclusion_fuses_into_one_id():
    """488 label fully inside a much larger 560 label -> one fused ID.
    IoU would fail here (low value); IoMin = 1.0 so both gates pass.
    """
    z, y, x = 4, 12, 12
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[1, 4:6, 4:6] = 1
    m560[:, 1:11, 1:11] = 7

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 1, diag
    assert diag["paired_1_1"] == 1
    assert _component_ids(out) == {1}
    assert int(out[1, 4, 4]) == 1
    assert int(out[0, 1, 1]) == 1
    _assert_strict_invariants(out, diag, m488, m560, tau=0.2)


def test_disjoint_labels_are_dropped_under_strict():
    """488-only and 560-only cells (no spatial overlap) are dropped entirely
    under strict construction. Each is its own UF component; gate-1 fails.
    """
    z, y, x = 3, 10, 10
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0:2, 0:2] = 5
    m560[2, 7:9, 7:9] = 11

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 0, diag
    assert diag["n_dropped_unpaired"] == 2, diag
    assert _component_ids(out) == set()


def test_488_oversegmented_dropped_under_strict_1_1():
    """Two adjacent 488 labels both contained in one 560 label form a single
    UF component with c488=2, c560=1. Under the default strict-1-1 policy
    (gate-3), this n-to-m component is DROPPED entirely (K=0). Under the
    relaxed ``allow_multimerge=True`` mode, it fuses to one cell (K=1).
    """
    z, y, x = 2, 8, 8
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 1:3, 1:3] = 1
    m488[0, 1:3, 5:7] = 2
    m560[0, 1:3, 1:7] = 9

    out_strict, diag_strict = union_488_560_labels(m488, m560, tau=0.2)
    assert diag_strict["K"] == 0, diag_strict
    assert diag_strict["n_dropped_multimerge"] == 1, diag_strict
    assert _component_ids(out_strict) == set()

    out_relax, diag_relax = union_488_560_labels(
        m488, m560, tau=0.2, allow_multimerge=True
    )
    assert diag_relax["K"] == 1, diag_relax
    assert diag_relax["n_to_m"] == 1
    assert _component_ids(out_relax) == {1}
    _assert_strict_invariants(out_relax, diag_relax, m488, m560, tau=0.2)


def test_two_488_cells_with_separate_560_partners_stay_separate():
    """Two 488 cells each with a clean matching 560 partner produce 2 cells."""
    z, y, x = 1, 8, 8
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0:3, 0:3] = 1
    m560[0, 0:3, 0:3] = 1
    m488[0, 0:3, 3:6] = 2
    m560[0, 0:3, 3:6] = 2

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 2, diag
    assert diag["paired_1_1"] == 2
    assert diag["n_to_m"] == 0
    assert _component_ids(out) == {1, 2}
    assert int(out[0, 0, 0]) != int(out[0, 0, 4])
    _assert_strict_invariants(out, diag, m488, m560, tau=0.2)


def test_sub_tau_overlap_drops_both_labels():
    """A tiny shared corner between two large labels (IoMin << tau) leaves
    each label as its own UF component (no edge). Both fail gate-1 and are
    dropped under strict construction.
    """
    z, y, x = 1, 20, 20
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0:10, 0:20] = 4
    m560[0, 9:19, 0:20] = 6

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 0, diag
    assert diag["edges"] == 0
    assert diag["n_dropped_unpaired"] == 2, diag


def test_gate2_drops_transitive_overmerge():
    """Strict gate-2: a chain a1-b1-(a1)-b2-a2 with a WEAK direct (a2, b1)
    cross-overlap. UF binds all four labels into one component, but the
    direct internal pair (a2, b1) has IoMin < tau, so the whole component
    is dropped (no partial salvage).

    Layout (z=1, y=5, x=30):
      a1 = 488 label 1 :  rows 0:3, cols  0:10  (30 vox)  -- spans BOTH b1 and b2
      b1 = 560 label 1 :  rows 0:5, cols  0:5   (25 vox)
      b2 = 560 label 2 :  rows 0:5, cols  5:10  (25 vox)
      a2 = 488 label 2 :  row 4,    cols  1:30  (29 vox)  -- weakly grazes b1, strong with b2

    IoMin pairs:
      (a1, b1) = 15/25 = 0.60  [edge]
      (a1, b2) = 15/25 = 0.60  [edge]
      (a2, b2) = 5/25  = 0.20  [edge at tau]
      (a2, b1) = 4/25  = 0.16  [NO edge; below tau]
    UF: a1-b1, a1-b2, a2-b2 -> component {a1, a2, b1, b2}.
    Gate-2 internal-pair check finds (a2, b1) with IoMin=0.16 -> drop.
    """
    z, y, x = 1, 5, 30
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0:3, 0:10] = 1
    m488[0, 4, 1:30] = 2
    m560[0, 0:5, 0:5] = 1
    m560[0, 0:5, 5:10] = 2

    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 0, diag
    assert diag["n_dropped_closure"] >= 1, diag
    wid = diag.get("worst_internal_iomin_dropped")
    assert wid is not None and wid < 0.2, diag


def test_strict_invariants_hold_on_random_pairs():
    """Property: after strict merge, every cell has both channels and the
    worst internal IoMin is >= tau."""
    rng = np.random.default_rng(42)
    z, y, x = 4, 16, 16
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    # Place 4 paired cells, each 3x3x3 with full overlap (IoMin=1).
    for k, (zc, yc, xc) in enumerate([(0, 1, 1), (0, 1, 10), (1, 10, 1), (1, 10, 10)], start=1):
        m488[zc:zc + 2, yc:yc + 3, xc:xc + 3] = k
        m560[zc:zc + 2, yc:yc + 3, xc:xc + 3] = k * 11  # different non-contiguous Cellpose IDs
    # Add one stray 488 label (no 560 partner) -- should be dropped by gate-1.
    m488[3, 14:16, 14:16] = 99
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 4, diag
    assert diag["n_dropped_unpaired"] == 1, diag
    _assert_strict_invariants(out, diag, m488, m560, tau=0.2)


def test_min_label_voxels_drops_tiny_labels():
    """Small labels below min_label_voxels are dropped before pair census,
    so they cannot contribute to gates either way.
    """
    z, y, x = 1, 6, 6
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0, 0] = 1                  # 1 voxel - tiny
    m488[0, 1:4, 1:4] = 2               # 9 voxels
    m560[0, 1:4, 1:4] = 5               # 9 voxels (matches label 2)

    out, diag = union_488_560_labels(m488, m560, tau=0.2, min_label_voxels=5)
    assert diag["N488"] == 1   # the 1-voxel label is dropped from 488
    assert diag["N560"] == 1
    assert diag["K"] == 1      # only one paired cell remains
    assert int(out[0, 0, 0]) == 0
    _assert_strict_invariants(out, diag, m488, m560, tau=0.2)


def test_dtype_escalates_to_uint32_when_K_exceeds_uint16_max():
    """Synthetic many-label case to validate uint32 escalation path.

    Every 488 label must be paired with a matching 560 label to survive
    strict construction.
    """
    z, y, x = 1, 1, 70000
    arr = np.arange(1, x + 1, dtype=np.int64).reshape(z, y, x)
    m488 = arr.copy()
    m560 = arr.copy()  # identical -> IoMin=1.0 per pair
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 70000
    assert out.dtype == np.uint32


def test_contiguous_relabel():
    """Output IDs are contiguous 1..K (no gaps), even when intermediate
    components are dropped by the strict gates.
    """
    z, y, x = 2, 6, 6
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 0:2, 0:2] = 17
    m488[1, 4:6, 4:6] = 42  # 488_only -> dropped by gate-1
    m560[0, 0:2, 0:2] = 99   # matches 488 label 17
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    assert diag["K"] == 1, diag
    used = sorted(_component_ids(out))
    assert used == list(range(1, diag["K"] + 1))
    _assert_strict_invariants(out, diag, m488, m560, tau=0.2)


def test_background_stays_zero():
    """A 488 label with no 560 partner is dropped; everything is background."""
    z, y, x = 2, 5, 5
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    m488[0, 1, 1] = 1
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    bg = (m488 == 0) & (m560 == 0)
    assert int(out[bg].max()) == 0
    assert diag["K"] == 0


def test_label_map_records_contributors():
    """label_map sidecar payload correctly maps cell_id -> 488/560 IDs."""
    z, y, x = 1, 6, 12
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    # Two paired cells with different Cellpose IDs in each channel.
    m488[0, 0:3, 0:3] = 7
    m560[0, 0:3, 0:3] = 13
    m488[0, 0:3, 6:9] = 23
    m560[0, 0:3, 6:9] = 31
    out, diag = union_488_560_labels(m488, m560, tau=0.2)
    label_map = diag["label_map"]
    assert set(label_map.keys()) == {1, 2}
    for entry in label_map.values():
        assert entry["n_488"] == 1
        assert entry["n_560"] == 1
        assert entry["total_vox"] == 9


def test_closure_global_drops_weak_external_graze():
    """In --closure global, a 488 label with ANY weak external graze is
    dropped before UF — even if it has a strong primary partner."""
    z, y, x = 1, 6, 30
    m488 = np.zeros((z, y, x), dtype=np.uint16)
    m560 = np.zeros((z, y, x), dtype=np.uint16)
    # Pair A: 488=1, 560=1 -> IoMin=1
    m488[0, 0:3, 0:5] = 1
    m560[0, 0:3, 0:5] = 1
    # 488=1 also weakly grazes 560=2 far away
    m560[0, 0:3, 22:30] = 2   # 24 vox
    m488[0, 0, 25] = 1        # extend 488=1 by 1 voxel to graze 560=2
    # IoMin(1, 2) = 1 / min(16, 24) = 1/16 = 0.0625 -> weak
    # In closure=component, the (1,1) component survives.
    # In closure=global, label 488=1 has a weak external graze and is dropped.
    out_c, diag_c = union_488_560_labels(m488, m560, tau=0.2, closure="component")
    out_g, diag_g = union_488_560_labels(m488, m560, tau=0.2, closure="global")
    # component: 1 cell survives (the strong pair)
    assert diag_c["K"] == 1, diag_c
    # global: 488=1 dropped -> 560=1 has no 488 partner -> all dropped
    assert diag_g["K"] == 0, diag_g
    assert diag_g["n_global_dropped_488"] >= 1
