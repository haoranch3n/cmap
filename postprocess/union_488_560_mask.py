#!/usr/bin/env python3
"""
Strict-overlap merge of 488nm + 560nm Cellpose indexed masks.

Algorithm (strict-by-construction)
---------------------------------

For every pair of labels ``(a, b)`` where ``a`` is a 488 cell and ``b`` is a
560 cell with at least one shared voxel, compute::

    IoMin(a, b) = |a INTERSECT b| / min(|a|, |b|)

A bipartite edge ``(a, b)`` is added whenever ``IoMin >= tau``. The connected
components of that graph are then filtered through **three construction gates**
before being emitted as cells:

1. **Gate-1 (mixed-channel):** a component must contain at least one 488
   label AND at least one 560 label. Single-channel components
   (``488_only`` / ``560_only``) are dropped entirely.
2. **Gate-2 (pairwise IoMin closure):** for every internal cross-pair
   ``(a, b)`` inside a component with ``|a INTERSECT b| > 0``,
   ``IoMin(a, b) >= tau`` must hold. If any internal pair is weak (e.g. a
   transitive UF over-merge introduced a direct overlap with
   ``IoMin < tau``), the entire component is dropped. There is no partial
   salvage.
3. **Gate-3 (strict 1-1, default):** a component must contain exactly ONE
   488 label and exactly ONE 560 label. Over-segmentation cases (one 488
   ID paired with multiple 560 IDs, or vice versa, or true n-to-m fusion)
   drop the entire component. Pass ``--allow-multimerge`` to disable this
   gate and keep the n-to-m fusion behavior (gate-1 + gate-2 only).

Surviving components are assigned a contiguous ID ``1..K_strict`` (sorted by
total voxel count, descending). Every emitted cell therefore satisfies the
strict-overlap invariant: both channels co-segment the cell, every direct
internal cross-overlap is IoMin-strong, and (by default) each cell is a
clean 1-1 pair of one 488 label and one 560 label.

Closure mode (``--closure``):

- ``component`` (default): gate-2 only checks pairs inside the component.
- ``global``: opt-in pre-pass that drops any label with ANY cross-overlap
  ``0 < IoMin < tau`` to any opposite-channel label, then rebuilds the pair
  census on the cleaned masks. Stricter; expected to dissolve more cells.

Tie-breaker on voxels where 488 and 560 disagree on the component (possible
when the pair's IoMin is below ``tau``): 488 wins by default. The disagreeing
voxel count is reported as ``conflict_voxels``.

**Writes** (by default):

- ``<output>/union_488_560_strict_overlap.tif`` — indexed 3D label volume
  (``uint16`` if ``K_strict <= 65535`` else ``uint32``).
- ``<output>/union_488_560_strict_overlap_manifest.json`` — params snapshot
  (tau, min_label_voxels, closure, gate counts, K_strict).
- ``<output>/union_488_560_strict_overlap_label_map.csv`` — per-cell
  contributor table (cell_id, n_488, n_560, ids_488, ids_560, total_vox).
- ``<output>/union_488_560_strict_overlap_combined.tif`` — 4-channel OME
  BigTIFF (Z, C, Y, X): 642 / 488 / 560 originals + strict union mask. Use
  ``--skip-combined`` to skip.

CLI
---

- ``--tau FLOAT`` (default ``0.2``): IoMin threshold for fusion AND gate-2.
- ``--min-label-voxels INT`` (default ``0``): drop labels smaller than this in
  either channel before stitching.
- ``--closure {component,global}`` (default ``component``).
- ``--allow-multimerge`` (flag, default off): disable gate-3 and keep the
  n-to-m fusion behavior.

Notes
-----

- Inputs are the per-channel indexed masks produced by segmentation:
  ``<output>/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif`` and
  ``<output>/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif``.
- Z-mismatch handling matches ``postprocess/filter_642_mask.py``: if Z differs,
  truncate both volumes to the smallest Z; XY mismatch raises.
- This script does not touch ``filtered_642*`` artifacts.
- The legacy non-strict artifact ``union_488_560.tif`` is no longer written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

UNION_CHANNELS = ["488nm_crop", "560nm_crop"]
ALL_CHANNELS = ["642nm_crop", "488nm_crop", "560nm_crop"]


def _find_original_volume(data_dir: Path, stem: str) -> Path:
    prefix = stem.replace("_crop", "")
    reg_matches = sorted(data_dir.glob(f"reg_{prefix}*.tif"))
    if len(reg_matches) == 1:
        return reg_matches[0]
    if len(reg_matches) > 1:
        warped = [m for m in reg_matches if "Warped" in m.name]
        if len(warped) == 1:
            return warped[0]
        raise ValueError(f"Multiple reg_{prefix}*.tif in {data_dir}: {reg_matches}")
    direct = data_dir / f"{stem}.tif"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"No original for {stem} in {data_dir}")


def _resolve_dirs(args) -> tuple[Path | None, Path]:
    if args.data_rel:
        data_dir = Path(DATA_DIR) / args.data_rel
        output_dir = (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
    elif args.output_dir:
        output_dir = args.output_dir.resolve()
        data_dir = args.data_dir.resolve() if args.data_dir else None
    else:
        raise SystemExit(
            "Provide either --data-rel or --output-dir (and --data-dir for combined.tif)"
        )
    return data_dir, output_dir


def _load_3d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D or 4-D, got shape {arr.shape} from {path}")
    return arr


def _find_indexed_mask(output_dir: Path, ch: str) -> Path:
    seg_dir = output_dir / ch / "segmentation_3D_masks"
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {seg_dir}")
    direct = seg_dir / f"{ch}_3D_indexed.tif"
    if direct.is_file():
        return direct
    matches = sorted(p for p in seg_dir.rglob("*_3D_indexed.tif") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"No *_3D_indexed.tif under {seg_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple indexed masks under {seg_dir}; expected one: {matches}")
    return matches[0]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n + 1, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def _pairwise_intersections(
    m488: np.ndarray, m560: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pair_a, pair_b, counts) for every (488_id, 560_id) sharing >=1 voxel."""
    both = (m488 > 0) & (m560 > 0)
    if not both.any():
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    a_arr = m488[both].astype(np.int64, copy=False)
    b_arr = m560[both].astype(np.int64, copy=False)
    n560_eff = int(b_arr.max()) + 1
    key = a_arr * n560_eff + b_arr
    u_keys, u_counts = np.unique(key, return_counts=True)
    pair_a = (u_keys // n560_eff).astype(np.int64)
    pair_b = (u_keys % n560_eff).astype(np.int64)
    return pair_a, pair_b, u_counts.astype(np.int64)


def _split_n_to_m_components(
    out: np.ndarray,
    m488_use: np.ndarray,
    m560_use: np.ndarray,
    root_to_488_ids: dict,
    root_to_560_ids: dict,
    root_to_id: dict,
    *,
    policy: str = "all",
    tie_prefer: str = "488",
) -> tuple[np.ndarray, dict]:
    """Redistribute voxels of components where 488 and 560 disagree on cell count.

    For each component root R with c488 != c560 (or c488 == c560 > 1), pick a
    splitter channel and reassign every voxel of the merged cell to one of the
    splitter's sub-IDs. Voxels where the splitter channel is 0 inside the cell
    are assigned to the nearest splitter seed via Euclidean distance transform.

    Policy
    ------
    - ``"off"``  : no-op (caller should not invoke; here for symmetry).
    - ``"clean"``: only split components with c488==1 xor c560==1 (unambiguous
      over-merge in one channel).
    - ``"all"``  : also split true n-to-m where both sides have >=2 components,
      using ``tie_prefer`` when c488 == c560 > 1.
    """
    from scipy.ndimage import distance_transform_edt

    new_out = out.astype(np.int64, copy=True)
    next_id = int(new_out.max()) + 1

    diag = {
        "split_components": 0,
        "new_subcells": 0,
        "splits_by_488": 0,
        "splits_by_560": 0,
        "skipped_no_seeds": 0,
        "events": [],
    }

    for r in sorted(root_to_id.keys()):
        cell_id = root_to_id[r]
        a_ids = root_to_488_ids.get(r, [])
        b_ids = root_to_560_ids.get(r, [])
        c488, c560 = len(a_ids), len(b_ids)

        if c488 <= 1 and c560 <= 1:
            continue
        if c488 == c560 and policy == "clean":
            continue
        if policy == "clean" and c488 > 1 and c560 > 1:
            continue

        if c488 > c560:
            splitter_mask = m488_use
            sub_ids = a_ids
            from_channel = "488"
        elif c560 > c488:
            splitter_mask = m560_use
            sub_ids = b_ids
            from_channel = "560"
        else:
            if tie_prefer == "488":
                splitter_mask = m488_use
                sub_ids = a_ids
                from_channel = "488"
            else:
                splitter_mask = m560_use
                sub_ids = b_ids
                from_channel = "560"

        cell_mask = new_out == cell_id
        if not cell_mask.any():
            continue

        zz, yy, xx = np.where(cell_mask)
        z0, z1 = int(zz.min()), int(zz.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        x0, x1 = int(xx.min()), int(xx.max()) + 1

        cell_box = cell_mask[z0:z1, y0:y1, x0:x1]
        splitter_box = splitter_mask[z0:z1, y0:y1, x0:x1].astype(np.int64, copy=False)

        sub_id_arr = np.array(sub_ids, dtype=np.int64)
        seed_box = np.isin(splitter_box, sub_id_arr) & cell_box

        if not seed_box.any():
            diag["skipped_no_seeds"] += 1
            continue

        bg = ~seed_box
        if bg.any():
            indices = distance_transform_edt(bg, return_distances=False, return_indices=True)
            nearest_box = splitter_box[tuple(indices)]
        else:
            nearest_box = splitter_box

        sub_to_new_id: dict[int, int] = {}
        for sub in sub_ids:
            sub_to_new_id[int(sub)] = next_id
            next_id += 1

        new_out_view = new_out[z0:z1, y0:y1, x0:x1]
        new_out_view[cell_box] = 0
        for sub, new_id in sub_to_new_id.items():
            assign = (nearest_box == sub) & cell_box
            new_out_view[assign] = new_id

        diag["split_components"] += 1
        diag["new_subcells"] += len(sub_ids)
        if from_channel == "488":
            diag["splits_by_488"] += 1
        else:
            diag["splits_by_560"] += 1
        diag["events"].append(
            {
                "old_cell_id": int(cell_id),
                "new_cell_ids": [int(v) for v in sub_to_new_id.values()],
                "from": from_channel,
                "c488": c488,
                "c560": c560,
            }
        )

    return new_out, diag


def _drop_disconnected_per_label(
    out: np.ndarray,
    *,
    connectivity: int = 1,
) -> tuple[np.ndarray, dict]:
    """Keep only the largest 3D connected component for each label.

    ``connectivity`` is passed to ``scipy.ndimage.generate_binary_structure``
    (1=face/6-conn, 2=face+edge/18-conn, 3=face+edge+corner/26-conn).
    """
    from scipy.ndimage import generate_binary_structure
    from scipy.ndimage import label as cc_label

    structure = generate_binary_structure(3, connectivity)
    new_out = out.copy()
    diag = {"dropped_components": 0, "dropped_voxels": 0, "labels_affected": 0}

    used = np.unique(out)
    used = used[used > 0]

    for L in used.tolist():
        mask = new_out == L
        if not mask.any():
            continue
        zz, yy, xx = np.where(mask)
        z0, z1 = int(zz.min()), int(zz.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        sub = mask[z0:z1, y0:y1, x0:x1]
        cc, n = cc_label(sub, structure=structure)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        to_drop = (cc != 0) & (cc != largest)
        if not to_drop.any():
            continue
        new_out_view = new_out[z0:z1, y0:y1, x0:x1]
        new_out_view[to_drop] = 0
        diag["dropped_components"] += int(n - 1)
        diag["dropped_voxels"] += int(to_drop.sum())
        diag["labels_affected"] += 1

    return new_out, diag


def _smooth_per_label(
    out: np.ndarray,
    *,
    radius: int,
) -> tuple[np.ndarray, dict]:
    """Per-label 3D morphological closing with a cubic structuring element of given radius.

    Only paints voxels that are currently background (0); other labels are not
    overwritten. Best-effort smoothing of OR-union ragged boundaries.
    """
    from scipy.ndimage import binary_closing

    diag = {"smoothed_labels": 0, "added_voxels": 0}
    if radius <= 0:
        return out, diag

    structure = np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
    new_out = out.copy()

    used = np.unique(out)
    used = used[used > 0]
    for L in used.tolist():
        mask = new_out == L
        if not mask.any():
            continue
        zz, yy, xx = np.where(mask)
        z0 = max(int(zz.min()) - radius, 0)
        z1 = min(int(zz.max()) + radius + 1, out.shape[0])
        y0 = max(int(yy.min()) - radius, 0)
        y1 = min(int(yy.max()) + radius + 1, out.shape[1])
        x0 = max(int(xx.min()) - radius, 0)
        x1 = min(int(xx.max()) + radius + 1, out.shape[2])
        sub = mask[z0:z1, y0:y1, x0:x1]
        closed = binary_closing(sub, structure=structure)
        new_out_view = new_out[z0:z1, y0:y1, x0:x1]
        paint = closed & (new_out_view == 0)
        if not paint.any():
            continue
        new_out_view[paint] = L
        diag["smoothed_labels"] += 1
        diag["added_voxels"] += int(paint.sum())

    return new_out, diag


def _compute_label_map(
    out: np.ndarray,
    lut_488: np.ndarray,
    lut_560: np.ndarray,
) -> dict[int, dict]:
    """UF-based label-map: invert the per-channel LUTs.

    Each surviving Cellpose label maps to exactly one final cell ID via the
    LUT, so each cell's ``ids_488`` / ``ids_560`` is the unambiguous UF
    membership (no voxel-level spillover from conflict resolution).
    ``total_vox`` is the voxel count of the final cell in ``out``.

    Note: under ``--split-on-disagree != "off"`` the LUTs do not see the
    new sub-cell IDs; those rows will be missing from the label-map. That
    is acceptable for the default (split off) configuration; a separate
    fallback would be needed before enabling split-on-disagree by default.
    """
    label_map: dict[int, dict] = {}
    used = np.unique(out)
    used = used[used > 0]
    if used.size == 0:
        return label_map

    total_counts = np.bincount(out.astype(np.int64, copy=False).ravel())

    cells_488: dict[int, list[int]] = defaultdict(list)
    cells_560: dict[int, list[int]] = defaultdict(list)
    for label_id, cell_id in enumerate(lut_488.tolist()):
        if label_id == 0 or cell_id == 0:
            continue
        cells_488[int(cell_id)].append(int(label_id))
    for label_id, cell_id in enumerate(lut_560.tolist()):
        if label_id == 0 or cell_id == 0:
            continue
        cells_560[int(cell_id)].append(int(label_id))

    for cid in used.tolist():
        cid_int = int(cid)
        ids_488 = sorted(cells_488.get(cid_int, []))
        ids_560 = sorted(cells_560.get(cid_int, []))
        label_map[cid_int] = {
            "n_488": len(ids_488),
            "n_560": len(ids_560),
            "ids_488": ids_488,
            "ids_560": ids_560,
            "total_vox": int(total_counts[cid_int]),
        }
    return label_map


def union_488_560_labels(
    m488: np.ndarray,
    m560: np.ndarray,
    *,
    tau: float = 0.2,
    min_label_voxels: int = 0,
    closure: str = "component",
    allow_multimerge: bool = False,
    split_on_disagree: str = "off",
    tie_prefer: str = "488",
    drop_disconnected: bool = False,
    connectivity: int = 1,
    smooth_radius: int = 0,
    conflict_rule: str = "488_wins",
    conflict_merge_threshold: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Strict-overlap label-stitching merge of two indexed Cellpose masks.

    The output mask is strict-by-construction: every emitted ``cell_id``
    satisfies three invariants

      1. **Mixed-channel** -- its connected component contains at least one
         488 label and at least one 560 label. Single-channel components
         (``488_only`` / ``560_only``) are dropped.
      2. **Pairwise IoMin closure** -- for every cross-pair ``(a_488, b_560)``
         of labels inside the component with ``|a INTERSECT b| > 0`` in the
         raw masks, ``IoMin(a, b) >= tau``. Components that fail this check
         are dropped entirely (no partial salvage).
      3. **Strict 1-1** (default) -- the component contains exactly ONE 488
         label and exactly ONE 560 label. Any over-segmentation case (one
         488 ID paired with multiple 560 IDs, or vice versa, or true n-to-m
         fusion) drops the entire component. Pass ``allow_multimerge=True``
         to relax this gate and keep the prior n-to-m fusion behavior.

    Gates run between UF and the final LUT build, BEFORE the optional
    split/drop_disconnected/smooth passes, so post-processing only operates
    on survivors.

    Parameters
    ----------
    m488, m560 : np.ndarray
        ZYX integer label volumes; same shape; 0 = background.
    tau : float
        IoMin threshold for fusing a 488 label with a 560 label and for the
        gate-2 pairwise closure check.
    min_label_voxels : int
        Drop labels smaller than this from either channel before stitching;
        their voxels become 0 in the output.
    closure : {'component', 'global'}
        Strict-overlap closure mode. ``'component'`` (default) only checks
        pairwise IoMin inside each connected component. ``'global'`` is an
        opt-in stricter rule: drop any label that has ANY cross-overlap with
        a label in the other channel where ``0 < IoMin < tau`` -- i.e. a
        single weak external graze invalidates the label entirely. Recompute
        the pair census on the cleaned masks before UF.
    allow_multimerge : bool
        If ``False`` (default), enforce strict 1-1: every surviving cell
        has exactly one 488 contributor and exactly one 560 contributor.
        Components with n-to-m fusion are dropped entirely. If ``True``,
        allow over-segmentation cases (one 488 paired with several 560 IDs,
        or vice versa, or n-to-m components where every internal cross-pair
        is IoMin-strong) to fuse into a single cell -- the gate-2 behavior
        without gate-3.
    split_on_disagree : {'off', 'clean', 'all'}
        Optional Mode-3 fix for components where 488 and 560 disagree on the
        number of cells. ``'off'`` keeps the current fused behavior. ``'clean'``
        only splits unambiguous 1-vs-N or N-vs-1 components. ``'all'`` also
        splits 2-vs-3, 3-vs-2 etc. (using ``tie_prefer`` when c488 == c560 > 1).
    tie_prefer : {'488', '560'}
        Channel preference when both channels have an equal number of >=2
        disagreeing components.
    drop_disconnected : bool
        Optional Mode-2 fix: after splitting, keep only the largest 3D
        connected component per label (drops conflict-voxel slivers).
    connectivity : int
        1=face (6-conn), 2=face+edge (18-conn), 3=face+edge+corner (26-conn).
    smooth_radius : int
        Optional Mode-1 fix: per-label 3D morphological closing radius (cubic
        SE). 0 disables. Only fills background voxels; never overwrites other
        labels.
    conflict_rule : {'488_wins', '560_wins', 'larger_wins', 'smaller_wins', 'merged_loses'}
        Rule for assigning voxels where ``out_488 != out_560`` and both > 0
        (i.e. 488 and 560 disagree on which union cell a voxel belongs to).
        ``'488_wins'`` (default, original behavior) always picks 488. ``'560_wins'``
        flips to 560. ``'larger_wins'`` / ``'smaller_wins'`` pick by global
        label volume. ``'merged_loses'`` picks the side whose label is *not*
        flagged as a "merger" — i.e. only overlaps one significant label in
        the other channel; if both or neither are merged, falls back to 488.
    conflict_merge_threshold : float
        IoMin threshold for flagging a label as a "merger" in ``'merged_loses'``
        rule. A 488 ID A is a merger if it has IoMin >= threshold with two or
        more 560 IDs (and similarly for 560).

    Returns
    -------
    (out, diagnostics) where ``out`` is an indexed label volume with contiguous
    IDs ``1..K`` and ``diagnostics`` is a dict with counts useful for logging.
    """
    if m488.shape != m560.shape:
        raise ValueError(f"shape mismatch: 488 {m488.shape} vs 560 {m560.shape}")
    if closure not in ("component", "global"):
        raise ValueError(
            f"closure must be 'component' or 'global', got {closure!r}"
        )

    m488 = m488.astype(np.int64, copy=False)
    m560 = m560.astype(np.int64, copy=False)

    n488_max = int(m488.max()) if m488.size else 0
    n560_max = int(m560.max()) if m560.size else 0

    vol_488 = np.bincount(m488.ravel(), minlength=n488_max + 1)
    vol_560 = np.bincount(m560.ravel(), minlength=n560_max + 1)

    valid_488 = vol_488 >= max(1, min_label_voxels)
    valid_488[0] = False
    valid_560 = vol_560 >= max(1, min_label_voxels)
    valid_560[0] = False

    if min_label_voxels > 0:
        bad_488 = ~valid_488[m488]
        bad_560 = ~valid_560[m560]
        m488_use = np.where(bad_488, 0, m488)
        m560_use = np.where(bad_560, 0, m560)
    else:
        m488_use = m488
        m560_use = m560

    pair_a, pair_b, pair_counts = _pairwise_intersections(m488_use, m560_use)

    if pair_a.size:
        denom = np.minimum(vol_488[pair_a], vol_560[pair_b]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            iomin = np.where(denom > 0, pair_counts / denom, 0.0)
    else:
        iomin = np.empty(0, dtype=np.float64)

    n_global_dropped_488 = 0
    n_global_dropped_560 = 0
    if closure == "global" and pair_a.size:
        weak_mask = (iomin > 0) & (iomin < tau)
        if weak_mask.any():
            bad_488 = np.unique(pair_a[weak_mask])
            bad_560 = np.unique(pair_b[weak_mask])
            n_global_dropped_488 = int(bad_488.size)
            n_global_dropped_560 = int(bad_560.size)
            if bad_488.size:
                m488_use = np.where(np.isin(m488_use, bad_488), 0, m488_use)
                valid_488 = valid_488.copy()
                valid_488[bad_488] = False
            if bad_560.size:
                m560_use = np.where(np.isin(m560_use, bad_560), 0, m560_use)
                valid_560 = valid_560.copy()
                valid_560[bad_560] = False
            pair_a, pair_b, pair_counts = _pairwise_intersections(m488_use, m560_use)
            if pair_a.size:
                denom = np.minimum(vol_488[pair_a], vol_560[pair_b]).astype(np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    iomin = np.where(denom > 0, pair_counts / denom, 0.0)
            else:
                iomin = np.empty(0, dtype=np.float64)

    edge_mask = iomin >= tau
    edges_a = pair_a[edge_mask]
    edges_b = pair_b[edge_mask]
    n_edges = int(edges_a.size)

    uf = _UnionFind(n488_max + n560_max)
    offset = n488_max
    for a, b in zip(edges_a.tolist(), edges_b.tolist()):
        uf.union(a, offset + b)

    valid_488_ids = np.where(valid_488)[0].astype(np.int64)
    valid_560_ids = np.where(valid_560)[0].astype(np.int64)

    roots_488 = np.array([uf.find(int(a)) for a in valid_488_ids], dtype=np.int64) \
        if valid_488_ids.size else np.empty(0, dtype=np.int64)
    roots_560 = np.array([uf.find(offset + int(b)) for b in valid_560_ids], dtype=np.int64) \
        if valid_560_ids.size else np.empty(0, dtype=np.int64)

    root_total: dict[int, int] = defaultdict(int)
    root_488_count: dict[int, int] = defaultdict(int)
    root_560_count: dict[int, int] = defaultdict(int)
    for a, r in zip(valid_488_ids.tolist(), roots_488.tolist()):
        root_total[int(r)] += int(vol_488[a])
        root_488_count[int(r)] += 1
    for b, r in zip(valid_560_ids.tolist(), roots_560.tolist()):
        root_total[int(r)] += int(vol_560[b])
        root_560_count[int(r)] += 1

    n_components_pre_gate = len(root_total)
    mixed_roots = {
        r
        for r in root_total.keys()
        if root_488_count[r] > 0 and root_560_count[r] > 0
    }
    n_dropped_unpaired = n_components_pre_gate - len(mixed_roots)

    worst_iomin_per_root: dict[int, float] = {}
    if pair_a.size:
        pra = np.array([uf.find(int(a)) for a in pair_a.tolist()], dtype=np.int64)
        prb = np.array(
            [uf.find(offset + int(b)) for b in pair_b.tolist()], dtype=np.int64
        )
        internal = pra == prb
        if internal.any():
            in_roots = pra[internal]
            in_iomin = iomin[internal]
            order = np.argsort(in_roots, kind="stable")
            sr = in_roots[order]
            si = in_iomin[order]
            u_roots, starts = np.unique(sr, return_index=True)
            worst = np.minimum.reduceat(si, starts)
            worst_iomin_per_root = dict(zip(u_roots.tolist(), worst.tolist()))

    failed_closure = {
        r
        for r, w in worst_iomin_per_root.items()
        if w < tau and r in mixed_roots
    }
    kept_after_closure = mixed_roots - failed_closure
    n_dropped_closure = len(failed_closure)

    kept_worsts = [
        worst_iomin_per_root[r]
        for r in kept_after_closure
        if r in worst_iomin_per_root
    ]
    dropped_worsts = [worst_iomin_per_root[r] for r in failed_closure]
    worst_internal_iomin_kept = (
        float(min(kept_worsts)) if kept_worsts else float("nan")
    )
    worst_internal_iomin_dropped = (
        float(min(dropped_worsts)) if dropped_worsts else float("nan")
    )

    if allow_multimerge:
        kept_roots = kept_after_closure
        n_dropped_multimerge = 0
    else:
        strict_1_1_roots = {
            r
            for r in kept_after_closure
            if root_488_count[r] == 1 and root_560_count[r] == 1
        }
        n_dropped_multimerge = len(kept_after_closure) - len(strict_1_1_roots)
        kept_roots = strict_1_1_roots

    sorted_roots = sorted(kept_roots, key=lambda r: (-root_total[r], r))
    K = len(sorted_roots)
    root_to_id = {r: i + 1 for i, r in enumerate(sorted_roots)}

    lut_488 = np.zeros(n488_max + 1, dtype=np.int64)
    for a, r in zip(valid_488_ids.tolist(), roots_488.tolist()):
        lut_488[a] = root_to_id.get(int(r), 0)
    lut_560 = np.zeros(n560_max + 1, dtype=np.int64)
    for b, r in zip(valid_560_ids.tolist(), roots_560.tolist()):
        lut_560[b] = root_to_id.get(int(r), 0)

    out_488 = lut_488[m488_use]
    out_560 = lut_560[m560_use]

    overlap = (out_488 != 0) & (out_560 != 0) & (out_488 != out_560)
    conflict_voxels = int(overlap.sum())

    conflict_diag = {
        "rule": conflict_rule,
        "total": conflict_voxels,
        "merge_threshold": float(conflict_merge_threshold),
        "merged_488_ids": 0,
        "merged_560_ids": 0,
        "wins_488": 0,
        "wins_560": 0,
        "wins_default": 0,
    }

    out = np.where(out_488 != 0, out_488, out_560)

    if conflict_rule == "488_wins" or not overlap.any():
        if overlap.any():
            conflict_diag["wins_488"] = conflict_voxels
    elif conflict_rule == "560_wins":
        out = np.where(overlap, out_560, out)
        conflict_diag["wins_560"] = conflict_voxels
    elif conflict_rule in ("larger_wins", "smaller_wins"):
        v488_arr = vol_488[m488_use]
        v560_arr = vol_560[m560_use]
        if conflict_rule == "larger_wins":
            prefer_560 = overlap & (v560_arr > v488_arr)
            prefer_488 = overlap & (v488_arr >= v560_arr)
        else:
            prefer_560 = overlap & (v560_arr < v488_arr)
            prefer_488 = overlap & (v488_arr <= v560_arr)
        out = np.where(prefer_560, out_560, out)
        conflict_diag["wins_560"] = int(prefer_560.sum())
        conflict_diag["wins_488"] = int(prefer_488.sum())
    elif conflict_rule == "merged_loses":
        merged_488: dict[int, int] = defaultdict(int)
        merged_560: dict[int, int] = defaultdict(int)
        if pair_a.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                iomin_all = np.where(
                    (vol_488[pair_a] > 0) & (vol_560[pair_b] > 0),
                    pair_counts
                    / np.minimum(vol_488[pair_a], vol_560[pair_b]).astype(np.float64),
                    0.0,
                )
            sig_mask = iomin_all >= conflict_merge_threshold
            for a, b in zip(pair_a[sig_mask].tolist(), pair_b[sig_mask].tolist()):
                merged_488[int(a)] += 1
                merged_560[int(b)] += 1
        merged_488_set = {a for a, c in merged_488.items() if c >= 2}
        merged_560_set = {b for b, c in merged_560.items() if c >= 2}
        conflict_diag["merged_488_ids"] = len(merged_488_set)
        conflict_diag["merged_560_ids"] = len(merged_560_set)

        if merged_488_set or merged_560_set:
            a_arr = m488_use
            b_arr = m560_use
            is_488_merged = np.isin(a_arr, np.fromiter(merged_488_set, dtype=np.int64)) if merged_488_set else np.zeros_like(a_arr, dtype=bool)
            is_560_merged = np.isin(b_arr, np.fromiter(merged_560_set, dtype=np.int64)) if merged_560_set else np.zeros_like(b_arr, dtype=bool)
            prefer_560 = overlap & is_488_merged & ~is_560_merged
            prefer_488 = overlap & is_560_merged & ~is_488_merged
            default_488 = overlap & ~prefer_560 & ~prefer_488
            out = np.where(prefer_560, out_560, out)
            conflict_diag["wins_560"] = int(prefer_560.sum())
            conflict_diag["wins_488"] = int(prefer_488.sum() + default_488.sum())
            conflict_diag["wins_default"] = int(default_488.sum())
        else:
            conflict_diag["wins_488"] = conflict_voxels
    else:
        raise ValueError(f"unknown conflict_rule: {conflict_rule}")

    breakdown_488_only = 0
    breakdown_560_only = 0
    breakdown_paired_1_1 = 0
    breakdown_n_to_m = 0
    for r in sorted_roots:
        c488 = root_488_count[r]
        c560 = root_560_count[r]
        if c488 == 0 and c560 > 0:
            breakdown_560_only += 1
        elif c560 == 0 and c488 > 0:
            breakdown_488_only += 1
        elif c488 == 1 and c560 == 1:
            breakdown_paired_1_1 += 1
        else:
            breakdown_n_to_m += 1

    split_diag = {
        "split_components": 0,
        "new_subcells": 0,
        "splits_by_488": 0,
        "splits_by_560": 0,
        "skipped_no_seeds": 0,
        "events": [],
    }
    if split_on_disagree != "off":
        root_to_488_ids: dict[int, list[int]] = defaultdict(list)
        root_to_560_ids: dict[int, list[int]] = defaultdict(list)
        for a, r in zip(valid_488_ids.tolist(), roots_488.tolist()):
            root_to_488_ids[int(r)].append(int(a))
        for b, r in zip(valid_560_ids.tolist(), roots_560.tolist()):
            root_to_560_ids[int(r)].append(int(b))
        out, split_diag = _split_n_to_m_components(
            out,
            m488_use,
            m560_use,
            root_to_488_ids,
            root_to_560_ids,
            root_to_id,
            policy=split_on_disagree,
            tie_prefer=tie_prefer,
        )

    drop_diag = {"dropped_components": 0, "dropped_voxels": 0, "labels_affected": 0}
    if drop_disconnected:
        out, drop_diag = _drop_disconnected_per_label(out, connectivity=connectivity)

    smooth_diag = {"smoothed_labels": 0, "added_voxels": 0}
    if smooth_radius > 0:
        out, smooth_diag = _smooth_per_label(out, radius=smooth_radius)

    used = np.unique(out)
    used = used[used > 0]
    if used.size:
        if not np.array_equal(used, np.arange(1, used.size + 1, dtype=used.dtype)):
            remap = np.zeros(int(used.max()) + 1, dtype=np.int64)
            remap[used] = np.arange(1, used.size + 1, dtype=np.int64)
            out = remap[out]
        K_final = int(used.size)
    else:
        K_final = 0

    if K_final <= np.iinfo(np.uint16).max:
        out = out.astype(np.uint16, copy=False)
    else:
        out = out.astype(np.uint32, copy=False)

    label_map = _compute_label_map(out, lut_488, lut_560)

    diagnostics = {
        "N488": int(valid_488_ids.size),
        "N560": int(valid_560_ids.size),
        "tau": float(tau),
        "min_label_voxels": int(min_label_voxels),
        "closure": str(closure),
        "merge_policy": "multimerge" if allow_multimerge else "strict_1_1",
        "allow_multimerge": bool(allow_multimerge),
        "n_global_dropped_488": int(n_global_dropped_488),
        "n_global_dropped_560": int(n_global_dropped_560),
        "edges": n_edges,
        "n_components_pre_gate": int(n_components_pre_gate),
        "n_dropped_unpaired": int(n_dropped_unpaired),
        "n_dropped_closure": int(n_dropped_closure),
        "n_dropped_multimerge": int(n_dropped_multimerge),
        "worst_internal_iomin_kept": worst_internal_iomin_kept,
        "worst_internal_iomin_dropped": worst_internal_iomin_dropped,
        "label_map": label_map,
        "K": int(K_final),
        "488_only": breakdown_488_only,
        "560_only": breakdown_560_only,
        "paired_1_1": breakdown_paired_1_1,
        "n_to_m": breakdown_n_to_m,
        "conflict_voxels": conflict_voxels,
        "conflict": conflict_diag,
        "split": split_diag,
        "drop": drop_diag,
        "smooth": smooth_diag,
        "split_on_disagree": split_on_disagree,
        "tie_prefer": tie_prefer,
        "drop_disconnected": bool(drop_disconnected),
        "smooth_radius": int(smooth_radius),
        "conflict_rule": conflict_rule,
        "conflict_merge_threshold": float(conflict_merge_threshold),
    }
    return out, diagnostics


def _print_diagnostics(diag: dict, out: np.ndarray) -> None:
    print(
        f"  N488={diag['N488']}, N560={diag['N560']}, "
        f"tau={diag['tau']:.3f}, min_label_voxels={diag['min_label_voxels']}, "
        f"closure={diag.get('closure', 'component')}, "
        f"merge_policy={diag.get('merge_policy', 'strict_1_1')}"
    )
    if diag.get("closure") == "global":
        print(
            f"  global-closure pre-pass dropped: "
            f"488={diag.get('n_global_dropped_488', 0)} "
            f"560={diag.get('n_global_dropped_560', 0)}"
        )
    print(f"  edges (IoMin >= {diag['tau']:.2f}): {diag['edges']}")
    print(
        f"  pre-gate components: {diag.get('n_components_pre_gate', 0)}  "
        f"-> gate-1 dropped (unpaired): {diag.get('n_dropped_unpaired', 0)}  "
        f"-> gate-2 dropped (closure): {diag.get('n_dropped_closure', 0)}  "
        f"-> gate-3 dropped (n-to-m): {diag.get('n_dropped_multimerge', 0)}"
    )
    wik = diag.get("worst_internal_iomin_kept", float("nan"))
    wid = diag.get("worst_internal_iomin_dropped", float("nan"))
    print(
        f"  worst internal IoMin: kept={wik:.4f}  dropped={wid:.4f}  "
        f"(strict-overlap guarantee: kept >= tau={diag['tau']:.3f})"
    )
    print(
        f"  components K_strict = {diag['K']} "
        f"(488_only={diag['488_only']}, 560_only={diag['560_only']}, "
        f"paired_1_1={diag['paired_1_1']}, n_to_m={diag['n_to_m']})"
    )
    cd = diag.get("conflict", {})
    rule = diag.get("conflict_rule", cd.get("rule", "488_wins"))
    print(
        f"  conflict_voxels: {diag['conflict_voxels']}  rule={rule}  "
        f"merged_488_ids={cd.get('merged_488_ids', 0)} merged_560_ids={cd.get('merged_560_ids', 0)} "
        f"wins_488={cd.get('wins_488', 0)} wins_560={cd.get('wins_560', 0)} "
        f"wins_default={cd.get('wins_default', 0)}"
    )
    sd = diag.get("split", {})
    if diag.get("split_on_disagree", "off") != "off":
        print(
            f"  split: policy={diag['split_on_disagree']} tie_prefer={diag.get('tie_prefer', '488')} "
            f"split_components={sd.get('split_components', 0)} "
            f"new_subcells={sd.get('new_subcells', 0)} "
            f"by_488={sd.get('splits_by_488', 0)} by_560={sd.get('splits_by_560', 0)} "
            f"skipped_no_seeds={sd.get('skipped_no_seeds', 0)}"
        )
        events = sd.get("events", [])
        if events:
            for ev in events[:10]:
                print(
                    f"    split: old={ev['old_cell_id']} -> new={ev['new_cell_ids']} "
                    f"(from={ev['from']}, c488={ev['c488']}, c560={ev['c560']})"
                )
            if len(events) > 10:
                print(f"    ... ({len(events) - 10} more split events)")
    dd = diag.get("drop", {})
    if diag.get("drop_disconnected", False):
        print(
            f"  drop: dropped_components={dd.get('dropped_components', 0)} "
            f"dropped_voxels={dd.get('dropped_voxels', 0)} "
            f"labels_affected={dd.get('labels_affected', 0)}"
        )
    md = diag.get("smooth", {})
    if diag.get("smooth_radius", 0) > 0:
        print(
            f"  smooth: radius={diag['smooth_radius']} "
            f"smoothed_labels={md.get('smoothed_labels', 0)} "
            f"added_voxels={md.get('added_voxels', 0)}"
        )
    if diag["K"] > 0:
        labels, counts = np.unique(out, return_counts=True)
        nz = labels > 0
        sizes = sorted(zip(labels[nz].tolist(), counts[nz].tolist()), key=lambda t: -t[1])
        top = sizes[:5]
        bottom = sizes[-5:] if len(sizes) > 5 else []
        median = sizes[len(sizes) // 2][1] if sizes else 0
        print(f"  top 5 sizes: {top}")
        if bottom:
            print(f"  bottom 5 sizes: {bottom}")
        print(f"  median size: {median}")


def _crop_zyx_to_ref(arr: np.ndarray, ref: tuple[int, int, int], label: str) -> np.ndarray:
    rz, ry, rx = ref
    if arr.shape[1:] != (ry, rx):
        raise ValueError(f"{label} spatial shape {arr.shape[1:]} != ref {(ry, rx)}")
    if arr.shape[0] < rz:
        raise ValueError(f"{label} Z={arr.shape[0]} < ref Z={rz}")
    return arr[:rz]


def _write_union_volume(
    output_dir: Path,
    union_mask: np.ndarray,
    force: bool,
    *,
    output_name: str = "union_488_560_strict_overlap.tif",
) -> Path:
    out_path = output_dir / output_name
    if out_path.exists() and not force:
        print(f"SKIP (exists): {out_path}")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp"
    tifffile.imwrite(
        tmp_path,
        union_mask,
        photometric="minisblack",
        compression="zlib",
        metadata={"axes": "ZYX"},
    )
    os.replace(tmp_path, str(out_path))
    print(f"WROTE: {out_path}  shape={union_mask.shape}  dtype={union_mask.dtype}")
    return out_path


def _sanitize_for_json(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _write_manifest(
    output_dir: Path,
    manifest_name: str,
    diag: dict,
    union_path: Path,
    force: bool,
) -> Path:
    """Write a small JSON manifest beside the strict TIFF.

    Includes the merge params, gate diagnostics, and the artifact name so
    downstream tools (lint, QC join, audit) can reproduce the strict identity.
    The ``label_map`` payload lives in a separate CSV; manifest stays small.
    """
    manifest_path = output_dir / manifest_name
    if manifest_path.exists() and not force:
        print(f"SKIP (exists): {manifest_path}")
        return manifest_path

    payload = {
        "artifact": str(union_path.name),
        "tau": diag["tau"],
        "min_label_voxels": diag["min_label_voxels"],
        "closure": diag.get("closure", "component"),
        "merge_policy": diag.get("merge_policy", "strict_1_1"),
        "conflict_rule": diag.get("conflict_rule"),
        "conflict_merge_threshold": diag.get("conflict_merge_threshold"),
        "split_on_disagree": diag.get("split_on_disagree"),
        "tie_prefer": diag.get("tie_prefer"),
        "drop_disconnected": diag.get("drop_disconnected"),
        "connectivity": diag.get("connectivity"),
        "smooth_radius": diag.get("smooth_radius"),
        "N488": diag["N488"],
        "N560": diag["N560"],
        "edges": diag["edges"],
        "n_global_dropped_488": diag.get("n_global_dropped_488", 0),
        "n_global_dropped_560": diag.get("n_global_dropped_560", 0),
        "n_components_pre_gate": diag.get("n_components_pre_gate", 0),
        "n_dropped_unpaired": diag.get("n_dropped_unpaired", 0),
        "n_dropped_closure": diag.get("n_dropped_closure", 0),
        "n_dropped_multimerge": diag.get("n_dropped_multimerge", 0),
        "worst_internal_iomin_kept": diag.get("worst_internal_iomin_kept"),
        "worst_internal_iomin_dropped": diag.get("worst_internal_iomin_dropped"),
        "K_strict": diag["K"],
        "conflict_voxels": diag["conflict_voxels"],
    }
    tmp_path = str(manifest_path) + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(_sanitize_for_json(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, str(manifest_path))
    print(f"WROTE: {manifest_path}")
    return manifest_path


def _write_label_map(
    output_dir: Path,
    label_map_name: str,
    label_map: dict[int, dict],
    force: bool,
) -> Path:
    """Write per-cell contributor table beside the strict TIFF.

    Columns: ``cell_id, n_488, n_560, ids_488, ids_560, total_vox``. The id
    lists are semicolon-joined so the file parses as plain CSV.
    """
    csv_path = output_dir / label_map_name
    if csv_path.exists() and not force:
        print(f"SKIP (exists): {csv_path}")
        return csv_path

    tmp_path = str(csv_path) + ".tmp"
    with open(tmp_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_id", "n_488", "n_560", "ids_488", "ids_560", "total_vox"])
        for cid in sorted(label_map.keys()):
            entry = label_map[cid]
            writer.writerow(
                [
                    cid,
                    entry["n_488"],
                    entry["n_560"],
                    ";".join(str(x) for x in entry["ids_488"]),
                    ";".join(str(x) for x in entry["ids_560"]),
                    entry["total_vox"],
                ]
            )
    os.replace(tmp_path, str(csv_path))
    print(f"WROTE: {csv_path}  rows={len(label_map)}")
    return csv_path


def run(
    data_dir: Path | None,
    output_dir: Path,
    force: bool = False,
    skip_combined: bool = False,
    tau: float = 0.2,
    min_label_voxels: int = 0,
    closure: str = "component",
    allow_multimerge: bool = False,
    split_on_disagree: str = "off",
    tie_prefer: str = "488",
    drop_disconnected: bool = False,
    connectivity: int = 1,
    smooth_radius: int = 0,
    conflict_rule: str = "488_wins",
    conflict_merge_threshold: float = 0.05,
    output_name: str = "union_488_560_strict_overlap.tif",
    combined_name: str = "union_488_560_strict_overlap_combined.tif",
    manifest_name: str | None = None,
    label_map_name: str | None = None,
) -> int:
    if manifest_name is None:
        manifest_name = output_name.replace(".tif", "_manifest.json")
    if label_map_name is None:
        label_map_name = output_name.replace(".tif", "_label_map.csv")

    print(f"Data dir:   {data_dir if data_dir else '(not provided)'}")
    print(f"Output dir: {output_dir}")
    print(f"tau:                  {tau}")
    print(f"min_label_voxels:     {min_label_voxels}")
    print(f"closure:              {closure}")
    print(f"merge_policy:         {'multimerge' if allow_multimerge else 'strict_1_1'}")
    print(f"conflict_rule:        {conflict_rule}  merge_threshold={conflict_merge_threshold}")
    print(f"split_on_disagree:    {split_on_disagree}  tie_prefer={tie_prefer}")
    print(f"drop_disconnected:    {drop_disconnected}  connectivity={connectivity}")
    print(f"smooth_radius:        {smooth_radius}")
    print(f"output_name:          {output_name}")
    print(f"combined_name:        {combined_name}")
    print(f"manifest_name:        {manifest_name}")
    print(f"label_map_name:       {label_map_name}")
    print(f"skip_combined:        {skip_combined}\n")

    masks: dict[str, np.ndarray] = {}
    for ch in UNION_CHANNELS:
        try:
            mask_path = _find_indexed_mask(output_dir, ch)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"Loading {ch} mask: {mask_path}")
        masks[ch] = _load_3d(mask_path)

    ref_shape = masks[UNION_CHANNELS[0]].shape
    for ch in UNION_CHANNELS:
        if masks[ch].shape != ref_shape:
            if masks[ch].shape[1:] != ref_shape[1:]:
                raise ValueError(
                    f"Mask XY mismatch: {UNION_CHANNELS[0]} {ref_shape} vs {ch} {masks[ch].shape}"
                )
            z_min = min(ref_shape[0], masks[ch].shape[0])
            print(f"  WARNING: Z mismatch for {ch} mask, truncating both masks to {z_min}")
            for k in UNION_CHANNELS:
                masks[k] = masks[k][:z_min]
            ref_shape = masks[UNION_CHANNELS[0]].shape

    print("\nStrict-overlap IoMin label stitching...")
    union_mask, diag = union_488_560_labels(
        masks["488nm_crop"],
        masks["560nm_crop"],
        tau=tau,
        min_label_voxels=min_label_voxels,
        closure=closure,
        allow_multimerge=allow_multimerge,
        split_on_disagree=split_on_disagree,
        tie_prefer=tie_prefer,
        drop_disconnected=drop_disconnected,
        connectivity=connectivity,
        smooth_radius=smooth_radius,
        conflict_rule=conflict_rule,
        conflict_merge_threshold=conflict_merge_threshold,
    )
    _print_diagnostics(diag, union_mask)
    print()

    union_path = _write_union_volume(
        output_dir, union_mask, force, output_name=output_name
    )
    _write_manifest(output_dir, manifest_name, diag, union_path, force)
    _write_label_map(output_dir, label_map_name, diag.get("label_map", {}), force)

    combined_out = output_dir / combined_name
    if skip_combined:
        print(f"SKIP: --skip-combined set; not writing {combined_name}")
        print("Done.")
        return 0
    if combined_out.exists() and not force:
        print(f"SKIP (exists): {combined_out}")
        print("Done.")
        return 0
    if data_dir is None:
        print(
            f"WARNING: --data-dir not provided and --skip-combined not set; "
            f"cannot write {combined_name}. Skipping combined.tif."
        )
        print("Done.")
        return 0

    originals: dict[str, np.ndarray] = {}
    try:
        for ch in ALL_CHANNELS:
            orig_path = _find_original_volume(data_dir, ch)
            o = _load_3d(orig_path).astype(np.float32)
            originals[ch] = _crop_zyx_to_ref(o, ref_shape, ch)
    except (FileNotFoundError, ValueError) as exc:
        print(f"WARNING: cannot assemble combined.tif (originals): {exc}")
        print("Done.")
        return 0

    combined = np.stack(
        [
            originals["642nm_crop"],
            originals["488nm_crop"],
            originals["560nm_crop"],
            union_mask.astype(np.float32),
        ],
        axis=1,
    )
    tmp_path = str(combined_out) + ".tmp"
    tifffile.imwrite(
        tmp_path,
        combined,
        bigtiff=True,
        ome=True,
        photometric="minisblack",
        compression="zlib",
        metadata={
            "axes": "ZCYX",
            "Channel": {
                "Name": [
                    "642_Original",
                    "488_Original",
                    "560_Original",
                    "Union_488_560_Strict_Overlap_Mask",
                ]
            },
        },
    )
    os.replace(tmp_path, str(combined_out))
    print(f"WROTE: {combined_out}")
    print("Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-rel", type=str, default=None,
                    help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Absolute input data dir (for combined.tif)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Absolute sample output dir (contains 488nm_crop/, 560nm_crop/, ...)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    ap.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not write the strict-overlap combined.tif (no originals needed)",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="IoMin threshold for fusing a 488 and 560 label and for the "
             "gate-2 pairwise closure check (default: 0.2)",
    )
    ap.add_argument(
        "--min-label-voxels",
        type=int,
        default=0,
        help="Drop labels smaller than this from either channel before stitching (default: 0)",
    )
    ap.add_argument(
        "--closure",
        choices=("component", "global"),
        default="component",
        help=(
            "Strict-overlap closure mode (default: component). 'component' "
            "checks pairwise IoMin only inside each connected component "
            "(handles transitive UF overmerges). 'global' is the stricter "
            "opt-in rule: drop any label that has ANY cross-overlap with a "
            "label in the other channel where 0 < IoMin < tau, then rebuild "
            "the pair census on the cleaned masks."
        ),
    )
    ap.add_argument(
        "--allow-multimerge",
        action="store_true",
        help=(
            "Allow n-to-m fusion within a component (the gate-2-only "
            "behavior). Default is strict 1-1: every surviving cell has "
            "exactly one 488 contributor and one 560 contributor; "
            "components with over-segmentation or n-to-m fusion are dropped "
            "entirely. Pass this flag to keep the relaxed multi-merge "
            "behavior (e.g. for debugging or auditing)."
        ),
    )
    ap.add_argument(
        "--split-on-disagree",
        choices=("off", "clean", "all"),
        default="off",
        help=(
            "Mode-3 fix: split components where 488 and 560 disagree on cell count. "
            "'off' (default): keep current fused behavior. 'clean': only split unambiguous "
            "1-vs-N or N-vs-1 components. 'all': also split 2-vs-3 etc. (uses --tie-prefer "
            "when both sides have the same N>=2)."
        ),
    )
    ap.add_argument(
        "--tie-prefer",
        choices=("488", "560"),
        default="488",
        help="Channel preference when c488 == c560 > 1 in --split-on-disagree all (default: 488)",
    )
    ap.add_argument(
        "--drop-disconnected",
        action="store_true",
        help=(
            "Mode-2 fix: keep only the largest 3D connected component per label "
            "(removes conflict-voxel slivers like cell-37 sample bump)."
        ),
    )
    ap.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="3D connectivity for --drop-disconnected (1=face/6-conn, 2=18-conn, 3=26-conn)",
    )
    ap.add_argument(
        "--smooth-radius",
        type=int,
        default=0,
        help=(
            "Mode-1 fix (optional): per-label 3D morphological closing radius. 0 disables. "
            "Only fills background voxels within bbox; never overwrites other labels."
        ),
    )
    ap.add_argument(
        "--conflict-rule",
        choices=("488_wins", "560_wins", "larger_wins", "smaller_wins", "merged_loses"),
        default="488_wins",
        help=(
            "Mode-4 fix: rule for resolving voxels where 488 and 560 disagree on which "
            "union cell they belong to. '488_wins' (default): original behavior. "
            "'560_wins': flip preference. 'larger_wins'/'smaller_wins': pick by global "
            "label volume. 'merged_loses': pick the side whose label is NOT a 'merger' "
            "(i.e. only overlaps one significant other-channel label above "
            "--conflict-merge-threshold); 488 wins on ties/no-info."
        ),
    )
    ap.add_argument(
        "--conflict-merge-threshold",
        type=float,
        default=0.05,
        help=(
            "IoMin threshold above which an overlap counts as 'significant' for "
            "--conflict-rule merged_loses (default: 0.05)."
        ),
    )
    ap.add_argument(
        "--output-name",
        type=str,
        default="union_488_560_strict_overlap.tif",
        help="Output mask filename (default: union_488_560_strict_overlap.tif)",
    )
    ap.add_argument(
        "--combined-name",
        type=str,
        default="union_488_560_strict_overlap_combined.tif",
        help="4-channel combined.tif filename (default: union_488_560_strict_overlap_combined.tif)",
    )
    ap.add_argument(
        "--manifest-name",
        type=str,
        default=None,
        help="Manifest JSON filename (default: derived from --output-name)",
    )
    ap.add_argument(
        "--label-map-name",
        type=str,
        default=None,
        help="Label-map CSV filename (default: derived from --output-name)",
    )
    args = ap.parse_args()
    data_dir, output_dir = _resolve_dirs(args)
    return run(
        data_dir,
        output_dir,
        force=args.force,
        skip_combined=args.skip_combined,
        tau=args.tau,
        min_label_voxels=args.min_label_voxels,
        closure=args.closure,
        allow_multimerge=args.allow_multimerge,
        split_on_disagree=args.split_on_disagree,
        tie_prefer=args.tie_prefer,
        drop_disconnected=args.drop_disconnected,
        connectivity=args.connectivity,
        smooth_radius=args.smooth_radius,
        conflict_rule=args.conflict_rule,
        conflict_merge_threshold=args.conflict_merge_threshold,
        output_name=args.output_name,
        combined_name=args.combined_name,
        manifest_name=args.manifest_name,
        label_map_name=args.label_map_name,
    )


if __name__ == "__main__":
    sys.exit(main())
