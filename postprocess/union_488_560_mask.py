#!/usr/bin/env python3
"""
Merge the 488nm and 560nm Cellpose indexed masks into a single indexed cell
mask using IoMin label stitching.

Algorithm
---------

For every pair of labels ``(a, b)`` where ``a`` is a 488 cell and ``b`` is a
560 cell with at least one shared voxel, compute::

    IoMin(a, b) = |a INTERSECT b| / min(|a|, |b|)

Build a bipartite graph with edges ``(a, b)`` whenever ``IoMin >= tau``. The
connected components of that graph define the merged cells:

- Component ``{a}`` only: 488-only cell with no 560 match.
- Component ``{b}`` only: 560-only cell with no 488 match.
- Component ``{a, b}``: clean 1-1 pair across channels.
- Component ``{a, b1, b2}`` or larger: same biological cell over-segmented in
  one channel; both channels' voxels go to one fused ID.

Each component is assigned a contiguous ID ``1..K`` (sorted by total voxel
count, descending). Cellpose IDs are mapped through per-channel LUTs so the
output is a true indexed label volume that preserves cell identity. A 488 cell
and a 560 cell that do **not** co-locate are never fused, no matter how close
they sit in space.

Tie-breaker on overlap voxels where 488 and 560 disagree on the component
(possible when the pair's IoMin is below ``tau``): 488 wins. The disagreeing
voxel count is reported as ``conflict_voxels``.

**Writes** (by default):

- ``<output>/union_488_560.tif`` — indexed 3D label volume (``uint16`` if
  ``K <= 65535`` else ``uint32``).
- ``<output>/union_488_560_combined.tif`` — 4-channel OME BigTIFF (Z, C, Y, X):
  642 / 488 / 560 originals + union mask. Use ``--skip-combined`` to skip.

CLI
---

- ``--tau FLOAT`` (default ``0.2``): IoMin threshold for fusion.
- ``--min-label-voxels INT`` (default ``0``): drop labels smaller than this in
  either channel before stitching (guards against tiny spurious fragments).

Notes
-----

- Inputs are the per-channel indexed masks produced by segmentation:
  ``<output>/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif`` and
  ``<output>/560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif``.
- Z-mismatch handling matches ``postprocess/filter_642_mask.py``: if Z differs,
  truncate both volumes to the smallest Z; XY mismatch raises.
- This script does not touch ``filtered_642*`` artifacts.
"""
from __future__ import annotations

import argparse
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


def union_488_560_labels(
    m488: np.ndarray,
    m560: np.ndarray,
    *,
    tau: float = 0.2,
    min_label_voxels: int = 0,
    split_on_disagree: str = "off",
    tie_prefer: str = "488",
    drop_disconnected: bool = False,
    connectivity: int = 1,
    smooth_radius: int = 0,
    conflict_rule: str = "488_wins",
    conflict_merge_threshold: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """IoMin label-stitching merge of two indexed Cellpose masks.

    Parameters
    ----------
    m488, m560 : np.ndarray
        ZYX integer label volumes; same shape; 0 = background.
    tau : float
        IoMin threshold for fusing a 488 label with a 560 label.
    min_label_voxels : int
        Drop labels smaller than this from either channel before stitching;
        their voxels become 0 in the output.
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

    sorted_roots = sorted(root_total.keys(), key=lambda r: (-root_total[r], r))
    K = len(sorted_roots)
    root_to_id = {r: i + 1 for i, r in enumerate(sorted_roots)}

    lut_488 = np.zeros(n488_max + 1, dtype=np.int64)
    for a, r in zip(valid_488_ids.tolist(), roots_488.tolist()):
        lut_488[a] = root_to_id[int(r)]
    lut_560 = np.zeros(n560_max + 1, dtype=np.int64)
    for b, r in zip(valid_560_ids.tolist(), roots_560.tolist()):
        lut_560[b] = root_to_id[int(r)]

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

    diagnostics = {
        "N488": int(valid_488_ids.size),
        "N560": int(valid_560_ids.size),
        "tau": float(tau),
        "min_label_voxels": int(min_label_voxels),
        "edges": n_edges,
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
        f"tau={diag['tau']:.3f}, min_label_voxels={diag['min_label_voxels']}"
    )
    print(f"  edges (IoMin >= {diag['tau']:.2f}): {diag['edges']}")
    print(
        f"  components K = {diag['K']} "
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
    output_name: str = "union_488_560.tif",
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


def run(
    data_dir: Path | None,
    output_dir: Path,
    force: bool = False,
    skip_combined: bool = False,
    tau: float = 0.2,
    min_label_voxels: int = 0,
    split_on_disagree: str = "off",
    tie_prefer: str = "488",
    drop_disconnected: bool = False,
    connectivity: int = 1,
    smooth_radius: int = 0,
    conflict_rule: str = "488_wins",
    conflict_merge_threshold: float = 0.05,
    output_name: str = "union_488_560.tif",
    combined_name: str = "union_488_560_combined.tif",
) -> int:
    print(f"Data dir:   {data_dir if data_dir else '(not provided)'}")
    print(f"Output dir: {output_dir}")
    print(f"tau:                  {tau}")
    print(f"min_label_voxels:     {min_label_voxels}")
    print(f"conflict_rule:        {conflict_rule}  merge_threshold={conflict_merge_threshold}")
    print(f"split_on_disagree:    {split_on_disagree}  tie_prefer={tie_prefer}")
    print(f"drop_disconnected:    {drop_disconnected}  connectivity={connectivity}")
    print(f"smooth_radius:        {smooth_radius}")
    print(f"output_name:          {output_name}")
    print(f"combined_name:        {combined_name}")
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

    print("\nIoMin label stitching...")
    union_mask, diag = union_488_560_labels(
        masks["488nm_crop"],
        masks["560nm_crop"],
        tau=tau,
        min_label_voxels=min_label_voxels,
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

    _write_union_volume(output_dir, union_mask, force, output_name=output_name)

    combined_out = output_dir / combined_name
    if skip_combined:
        print("SKIP: --skip-combined set; not writing union_488_560_combined.tif")
        print("Done.")
        return 0
    if combined_out.exists() and not force:
        print(f"SKIP (exists): {combined_out}")
        print("Done.")
        return 0
    if data_dir is None:
        print(
            "WARNING: --data-dir not provided and --skip-combined not set; "
            "cannot write union_488_560_combined.tif. Skipping combined.tif."
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
                    "Union_488_560_Mask",
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
        help="Do not write union_488_560_combined.tif (no originals needed)",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="IoMin threshold for fusing a 488 and 560 label (default: 0.2)",
    )
    ap.add_argument(
        "--min-label-voxels",
        type=int,
        default=0,
        help="Drop labels smaller than this from either channel before stitching (default: 0)",
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
        default="union_488_560.tif",
        help="Output mask filename in the sample dir (default: union_488_560.tif)",
    )
    ap.add_argument(
        "--combined-name",
        type=str,
        default="union_488_560_combined.tif",
        help="4-channel combined.tif filename in the sample dir (default: union_488_560_combined.tif)",
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
        split_on_disagree=args.split_on_disagree,
        tie_prefer=args.tie_prefer,
        drop_disconnected=args.drop_disconnected,
        connectivity=args.connectivity,
        smooth_radius=args.smooth_radius,
        conflict_rule=args.conflict_rule,
        conflict_merge_threshold=args.conflict_merge_threshold,
        output_name=args.output_name,
        combined_name=args.combined_name,
    )


if __name__ == "__main__":
    sys.exit(main())
