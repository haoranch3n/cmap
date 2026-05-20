"""
2D Cellpose segmentation with multiscale diameter merging.

Adapted from segmentation/cellpose2d/segmentation_cellpose_2d.py for use with
registered cell crop per-channel planes.

Phase 1: Run Cellpose at each diameter in CELLPOSE_DIAMETERS, save per-diameter
         masks to ``<diameters_root>/``.
Phase 2: Merge masks from largest to smallest diameter, filter small objects,
         split merged cells, run connected-component cleanup, write one final
         mask per plane to ``<seg_root>/``.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from pipeline_config import (
    AREA_THRESHOLD,
    CELLPOSE_CELLPROB_THRESHOLD,
    CELLPOSE_DIAMETERS,
    CELLPOSE_EVAL_NORMALIZE,
    CELLPOSE_FLOW_THRESHOLD,
    CELLPOSE_PRETRAINED_MODEL,
    SEG_PLANE_TAGS,
    SEGMENTATION_GLOBAL_VOLUME_PERCENTILES,
    SEGMENTATION_NORM_BOUNDS_CSV,
    strip_path_shared_with_output_mirror,
)

try:
    import torch

    def gpu_available() -> bool:
        return torch.cuda.is_available()
except Exception:
    def gpu_available() -> bool:
        return False


def discover_seg_plane_tifs(tif_planes_root: Path) -> list[str]:
    """Find all per-Z segmentation input planes for any known channel tag."""
    found: set[str] = set()
    for tag in SEG_PLANE_TAGS:
        found.update(glob.glob(str(tif_planes_root / "**" / f"*_{tag}.tif"), recursive=True))
    return sorted(found)


# Backward-compat alias.
discover_dapi_tifs = discover_seg_plane_tifs


def plane_stem_from_seg_path(seg_path: str) -> str:
    """Strip the channel-tag suffix (e.g. ``_ch642``, ``_488``) from a plane filename."""
    base = os.path.basename(seg_path)
    for tag in SEG_PLANE_TAGS:
        suffix = f"_{tag}.tif"
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return Path(base).stem


plane_stem_from_dapi_path = plane_stem_from_seg_path

_PLANE_TZ_RE = re.compile(r"_t(\d+)_z(\d+)_")


def sort_plane_paths_by_tz(plane_paths: list[str]) -> list[str]:
    def sort_key(p: str) -> tuple:
        m = _PLANE_TZ_RE.search(os.path.basename(p))
        if m:
            return (0, int(m.group(1)), int(m.group(2)))
        return (1, os.path.basename(p))
    return sorted(plane_paths, key=sort_key)


def group_dapi_paths_by_volume_dir(dapi_files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for p in dapi_files:
        groups[os.path.realpath(os.path.dirname(p))].append(p)
    return {k: sort_plane_paths_by_tz(v) for k, v in groups.items()}


def dapi_volume_percentile_bounds(
    plane_paths: list[str], p_low: float, p_high: float
) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    for path in plane_paths:
        img = imread(path)
        if img.ndim == 3:
            img = img.squeeze()
        if img.ndim != 2:
            raise ValueError(f"Expected 2D plane, got shape {getattr(img, 'shape', None)} for {path}")
        chunks.append(np.asarray(img, dtype=np.float32).ravel())
    if not chunks:
        return 0.0, 1.0
    flat = np.concatenate(chunks)
    lo, hi = np.percentile(flat, [p_low, p_high])
    lo_f, hi_f = float(lo), float(hi)
    if hi_f - lo_f <= 1e-3:
        return 0.0, 1.0
    return lo_f, hi_f


def apply_global_volume_affine(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Affine normalize: (x - lo) / (hi - lo); no clipping."""
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3:
        x = x.squeeze()
    return (x - lo) / (hi - lo)


def read_norm_bounds_csv(csv_path: Path) -> tuple[float, float] | None:
    """Read channel 0 (lo, hi) from a ``segmentation_norm_bounds.csv``."""
    try:
        with open(csv_path, newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith("#") or row[0] == "channel_index":
                    continue
                if int(row[0]) == 0:
                    return float(row[1]), float(row[2])
    except Exception:
        pass
    return None


def diameter_mask_dir(seg_path: str, tif_planes_root: str, diameters_root: str) -> str:
    rel_parent = strip_path_shared_with_output_mirror(
        os.path.relpath(os.path.dirname(seg_path), tif_planes_root)
    )
    stem = plane_stem_from_seg_path(seg_path)
    out_dir = (
        os.path.join(diameters_root, rel_parent, stem)
        if rel_parent
        else os.path.join(diameters_root, stem)
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def output_final_mask_path(seg_path: str, tif_planes_root: str, seg_root: str) -> str:
    rel_parent = strip_path_shared_with_output_mirror(
        os.path.relpath(os.path.dirname(seg_path), tif_planes_root)
    )
    stem = plane_stem_from_seg_path(seg_path)
    out_dir = (
        os.path.join(seg_root, rel_parent, stem)
        if rel_parent
        else os.path.join(seg_root, stem)
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{stem}_final_mask.tif")


def extract_coords_by_label(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {}, {}
    labels = mask[ys, xs].astype(np.int64)
    order = np.argsort(labels)
    sorted_labels = labels[order]
    sorted_ys = ys[order]
    sorted_xs = xs[order]
    unique_labels, split_points = np.unique(sorted_labels, return_index=True)
    coords = {}
    areas = {}
    for idx, lbl in enumerate(unique_labels):
        if lbl == 0:
            continue
        start = split_points[idx]
        end = split_points[idx + 1] if idx + 1 < len(split_points) else len(sorted_labels)
        lbl_int = int(lbl)
        coords[lbl_int] = np.array([sorted_ys[start:end], sorted_xs[start:end]])
        areas[lbl_int] = end - start
    return coords, areas


def map_smaller_to_combined(smaller_mask: np.ndarray, combined_mask: np.ndarray, current_max_id: int):
    updated = np.zeros_like(smaller_mask, dtype=np.int32)
    flat_indices = np.flatnonzero(smaller_mask)
    if flat_indices.size == 0:
        return updated, current_max_id
    flat_labels = smaller_mask.flat[flat_indices]
    unique_labels = np.unique(flat_labels)
    order = np.argsort(flat_labels)
    sorted_flat_labels = flat_labels[order]
    sorted_flat_indices = flat_indices[order]
    label_starts = np.searchsorted(sorted_flat_labels, unique_labels, side="left")
    label_ends = np.searchsorted(sorted_flat_labels, unique_labels, side="right")
    for idx, lbl in enumerate(unique_labels):
        indices = sorted_flat_indices[label_starts[idx]: label_ends[idx]]
        overlap_vals = combined_mask.flat[indices].astype(np.intp)
        if overlap_vals.size == 0:
            continue
        bincount = np.bincount(overlap_vals)
        if len(bincount) > 1:
            bincount[0] = 0
            assigned_id = int(bincount.argmax())
        else:
            assigned_id = 0
        if assigned_id == 0:
            current_max_id += 1
            assigned_id = current_max_id
        updated.flat[indices] = assigned_id
    return updated, current_max_id


def filter_small_objects(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    areas = np.bincount(mask.ravel().astype(np.intp))
    large_labels = np.flatnonzero(areas >= area_threshold)
    large_labels = large_labels[large_labels != 0]
    return np.where(np.isin(mask, large_labels), mask, 0).astype(np.int32)


def split_merged_cells(filtered_mask: np.ndarray, all_masks_sorted: list[np.ndarray], threshold: float = 0.8):
    current_max_id = int(filtered_mask.max())
    splitted_cells = set()
    fm_coords, fm_areas = extract_coords_by_label(filtered_mask)
    smaller_coords_list = []
    for mask in all_masks_sorted[1:]:
        coords, _ = extract_coords_by_label(mask.astype(np.int32))
        smaller_coords_list.append(coords)

    updates = []
    for label_id in sorted(fm_coords.keys()):
        if label_id in splitted_cells:
            continue
        coords = fm_coords[label_id]
        cell_area = fm_areas[label_id]
        for mask_idx, smaller_mask in enumerate(all_masks_sorted[1:]):
            ys, xs = coords[0], coords[1]
            overlap = smaller_mask[ys, xs].ravel()
            overlap = overlap[overlap > 0]
            if overlap.size == 0:
                continue
            counts = np.bincount(overlap.astype(np.intp))
            if len(counts) <= 1:
                continue
            counts[0] = 0
            nonzero_ids = np.flatnonzero(counts)
            nonzero_counts = counts[nonzero_ids]
            sorted_idx = np.argsort(nonzero_counts)[::-1]
            sorted_ids = nonzero_ids[sorted_idx]
            cumulative = 0
            significant = []
            for cid in sorted_ids:
                cumulative += counts[cid]
                significant.append(int(cid))
                if cumulative / cell_area > threshold:
                    break
            if len(significant) > 1 and cumulative / cell_area > threshold:
                updates.append((coords, 0))
                largest = significant[0]
                sm_coords = smaller_coords_list[mask_idx]
                if largest in sm_coords:
                    updates.append((sm_coords[largest], label_id))
                for cell in significant[1:]:
                    current_max_id += 1
                    if cell in sm_coords:
                        updates.append((sm_coords[cell], current_max_id))
                splitted_cells.update(significant)
                break

    for coords, val in updates:
        filtered_mask[coords[0], coords[1]] = val
    if splitted_cells:
        print(f"    Split {len(splitted_cells)} cell(s)")
    return filtered_mask


def final_cleanup(mask: np.ndarray, area_threshold: int, connectivity: int = 1) -> np.ndarray:
    mask, _, _ = relabel_sequential(mask.astype(np.int32))
    new_mask = np.zeros_like(mask, dtype=np.int32)
    new_id = 1
    props = regionprops(mask, cache=False)
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        region = mask[minr:maxr, minc:maxc] == prop.label
        labeled_local, n_components = label(region, connectivity=connectivity, return_num=True)
        if n_components > 1:
            sizes = np.bincount(labeled_local.ravel())[1:]
            largest_cc = int(np.argmax(sizes)) + 1
            region = labeled_local == largest_cc
        area = int(region.sum())
        if area >= area_threshold:
            view = new_mask[minr:maxr, minc:maxc]
            view[region] = new_id
            new_id += 1
    return new_mask


def merge_diameter_masks(seg_path: str, diameters: list[int], tif_planes_root: str, diameters_root: str):
    stem = plane_stem_from_seg_path(seg_path)
    dm_dir = os.path.join(
        diameters_root,
        os.path.relpath(os.path.dirname(seg_path), tif_planes_root),
        stem,
    )
    sorted_diameters = sorted(diameters, reverse=True)
    masks = []
    for diameter in sorted_diameters:
        path = os.path.join(dm_dir, f"{stem}_diameter_{diameter}.tif")
        if os.path.exists(path):
            masks.append(tifffile.imread(path).astype(np.int32))
    if not masks:
        return None
    combined = masks[0].copy()
    current_max_id = int(combined.max())
    for smaller in masks[1:]:
        updated, current_max_id = map_smaller_to_combined(smaller, combined, current_max_id)
        combined = np.where(updated > 0, updated, combined).astype(np.int32)
    combined = filter_small_objects(combined, AREA_THRESHOLD)
    combined = final_cleanup(combined, AREA_THRESHOLD)
    return combined


def run_segmentation(
    tif_planes_root,
    seg_root,
    diameters_root,
    pretrained_model=None,
    gpu=None,
) -> None:
    """
    Run multiscale 2D Cellpose segmentation on planes under *tif_planes_root*.

    Args:
        tif_planes_root: Directory containing per-Z plane TIFFs.
        seg_root:        Output directory for merged per-plane masks.
        diameters_root:  Output directory for per-diameter masks.
        pretrained_model: Cellpose model name (default: CELLPOSE_PRETRAINED_MODEL).
        gpu:             Whether to use GPU (default: auto-detect).
    """
    tif_planes_root = Path(tif_planes_root)
    seg_root = Path(seg_root)
    diameters_root = Path(diameters_root)
    pretrained_model = pretrained_model or CELLPOSE_PRETRAINED_MODEL
    if gpu is None:
        gpu = gpu_available()

    seg_files = discover_seg_plane_tifs(tif_planes_root)
    if not seg_files:
        print(f"No segmentation input planes found under {tif_planes_root}")
        return

    volume_groups = group_dapi_paths_by_volume_dir(seg_files)
    diameters = CELLPOSE_DIAMETERS

    print(f"Using Cellpose pretrained_model={pretrained_model!r}, gpu={gpu}")
    print(f"Diameters: {diameters}")
    print(
        f"Cellpose eval normalize={CELLPOSE_EVAL_NORMALIZE!r}  "
        f"[global volume percentiles={SEGMENTATION_GLOBAL_VOLUME_PERCENTILES!r}]"
    )
    from cellpose import models
    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    print("\n=== Phase 1: Multi-diameter Cellpose segmentation (batched per diameter) ===")
    # Process all planes at once per diameter for GPU efficiency.
    # NOTE: planes are pre-normalized by extract_channel.py (values in [0,1]);
    # we do NOT apply affine normalization here. The CSV is read only for logging.
    for vol_dir, plane_paths in volume_groups.items():
        if not CELLPOSE_EVAL_NORMALIZE:
            csv_path = Path(vol_dir) / SEGMENTATION_NORM_BOUNDS_CSV
            bounds_from_csv = read_norm_bounds_csv(csv_path)
            if bounds_from_csv is not None:
                lo_b, hi_b = bounds_from_csv
                print(f"  Planes pre-normalized (lo={lo_b:.6g}, hi={hi_b:.6g}) — no re-normalization applied.")
            else:
                print(f"  [warn] No norm CSV in {vol_dir}; planes used as-is.")

        # Load all planes for this volume (already normalized).
        imgs: list[np.ndarray] = []
        for seg_path in plane_paths:
            img = imread(seg_path)
            if img.ndim == 3:
                img = img.squeeze()
            if img.ndim != 2:
                raise ValueError(f"Expected 2D plane, got shape {img.shape} for {seg_path}")
            imgs.append(np.asarray(img, dtype=np.float32))

        for diameter in tqdm(diameters, desc=f"  Cellpose diameters ({Path(vol_dir).name})"):
            # Check if all planes already done for this diameter.
            out_paths = []
            todo_indices = []
            todo_imgs = []
            for i, seg_path in enumerate(plane_paths):
                dm_dir = diameter_mask_dir(
                    seg_path, os.fspath(tif_planes_root), os.fspath(diameters_root)
                )
                stem = plane_stem_from_seg_path(seg_path)
                out_path = os.path.join(dm_dir, f"{stem}_diameter_{diameter}.tif")
                out_paths.append(out_path)
                if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
                    todo_indices.append(i)
                    todo_imgs.append(imgs[i])

            if not todo_imgs:
                continue

            # Batch eval: one GPU call for all pending planes at this diameter.
            batch_masks, _, _ = model.eval(
                todo_imgs,
                diameter=diameter,
                channels=None,
                normalize=CELLPOSE_EVAL_NORMALIZE,
                flow_threshold=CELLPOSE_FLOW_THRESHOLD,
                cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
            )
            for plane_idx, mask in zip(todo_indices, batch_masks):
                tifffile.imwrite(out_paths[plane_idx], np.asarray(mask).astype(np.uint32))

    print("\n=== Phase 2: Merging multiscale masks ===")
    for seg_path in tqdm(seg_files, desc="Merging masks"):
        out_mask = output_final_mask_path(
            seg_path, os.fspath(tif_planes_root), os.fspath(seg_root)
        )
        if os.path.exists(out_mask) and os.path.getsize(out_mask) > 0:
            continue
        final = merge_diameter_masks(
            seg_path, diameters, os.fspath(tif_planes_root), os.fspath(diameters_root)
        )
        if final is not None:
            tifffile.imwrite(out_mask, final.astype(np.uint16))
