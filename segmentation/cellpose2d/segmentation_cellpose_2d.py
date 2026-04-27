"""
2D Cellpose segmentation with multiscale diameter merging.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

_SEG_ROOT = Path(__file__).resolve().parents[1]
if str(_SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEG_ROOT))

try:
    from config import (
        AREA_THRESHOLD,
        CELLPOSE_CELLPROB_THRESHOLD,
        CELLPOSE_DIAMETERS,
        CELLPOSE_FLOW_THRESHOLD,
        CELLPOSE_PRETRAINED_MODEL,
        SEGMENTATION_2D_DIAMETERS_DIR,
        SEGMENTATION_2D_DIR,
        TIF_PLANES_DIR,
        strip_path_shared_with_output_mirror,
    )
except ModuleNotFoundError:
    from segmentation.config import (
        AREA_THRESHOLD,
        CELLPOSE_CELLPROB_THRESHOLD,
        CELLPOSE_DIAMETERS,
        CELLPOSE_FLOW_THRESHOLD,
        CELLPOSE_PRETRAINED_MODEL,
        SEGMENTATION_2D_DIAMETERS_DIR,
        SEGMENTATION_2D_DIR,
        TIF_PLANES_DIR,
        strip_path_shared_with_output_mirror,
    )

try:
    import torch

    def gpu_available() -> bool:
        return torch.cuda.is_available()
except Exception:
    def gpu_available() -> bool:
        return False

from cellpose import models


def discover_dapi_tifs(tif_planes_root: Path):
    return sorted(glob.glob(str(tif_planes_root / "**" / "*_DAPI.tif"), recursive=True))


def plane_stem_from_dapi_path(dapi_path: str) -> str:
    base = os.path.basename(dapi_path)
    if base.endswith("_DAPI.tif"):
        return base[: -len("_DAPI.tif")]
    return Path(base).stem


def diameter_mask_dir(dapi_path: str, tif_planes_root: str, diameters_root: str) -> str:
    rel_parent = strip_path_shared_with_output_mirror(
        os.path.relpath(os.path.dirname(dapi_path), tif_planes_root)
    )
    stem = plane_stem_from_dapi_path(dapi_path)
    out_dir = os.path.join(diameters_root, rel_parent, stem) if rel_parent else os.path.join(diameters_root, stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def output_final_mask_path(dapi_path: str, tif_planes_root: str, seg_root: str) -> str:
    rel_parent = strip_path_shared_with_output_mirror(
        os.path.relpath(os.path.dirname(dapi_path), tif_planes_root)
    )
    stem = plane_stem_from_dapi_path(dapi_path)
    out_dir = os.path.join(seg_root, rel_parent, stem) if rel_parent else os.path.join(seg_root, stem)
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


def merge_diameter_masks(dapi_path: str, diameters: list[int], tif_planes_root: str, diameters_root: str):
    stem = plane_stem_from_dapi_path(dapi_path)
    dm_dir = os.path.join(
        diameters_root,
        os.path.relpath(os.path.dirname(dapi_path), tif_planes_root),
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


def run_segmentation(tif_planes_root=None, seg_root=None, diameters_root=None, pretrained_model=None, gpu=None):
    tif_planes_root = Path(tif_planes_root or TIF_PLANES_DIR)
    seg_root = Path(seg_root or SEGMENTATION_2D_DIR)
    diameters_root = Path(diameters_root or SEGMENTATION_2D_DIAMETERS_DIR)
    pretrained_model = pretrained_model or CELLPOSE_PRETRAINED_MODEL
    if gpu is None:
        gpu = gpu_available()

    dapi_files = discover_dapi_tifs(tif_planes_root)
    if not dapi_files:
        print(f"No *_DAPI.tif files under {tif_planes_root}")
        return

    diameters = CELLPOSE_DIAMETERS
    print(f"Using Cellpose pretrained_model={pretrained_model!r}, gpu={gpu}")
    print(f"Diameters: {diameters}")
    print(f"Found {len(dapi_files)} DAPI plane(s).")
    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    print("\n=== Phase 1: Multi-diameter Cellpose segmentation ===")
    for dapi_path in tqdm(dapi_files, desc="Cellpose 2D (multi-diameter)"):
        dm_dir = diameter_mask_dir(dapi_path, os.fspath(tif_planes_root), os.fspath(diameters_root))
        stem = plane_stem_from_dapi_path(dapi_path)
        all_done = all(
            os.path.exists(os.path.join(dm_dir, f"{stem}_diameter_{d}.tif"))
            and os.path.getsize(os.path.join(dm_dir, f"{stem}_diameter_{d}.tif")) > 0
            for d in diameters
        )
        if all_done:
            continue
        img = imread(dapi_path)
        if img.ndim == 3:
            img = img.squeeze()
        if img.ndim != 2:
            raise ValueError(f"Expected 2D plane, got shape {img.shape} for {dapi_path}")
        img = np.asarray(img, dtype=np.float32)
        for diameter in diameters:
            out_path = os.path.join(dm_dir, f"{stem}_diameter_{diameter}.tif")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                continue
            masks, _, _ = model.eval(
                img,
                diameter=diameter,
                channels=None,
                normalize=True,
                flow_threshold=CELLPOSE_FLOW_THRESHOLD,
                cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
            )
            tifffile.imwrite(out_path, np.asarray(masks).astype(np.uint32))

    print("\n=== Phase 2: Merging multiscale masks ===")
    for dapi_path in tqdm(dapi_files, desc="Merging masks"):
        out_mask = output_final_mask_path(dapi_path, os.fspath(tif_planes_root), os.fspath(seg_root))
        if os.path.exists(out_mask) and os.path.getsize(out_mask) > 0:
            continue
        final = merge_diameter_masks(dapi_path, diameters, os.fspath(tif_planes_root), os.fspath(diameters_root))
        if final is not None:
            tifffile.imwrite(out_mask, final.astype(np.uint16))


if __name__ == "__main__":
    run_segmentation()

