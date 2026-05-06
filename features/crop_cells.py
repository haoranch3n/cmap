#!/usr/bin/env python3
"""
Crop individual 3D cells from filtered_642_combined.tif with margin.

With ``--also-full-z``, also writes full-Z-depth crops (same XY padding as
``cell_boxing/``) under ``cell_boxing_full_z/`` for each cell.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files

OUTPUT_SUBDIR_KEYS = ("cell_box", "cell_box_filtered")

MARGIN_XY_DEFAULT = 20
MARGIN_Z_DEFAULT = 5
_IMAGEJ_MAX_BYTES = 3_900_000_000


def _resolve_dirs(args) -> tuple[Path, Path]:
    if args.data_rel:
        data_dir = Path(DATA_DIR) / args.data_rel
        output_dir = (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
    elif args.data_dir and args.output_dir:
        data_dir = args.data_dir.resolve()
        output_dir = args.output_dir.resolve()
    else:
        raise SystemExit("Provide either --data-rel or both --data-dir and --output-dir")
    return data_dir, output_dir


def _load_3d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D or 4-D, got shape {arr.shape} from {path}")
    return arr


def _cell_box_dirnames(v: dict[str, str], output_subdir_key: str) -> tuple[str, str]:
    if output_subdir_key == "cell_box":
        return v["cell_box"], v["cell_box_full_z"]
    if output_subdir_key == "cell_box_filtered":
        return v["cell_box_filtered"], v["cell_box_full_z_filtered"]
    raise ValueError(f"output_subdir_key must be one of {OUTPUT_SUBDIR_KEYS}")


def _find_unfiltered_mask(output_dir: Path) -> Path:
    seg_dir = output_dir / "642nm_crop" / "segmentation_3D_masks"
    direct = seg_dir / "642nm_crop_3D_indexed.tif"
    if direct.is_file():
        return direct
    matches = sorted(p for p in seg_dir.rglob("*_3D_indexed.tif") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"No *_3D_indexed.tif under {seg_dir}")
    return matches[0]


def _save_crop(path: Path, data: np.ndarray, channel_names: list[str]) -> None:
    tmp = str(path) + ".tmp"
    nbytes = data.nbytes
    if nbytes < _IMAGEJ_MAX_BYTES:
        tifffile.imwrite(
            tmp,
            data,
            imagej=True,
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "ZCYX", "mode": "grayscale"},
        )
    else:
        tifffile.imwrite(
            tmp,
            data,
            bigtiff=True,
            ome=True,
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "ZCYX", "Channel": {"Name": channel_names}},
        )
    os.replace(tmp, str(path))


def _five_channel_crop(
    crop_combined: np.ndarray,
    crop_unfiltered: np.ndarray,
    lbl: int,
) -> tuple[np.ndarray, dict]:
    """Build (Z, 5, Y, X) float32 crop and per-crop stats (no bbox / margins)."""
    primary_binary = (crop_unfiltered == lbl).astype(np.float32)
    primary_volume = int(primary_binary.sum())
    neighbour_labels = set(int(v) for v in np.unique(crop_unfiltered)) - {0, lbl}
    neighbour_count = len(neighbour_labels)
    neighbour_voxels = int(np.isin(crop_unfiltered, list(neighbour_labels)).sum()) if neighbour_labels else 0
    total_fg = int((crop_unfiltered > 0).sum())
    neighbour_frac = neighbour_voxels / total_fg if total_fg > 0 else 0.0
    is_isolated = neighbour_count == 0

    boundary_vol = np.zeros_like(primary_binary)
    for zi in range(primary_binary.shape[0]):
        boundary_vol[zi] = find_boundaries(primary_binary[zi] > 0, mode="inner").astype(np.float32)

    intensity_3ch = crop_combined[:, :3, :, :]
    primary_ch = primary_binary[:, np.newaxis, :, :]
    boundary_ch = boundary_vol[:, np.newaxis, :, :]
    crop_5ch = np.concatenate([intensity_3ch, primary_ch, boundary_ch], axis=1).astype(np.float32)

    stats = {
        "primary_volume_voxels": primary_volume,
        "is_isolated": is_isolated,
        "neighbor_count": neighbour_count,
        "neighbor_voxels": neighbour_voxels,
        "neighbor_voxel_fraction": round(neighbour_frac, 4),
        "crop_shape_z": crop_5ch.shape[0],
        "crop_shape_y": crop_5ch.shape[2],
        "crop_shape_x": crop_5ch.shape[3],
    }
    return crop_5ch, stats


def run(
    data_dir: Path,
    output_dir: Path,
    margin_xy: int = MARGIN_XY_DEFAULT,
    margin_z: int = MARGIN_Z_DEFAULT,
    force: bool = False,
    also_full_z: bool = False,
    variant: str = DEFAULT_VARIANT,
    label_mask_name: str | None = None,
    output_subdir_key: str = "cell_box",
    cell_ids: set[int] | None = None,
) -> int:
    v = variant_files(variant)
    if output_subdir_key not in OUTPUT_SUBDIR_KEYS:
        raise ValueError(f"output_subdir_key must be one of {OUTPUT_SUBDIR_KEYS}")
    box_rel, full_z_rel = _cell_box_dirnames(v, output_subdir_key)
    mask_fname = label_mask_name or v["mask"]
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(
        f"Variant:    {variant}  (label_mask={mask_fname}, combined={v['combined']}, "
        f"out_subdir={output_subdir_key})"
    )
    print(f"Margin: XY={margin_xy}  Z={margin_z}" + ("  also_full_z=1" if also_full_z else ""))
    combined_path = output_dir / v["combined"]
    filtered_mask_path = output_dir / mask_fname
    for p, name in [(combined_path, v["combined"]), (filtered_mask_path, mask_fname)]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            return 1

    if variant == "filtered_642":
        try:
            unfiltered_mask_path = _find_unfiltered_mask(output_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1
    else:
        # Union variant has no separate "before-filter" Cellpose mask; the
        # union mask is its own canonical cell labeling. Neighbor stats then
        # reflect "other fused union cells inside this cell's bbox", which is
        # the natural meaning under the union variant.
        unfiltered_mask_path = filtered_mask_path

    box_dir = output_dir / box_rel
    full_z_dir = output_dir / full_z_rel
    summary_path = box_dir / "summary.csv"
    full_z_summary_path = full_z_dir / "summary.csv"

    run_padded = force or not summary_path.exists()
    run_full_z_pass = also_full_z and (force or not full_z_summary_path.exists())

    if not also_full_z:
        if summary_path.exists() and not force:
            print(f"SKIP (exists): {summary_path}  (use --force to overwrite)")
            return 0
        run_padded = True
        run_full_z_pass = False
    elif not run_padded and not run_full_z_pass:
        print(
            f"SKIP (exists): {summary_path} and {full_z_summary_path}  (use --force to overwrite)",
            flush=True,
        )
        return 0

    if run_padded:
        box_dir.mkdir(parents=True, exist_ok=True)
    if run_full_z_pass:
        full_z_dir.mkdir(parents=True, exist_ok=True)

    filtered_mask = _load_3d(filtered_mask_path)
    unfiltered_mask = _load_3d(unfiltered_mask_path)
    z_min = min(filtered_mask.shape[0], unfiltered_mask.shape[0])
    if filtered_mask.shape[0] != unfiltered_mask.shape[0]:
        print(f"  WARNING: Z mismatch ({filtered_mask.shape[0]} vs {unfiltered_mask.shape[0]}), truncating to {z_min}")
        filtered_mask = filtered_mask[:z_min]
        unfiltered_mask = unfiltered_mask[:z_min]

    combined = tifffile.imread(str(combined_path))
    if combined.ndim != 4 or combined.shape[1] != 4:
        raise ValueError(f"Expected (Z, 4, Y, X), got {combined.shape}")
    combined = combined[:z_min]

    cell_labels = np.unique(filtered_mask)
    cell_labels = cell_labels[cell_labels > 0]
    if cell_ids is not None:
        cell_labels = np.array(
            sorted(set(int(x) for x in cell_labels.tolist()) & cell_ids),
            dtype=cell_labels.dtype,
        )
        if cell_labels.size == 0:
            print("ERROR: no labels match --cell-id filter")
            return 1
    vol_z, vol_y, vol_x = filtered_mask.shape
    props = regionprops(filtered_mask.astype(np.int32))
    prop_map = {p.label: p for p in props}
    channel_names = ["642_Original", "488_Original", "560_Original", "Primary_Cell_Mask", "Mask_Boundary"]
    rows_padded: list[dict] = []
    rows_full_z: list[dict] = []

    for idx, lbl in enumerate(cell_labels):
        lbl = int(lbl)
        p = prop_map.get(lbl)
        if p is None:
            continue
        bbox_zmin, bbox_ymin, bbox_xmin, bbox_zmax, bbox_ymax, bbox_xmax = p.bbox
        z0, z1 = max(0, bbox_zmin - margin_z), min(vol_z, bbox_zmax + margin_z)
        y0, y1 = max(0, bbox_ymin - margin_xy), min(vol_y, bbox_ymax + margin_xy)
        x0, x1 = max(0, bbox_xmin - margin_xy), min(vol_x, bbox_xmax + margin_xy)

        slab_combined = combined[:, :, y0:y1, x0:x1]
        slab_unfiltered = unfiltered_mask[:, y0:y1, x0:x1]

        if run_padded:
            crop_combined_p = slab_combined[z0:z1, :, :, :]
            crop_unfiltered_p = slab_unfiltered[z0:z1, :, :]
            crop_5ch_p, stats_p = _five_channel_crop(crop_combined_p, crop_unfiltered_p, lbl)
            out_path = box_dir / f"cell_{lbl:04d}.tif"
            if out_path.exists() and not force:
                status_msg = "SKIP"
            else:
                _save_crop(out_path, crop_5ch_p, channel_names)
                nc = stats_p["neighbor_count"]
                status_msg = "isolated" if stats_p["is_isolated"] else "has_neighbors"
            print(
                f"  [{idx + 1}/{len(cell_labels)}] cell {lbl:4d}  padded "
                f"neighbours={stats_p['neighbor_count']}  ({status_msg})",
                flush=True,
            )
            rows_padded.append(
                {
                    "cell_id": lbl,
                    "z0": z0,
                    "z1": z1,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "margin_xy": margin_xy,
                    "margin_z": margin_z,
                    **stats_p,
                }
            )

        if run_full_z_pass:
            crop_5ch_f, stats_f = _five_channel_crop(slab_combined, slab_unfiltered, lbl)
            z0f, z1f = 0, vol_z
            out_f = full_z_dir / f"cell_{lbl:04d}.tif"
            if out_f.exists() and not force:
                fz_status = "SKIP"
            else:
                _save_crop(out_f, crop_5ch_f, channel_names)
                fz_status = "isolated" if stats_f["is_isolated"] else "has_neighbors"
            prefix = f"  [{idx + 1}/{len(cell_labels)}] cell {lbl:4d}  " if not run_padded else "      "
            print(
                f"{prefix}full_z neighbours={stats_f['neighbor_count']}  ({fz_status})",
                flush=True,
            )
            rows_full_z.append(
                {
                    "cell_id": lbl,
                    "z0": z0f,
                    "z1": z1f,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "margin_xy": margin_xy,
                    "margin_z": margin_z,
                    **stats_f,
                }
            )

    if run_padded and rows_padded:
        fieldnames = list(rows_padded[0].keys())
        with open(str(summary_path), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_padded)
        print(f"\nSummary CSV: {summary_path}  ({len(rows_padded)} cells)")
    elif run_padded:
        print("\nNo cells found (padded pass).")

    if run_full_z_pass and rows_full_z:
        fieldnames_f = list(rows_full_z[0].keys())
        with open(str(full_z_summary_path), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_f)
            writer.writeheader()
            writer.writerows(rows_full_z)
        print(f"Summary CSV: {full_z_summary_path}  ({len(rows_full_z)} cells)")
    elif run_full_z_pass:
        print("No cells found (full-z pass).")

    print("Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--margin-xy", type=int, default=MARGIN_XY_DEFAULT)
    ap.add_argument("--margin-z", type=int, default=MARGIN_Z_DEFAULT)
    ap.add_argument(
        "--also-full-z",
        action="store_true",
        help="Also write full-Z crops under cell_boxing_full_z/ (same XY margin as cell_boxing/)",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=f"Mask variant to crop (default: {DEFAULT_VARIANT})",
    )
    ap.add_argument(
        "--label-mask-name",
        type=str,
        default=None,
        help="Label TIFF basename under output_dir (default: variant mask). "
        "Use e.g. union_488_560_otsu_shape.tif after apply_qc_pass_to_label_mask.",
    )
    ap.add_argument(
        "--output-subdir-key",
        choices=OUTPUT_SUBDIR_KEYS,
        default="cell_box",
        help="Where to write crops: cell_box (default) or cell_box_filtered.",
    )
    ap.add_argument(
        "--cell-id",
        type=int,
        action="append",
        default=None,
        help="Only crop these cell ids (repeatable). Omit for all labels in mask.",
    )
    args = ap.parse_args()
    data_dir, output_dir = _resolve_dirs(args)
    cid_set = set(args.cell_id) if args.cell_id else None
    return run(
        data_dir,
        output_dir,
        margin_xy=args.margin_xy,
        margin_z=args.margin_z,
        force=args.force,
        also_full_z=args.also_full_z,
        variant=args.variant,
        label_mask_name=args.label_mask_name,
        output_subdir_key=args.output_subdir_key,
        cell_ids=cid_set,
    )


if __name__ == "__main__":
    sys.exit(main())
