#!/usr/bin/env python3
"""
Crop individual 3D cells from filtered_642_combined.tif with margin.
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


def run(data_dir: Path, output_dir: Path, margin_xy: int = MARGIN_XY_DEFAULT, margin_z: int = MARGIN_Z_DEFAULT, force: bool = False) -> int:
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Margin: XY={margin_xy}  Z={margin_z}")
    combined_path = output_dir / "filtered_642_combined.tif"
    filtered_mask_path = output_dir / "filtered_642.tif"
    for p, name in [(combined_path, "filtered_642_combined.tif"), (filtered_mask_path, "filtered_642.tif")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            return 1

    try:
        unfiltered_mask_path = _find_unfiltered_mask(output_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    box_dir = output_dir / "cell_boxing"
    summary_path = box_dir / "summary.csv"
    if summary_path.exists() and not force:
        print(f"SKIP (exists): {summary_path}  (use --force to overwrite)")
        return 0
    box_dir.mkdir(parents=True, exist_ok=True)

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
    vol_z, vol_y, vol_x = filtered_mask.shape
    props = regionprops(filtered_mask.astype(np.int32))
    prop_map = {p.label: p for p in props}
    channel_names = ["642_Original", "488_Original", "560_Original", "Primary_Cell_Mask", "Mask_Boundary"]
    rows: list[dict] = []

    for idx, lbl in enumerate(cell_labels):
        lbl = int(lbl)
        p = prop_map.get(lbl)
        if p is None:
            continue
        bbox_zmin, bbox_ymin, bbox_xmin, bbox_zmax, bbox_ymax, bbox_xmax = p.bbox
        z0, z1 = max(0, bbox_zmin - margin_z), min(vol_z, bbox_zmax + margin_z)
        y0, y1 = max(0, bbox_ymin - margin_xy), min(vol_y, bbox_ymax + margin_xy)
        x0, x1 = max(0, bbox_xmin - margin_xy), min(vol_x, bbox_xmax + margin_xy)

        crop_combined = combined[z0:z1, :, y0:y1, x0:x1]
        crop_unfiltered = unfiltered_mask[z0:z1, y0:y1, x0:x1]
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

        out_path = box_dir / f"cell_{lbl:04d}.tif"
        if out_path.exists() and not force:
            status_msg = "SKIP"
        else:
            _save_crop(out_path, crop_5ch, channel_names)
            status_msg = "isolated" if is_isolated else "has_neighbors"
        print(f"  [{idx + 1}/{len(cell_labels)}] cell {lbl:4d}  neighbours={neighbour_count}  ({status_msg})")

        rows.append(
            {
                "cell_id": lbl,
                "z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "margin_xy": margin_xy, "margin_z": margin_z,
                "primary_volume_voxels": primary_volume,
                "is_isolated": is_isolated,
                "neighbor_count": neighbour_count,
                "neighbor_voxels": neighbour_voxels,
                "neighbor_voxel_fraction": round(neighbour_frac, 4),
                "crop_shape_z": crop_5ch.shape[0],
                "crop_shape_y": crop_5ch.shape[2],
                "crop_shape_x": crop_5ch.shape[3],
            }
        )

    if rows:
        fieldnames = list(rows[0].keys())
        with open(str(summary_path), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV: {summary_path}  ({len(rows)} cells)")
    else:
        print("\nNo cells found.")
    print("Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--margin-xy", type=int, default=MARGIN_XY_DEFAULT)
    ap.add_argument("--margin-z", type=int, default=MARGIN_Z_DEFAULT)
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()
    data_dir, output_dir = _resolve_dirs(args)
    return run(data_dir, output_dir, margin_xy=args.margin_xy, margin_z=args.margin_z, force=args.force)


if __name__ == "__main__":
    sys.exit(main())

