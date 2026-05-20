#!/usr/bin/env python3
"""
Combine per-channel segmentation masks into two merged outputs per cell:

  <cell_id>_combined_union.tif         — union  (any 1 of 3 channels votes)
  <cell_id>_combined_majority_vote.tif — majority vote (≥ 2 of 3 channels)

Each output TIFF has shape (Z, 5, Y, X) float32:
  ch0  642nm intensity   (from ch642_segmented source)
  ch1  488nm intensity
  ch2  560nm intensity
  ch3  combined binary mask  (0=background, 1=nucleus)
  ch4  combined boundary map (thick, connectivity=1)

Usage:
    python combine_channel_masks.py [--output-root PATH] [--force] [--dry-run]
                                    [--batch BATCH] [--crop CELL_ID]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

OUTPUT_ROOT_DEFAULT = Path(
    "/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap"
    "/output_registered_crop_seg"
)


def compute_2d_boundaries(mask_3d: np.ndarray) -> np.ndarray:
    """
    Compute thick 2D boundary per Z-slice.

    Runs find_boundaries independently on each Z-plane so the boundary shows
    the cell outline in every slice, rather than the surface of the 3D volume.
    """
    boundary = np.zeros_like(mask_3d, dtype=np.uint8)
    for z in range(mask_3d.shape[0]):
        if mask_3d[z].any():
            boundary[z] = find_boundaries(
                mask_3d[z].astype(bool), mode="thick", connectivity=1
            ).astype(np.uint8)
    return boundary


def combine_one_cell(cell_dir: Path, force: bool = False) -> str:
    """
    Combine the three per-channel masks for one cell directory.
    Returns "done", "skipped", or "error: <msg>" or "missing".
    """
    cell_id = cell_dir.name

    ch642_tif = cell_dir / f"{cell_id}_ch642_segmented.tif"
    ch488_tif = cell_dir / f"{cell_id}_ch488_segmented.tif"
    ch560_tif = cell_dir / f"{cell_id}_ch560_segmented.tif"

    missing = [t.name for t in [ch642_tif, ch488_tif, ch560_tif] if not t.exists()]
    if missing:
        return f"missing: {missing}"

    out_union = cell_dir / f"{cell_id}_combined_union.tif"
    out_majority = cell_dir / f"{cell_id}_combined_majority_vote.tif"

    if not force and out_union.exists() and out_majority.exists():
        return "skipped"

    try:
        arr642 = tifffile.imread(str(ch642_tif))  # (Z, 5, Y, X) float32
        arr488 = tifffile.imread(str(ch488_tif))
        arr560 = tifffile.imread(str(ch560_tif))

        # Sanity check shapes match
        if not (arr642.shape == arr488.shape == arr560.shape):
            return f"error: shape mismatch {arr642.shape} {arr488.shape} {arr560.shape}"
        if arr642.ndim != 4 or arr642.shape[1] != 5:
            return f"error: unexpected shape {arr642.shape}, expected (Z,5,Y,X)"

        # Intensity channels are identical across all three TIFs (same registered source).
        # Take from ch642 as canonical.
        intensities = arr642[:, :3, :, :].copy()  # (Z, 3, Y, X)

        # Binary masks from channel index 3
        mask642 = (arr642[:, 3, :, :] > 0).astype(np.uint8)
        mask488 = (arr488[:, 3, :, :] > 0).astype(np.uint8)
        mask560 = (arr560[:, 3, :, :] > 0).astype(np.uint8)

        vote_sum = mask642 + mask488 + mask560  # 0–3 per voxel

        union_mask    = (vote_sum >= 1).astype(np.uint8)
        majority_mask = (vote_sum >= 2).astype(np.uint8)

        z, _, h, w = arr642.shape

        for combined_mask, out_path in [
            (union_mask,    out_union),
            (majority_mask, out_majority),
        ]:
            boundary = compute_2d_boundaries(combined_mask)

            out_arr = np.zeros((z, 5, h, w), dtype=np.float32)
            out_arr[:, :3] = intensities
            out_arr[:, 3]  = combined_mask.astype(np.float32)
            out_arr[:, 4]  = boundary.astype(np.float32)

            tmp = str(out_path) + ".tmp"
            tifffile.imwrite(tmp, out_arr, imagej=True, metadata={"axes": "ZCYX"})
            os.replace(tmp, str(out_path))

        return "done"

    except Exception as exc:
        import traceback
        return f"error: {traceback.format_exc()}"


def discover_cell_dirs(output_root: Path, batch: str | None, crop: str | None) -> list[Path]:
    """Find all leaf cell directories (containing *_segmented.tif files)."""
    cell_dirs = sorted({
        p.parent
        for p in output_root.glob("**/*_ch642_segmented.tif")
    })
    if batch:
        cell_dirs = [d for d in cell_dirs if batch in d.parts]
    if crop:
        cell_dirs = [d for d in cell_dirs if d.name == crop]
    return cell_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    ap.add_argument("--batch",  default=None, help="Process only this batch folder (e.g. 4_18_25).")
    ap.add_argument("--crop",   default=None, help="Process only this cell_id (e.g. cell_0010).")
    ap.add_argument("--force",   action="store_true", help="Overwrite existing combined TIFs.")
    ap.add_argument("--dry-run", action="store_true", help="List cells but do not process.")
    args = ap.parse_args()

    cell_dirs = discover_cell_dirs(args.output_root, args.batch, args.crop)
    if not cell_dirs:
        print(f"No cell directories found under {args.output_root}", file=sys.stderr)
        return 1

    print(f"Found {len(cell_dirs)} cell dir(s) to process.", flush=True)

    if args.dry_run:
        for d in cell_dirs:
            print(f"  {d}")
        return 0

    n_done = n_skip = n_err = 0
    errors: list[str] = []

    for i, cell_dir in enumerate(cell_dirs, 1):
        result = combine_one_cell(cell_dir, force=args.force)
        if result == "done":
            n_done += 1
        elif result == "skipped":
            n_skip += 1
        else:
            n_err += 1
            msg = f"{cell_dir.parent.parent.name}/{cell_dir.parent.name}/{cell_dir.name}: {result}"
            errors.append(msg)
            print(f"  [ERROR] {msg}", file=sys.stderr, flush=True)

        if i % 25 == 0:
            print(f"  {i}/{len(cell_dirs)} done={n_done} skip={n_skip} err={n_err}", flush=True)

    print(f"\nDone. processed={n_done} skipped={n_skip} errors={n_err}", flush=True)
    if errors:
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
