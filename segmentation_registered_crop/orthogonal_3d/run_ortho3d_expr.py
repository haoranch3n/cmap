#!/usr/bin/env python3
"""
Orthogonal-3D segmentation experiment (ortho-3d-expr branch).

Runs the XY + YZ orthogonal Cellpose pipeline on 5 representative registered
cell crops and writes results to output_registered_crop_seg_ortho3d/.

The XZ plane is intentionally skipped because the Z-step size (≈1 µm) is much
larger than the XY pixel size (≈0.1–0.2 µm), making XZ slices geometrically
skewed.  YZ slices (indexed along X) have the same Z-anisotropy but are
included as a second orthogonal constraint.

For each cell and each channel (ch642, ch488, ch560):
  1. extract_channel → normalised per-Z planes (XY tif_planes/)
  2. segment_xy_axis → (Z,Y,X) matched label volume from Z-slices
  3. segment_yz_axis → (Z,Y,X) matched label volume from X-slices (YZ)
  4. matching_cells_3D(mask_XY, mask_YZ, mask_YZ, minslices)  — YZ used twice
  5. Post-filtering (bridge, absorb, z-span, volume)
  6. run_selection → 5-channel output TIFF

Results are written to:
  <output_root>/<batch>/<Sample_Position>/<cell_id>/
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import tifffile

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _MODULE_ROOT.parent
for _p in (_MODULE_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline_config import (
    CROPS_ROOT,
    OUTPUT_ROOT,
    SEGMENTATION_CHANNELS,
    MIN_CELL_Z_SPAN,
)
from preprocessing.extract_channel import extract_channel
from cell_selector.select_main_cell import run_selection
from orthogonal_3d.ortho_segmentation import (
    segment_xy_axis,
    segment_yz_axis,
    apply_postfilters,
    _read_normalized_volume,
)
from orthogonal_3d.match_3d_cells import matching_cells_3D

# -------------------------------------------------------------------------
# 5 test cells for the experiment (batch, sample_position, cell_id)
# -------------------------------------------------------------------------
TEST_CELLS: list[tuple[str, str, str]] = [
    ("4_24_25_CGN_6_10_2", "Sample10_Position1", "cell_0005"),  # problem cell
    ("4_18_25",            "CGNSample1_Position0", "cell_0010"),
    ("4_18_25",            "CGNSample1_Position2", "cell_0013"),
    ("4_18_25",            "CGNSample1_Position4", "cell_0005"),
    ("4_24_25_CGN_6_10_2", "Sample10_Position1",   "cell_0010"),
]

OUTPUT_ROOT_ORTHO = REPO_ROOT = Path(__file__).resolve().parents[2] / "output_registered_crop_seg_ortho3d"


def process_one_cell_channel(
    registered_tif: Path,
    channel_name: str,
    channel_idx: int,
    output_root: Path,
    gpu: bool,
    force: bool,
) -> str:
    """Run the full XY+YZ pipeline for one cell × one channel.

    Returns "done", "skipped", or "error: <msg>".
    """
    crop_dir = registered_tif.parent
    cell_id = registered_tif.stem.replace("_registered", "")

    # Preserve batch/sample/cell directory hierarchy.
    rel_path = crop_dir.relative_to(crop_dir.parents[2])
    out_crop_dir = output_root / rel_path
    out_chan_dir = out_crop_dir / channel_name

    final_tif = out_crop_dir / f"{cell_id}_{channel_name}_segmented.tif"
    if not force and final_tif.exists() and final_tif.stat().st_size > 0:
        return "skipped"

    try:
        # Step 1: Extract channel (normalised per-Z planes).
        print(f"\n[{cell_id} | {channel_name}] Step 1: extract channel")
        extract_info = extract_channel(
            registered_tif=registered_tif,
            channel_idx=channel_idx,
            channel_name=channel_name,
            out_dir=out_chan_dir,
            skip_existing=(not force),
        )
        tif_planes_dir = Path(extract_info["tif_planes_dir"])
        z_size = extract_info["n_planes"]

        # Step 2: XY (Z-slice) segmentation + matching_cells_2D.
        print(f"[{cell_id} | {channel_name}] Step 2: XY segmentation")
        mask_XY = segment_xy_axis(
            tif_planes_dir=tif_planes_dir,
            out_dir=out_chan_dir,
            gpu=gpu,
            force=force,
        )
        if mask_XY is None:
            return f"error: no XY masks for {cell_id}/{channel_name}"

        # Step 3: YZ segmentation + matching_cells_2D.
        print(f"[{cell_id} | {channel_name}] Step 3: YZ segmentation")
        norm_vol = _read_normalized_volume(tif_planes_dir, cell_id, channel_name, z_size)
        mask_YZ = segment_yz_axis(
            tif_planes_dir=tif_planes_dir,
            norm_vol=norm_vol,
            cell_id=cell_id,
            channel_name=channel_name,
            out_dir=out_chan_dir,
            gpu=gpu,
            force=force,
        )
        if mask_YZ is None:
            print(f"  [warn] No YZ masks; falling back to XY-only for {cell_id}/{channel_name}")
            mask_YZ = mask_XY

        # Step 4: 3D matching (XY ∩ YZ; YZ passed for both XZ and YZ slots).
        print(f"[{cell_id} | {channel_name}] Step 4: matching_cells_3D (XY + YZ)")
        seg_3d = matching_cells_3D(
            mask_XY=mask_XY,
            mask_XZ=mask_YZ,   # YZ used for both — XZ skipped (anisotropic)
            mask_YZ=mask_YZ,
            minslices=MIN_CELL_Z_SPAN,
        )
        print(f"  Raw 3D cells: {int(seg_3d.max())}")

        # Step 5: Post-filtering.
        print(f"[{cell_id} | {channel_name}] Step 5: post-filtering")
        seg_3d = apply_postfilters(seg_3d.astype(np.int32))
        n_cells = int(seg_3d.max())
        print(f"  After filtering: {n_cells} cell(s)")

        # Write indexed 3D mask.
        seg3d_dir = out_chan_dir / "segmentation_3D_masks"
        seg3d_dir.mkdir(parents=True, exist_ok=True)
        indexed_tif = seg3d_dir / f"{cell_id}_{channel_name}_3D_indexed.tif"
        tmp = str(indexed_tif) + ".tmp"
        tifffile.imwrite(tmp, seg_3d.astype(np.uint16))
        os.replace(tmp, str(indexed_tif))

        # Step 6: Select main cell + write 5-channel output TIFF.
        print(f"[{cell_id} | {channel_name}] Step 6: select main cell + write output")
        run_selection(
            indexed_mask_tif=indexed_tif,
            registered_tif=registered_tif,
            channel_name=channel_name,
            out_dir=out_crop_dir,
            skip_existing=(not force),
        )
        return "done"

    except Exception:
        return f"error: {traceback.format_exc()}"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops-root",  type=Path, default=CROPS_ROOT)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_ORTHO)
    ap.add_argument("--channel", choices=list(SEGMENTATION_CHANNELS.keys()), default=None,
                    help="Process only this channel (default: all three).")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    ap.add_argument("--dry-run", action="store_true", help="Print cells to process, then exit.")
    args = ap.parse_args()

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # GPU detection.
    if args.no_gpu:
        gpu = False
    else:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
    print(f"GPU: {gpu}")
    print(f"Output root: {output_root}")

    channels_to_process = (
        {args.channel: SEGMENTATION_CHANNELS[args.channel]}
        if args.channel
        else SEGMENTATION_CHANNELS
    )

    # Resolve registered TIF paths for the 5 test cells.
    test_tifs: list[tuple[Path, str]] = []
    for batch, sample_pos, cell_id in TEST_CELLS:
        tif = args.crops_root / batch / sample_pos / cell_id / f"{cell_id}_registered.tif"
        if not tif.exists():
            print(f"  [warn] Missing: {tif}")
            continue
        test_tifs.append((tif, f"{batch}/{sample_pos}/{cell_id}"))

    if not test_tifs:
        print("No test TIFs found — check CROPS_ROOT and TEST_CELLS.")
        return 1

    print(f"\nTest cells ({len(test_tifs)}):")
    for tif, label in test_tifs:
        print(f"  {label}")

    if args.dry_run:
        return 0

    n_done = n_skip = n_err = 0
    errors: list[str] = []

    for registered_tif, label in test_tifs:
        for channel_name, channel_idx in channels_to_process.items():
            result = process_one_cell_channel(
                registered_tif=registered_tif,
                channel_name=channel_name,
                channel_idx=channel_idx,
                output_root=output_root,
                gpu=gpu,
                force=args.force,
            )
            if result == "done":
                n_done += 1
            elif result == "skipped":
                n_skip += 1
            else:
                n_err += 1
                msg = f"{label}/{channel_name}: {result}"
                errors.append(msg)
                print(f"  [ERROR] {msg}", file=sys.stderr)

    print(f"\nDone. processed={n_done} skipped={n_skip} errors={n_err}")
    if errors:
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
