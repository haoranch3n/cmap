#!/usr/bin/env python3
"""
Segmentation pipeline for registered cell crops.

For each registered cell crop and for each of the three fluorescence channels
(642, 488, 560), runs:
  1. Channel extraction + per-volume normalization
  2. Multiscale Cellpose 2D segmentation
  3. Stack 2D masks → 3D volume
  4. 3D cell assembly with internal quality filters
  5. Main-cell selection (centroid closest to volume centre)
  6. Assemble 5-channel output TIFF (642/488/560 original + mask + boundary)

Usage (on a GPU node):
  python run_pipeline.py [--crops-root PATH] [--output-root PATH] [--batch BATCH]
                         [--dry-run] [--force] [--no-gpu]

Environment variables:
  CROPS_ROOT           Override registered crops root directory.
  SEG_CROP_OUTPUT_ROOT Override output root directory.
  NORM_BOUNDS_ROOT     Override norm-bounds root (output_no_deconv tree).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

# Also ensure the parent repo root is on path so the base segmentation module
# (needed by create_3d_cells.py for match_2d_cells import) is accessible.
_REPO_ROOT = _MODULE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_config import CROPS_ROOT, OUTPUT_ROOT, SEGMENTATION_CHANNELS

from preprocessing.extract_channel import extract_channel
from multiscale_cellpose.segmentation_cellpose_2d import run_segmentation
from cellcomposor.stack_2d_planes import stack_volume
from cellcomposor.create_3d_cells import create_3d_cells
from cell_selector.select_main_cell import run_selection


def discover_registered_tifs(crops_root: Path, batch_filter: str | None = None) -> list[Path]:
    """Find all ``*_registered.tif`` files under crops_root, excluding flagged folders."""
    _EXCLUDE_DIRS = {"flagged_rotation_increase_gt5deg"}
    pattern = "**/*_registered.tif"
    tifs = sorted(crops_root.glob(pattern))
    tifs = [t for t in tifs if not _EXCLUDE_DIRS.intersection(t.parts)]
    if batch_filter:
        tifs = [t for t in tifs if batch_filter in t.parts]
    return tifs


def process_one_crop_channel(
    registered_tif: Path,
    channel_name: str,
    channel_idx: int,
    output_root: Path,
    gpu: bool,
    force: bool,
    skip_existing: bool,
) -> str:
    """
    Run the full segmentation pipeline for one crop × one channel.
    Returns "done", "skipped", or "error: <msg>".
    """
    # Output layout: output_root/<batch>/<Sample_Position>/<cell_id>/
    # The registered_tif lives at: crops_root/<batch>/<Sample_Position>/<cell_id>/<cell_id>_registered.tif
    crop_dir = registered_tif.parent
    cell_id = registered_tif.stem.replace("_registered", "")
    rel_path = crop_dir.relative_to(crop_dir.parents[2])  # <batch>/<Sample_Position>/<cell_id>
    out_crop_dir = output_root / rel_path
    out_chan_dir = out_crop_dir / channel_name

    # Quick skip: if the final 5-ch TIFF exists, skip the whole channel.
    final_tif = out_crop_dir / f"{cell_id}_{channel_name}_segmented.tif"
    if not force and skip_existing and final_tif.exists() and final_tif.stat().st_size > 0:
        return "skipped"

    try:
        # Step 1: Extract channel + write normalised planes + norm bounds CSV.
        print(f"\n[{cell_id} | {channel_name}] Step 1: extracting channel {channel_idx}")
        extract_info = extract_channel(
            registered_tif=registered_tif,
            channel_idx=channel_idx,
            channel_name=channel_name,
            out_dir=out_chan_dir,
            skip_existing=(not force) and skip_existing,
        )
        tif_planes_dir = Path(extract_info["tif_planes_dir"])

        # Step 2: Multiscale Cellpose 2D segmentation.
        seg2d_dir = out_chan_dir / "segmentation_2D_planes"
        diameters_dir = out_chan_dir / "segmentation_2D_diameters"
        print(f"[{cell_id} | {channel_name}] Step 2: Cellpose 2D (gpu={gpu})")
        run_segmentation(
            tif_planes_root=tif_planes_dir,
            seg_root=seg2d_dir,
            diameters_root=diameters_dir,
            gpu=gpu,
        )

        # Step 3: Stack 2D planes → single 3D TIFF.
        stacked_dir = out_chan_dir / "segmentation_2D_stack"
        print(f"[{cell_id} | {channel_name}] Step 3: stacking 2D masks")
        stacked_tif_path = stack_volume(
            seg2d_dir=seg2d_dir,
            stacked_dir=stacked_dir,
        )
        if stacked_tif_path is None:
            return f"error: no 2D masks to stack for {cell_id}/{channel_name}"

        # Step 4: 3D assembly + internal quality filters.
        seg3d_dir = out_chan_dir / "segmentation_3D_masks"
        print(f"[{cell_id} | {channel_name}] Step 4: 3D cell assembly")
        indexed_tif_path = create_3d_cells(
            stacked_tif=Path(stacked_tif_path),
            seg3d_dir=seg3d_dir,
            force=force,
        )
        if indexed_tif_path is None:
            return f"error: 3D assembly produced no output for {cell_id}/{channel_name}"

        # Step 5 + 6: Select main cell, assemble 5-channel output TIFF.
        print(f"[{cell_id} | {channel_name}] Step 5: selecting main cell + writing output")
        run_selection(
            indexed_mask_tif=Path(indexed_tif_path),
            registered_tif=registered_tif,
            channel_name=channel_name,
            out_dir=out_crop_dir,
            skip_existing=(not force) and skip_existing,
        )
        return "done"

    except Exception:
        return f"error: {traceback.format_exc()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops-root", type=Path, default=CROPS_ROOT)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument(
        "--batch",
        default=None,
        help="Process only crops under this batch folder name (e.g. 4_24_25_CGN_6_10_2).",
    )
    ap.add_argument(
        "--crop",
        default=None,
        help="Process only this specific cell_id (e.g. cell_0002).",
    )
    ap.add_argument(
        "--channel",
        choices=list(SEGMENTATION_CHANNELS.keys()),
        default=None,
        help="Process only this channel (default: all three).",
    )
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs already exist.")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU (ignore GPU availability).")
    ap.add_argument("--dry-run", action="store_true", help="Discover crops but do not process.")
    args = ap.parse_args()

    crops_root = args.crops_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Determine GPU availability.
    if args.no_gpu:
        gpu = False
    else:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
    print(f"GPU: {gpu}")

    # Discover registered TIFs.
    registered_tifs = discover_registered_tifs(crops_root, batch_filter=args.batch)
    if args.crop:
        registered_tifs = [t for t in registered_tifs if args.crop in t.parts]

    if not registered_tifs:
        print(f"No *_registered.tif found under {crops_root}")
        return 1
    print(f"Found {len(registered_tifs)} registered crop(s).")

    channels_to_process = (
        {args.channel: SEGMENTATION_CHANNELS[args.channel]}
        if args.channel
        else SEGMENTATION_CHANNELS
    )

    if args.dry_run:
        for tif in registered_tifs:
            print(f"  {tif}")
        return 0

    n_done = n_skip = n_err = 0
    errors: list[str] = []

    for registered_tif in registered_tifs:
        for channel_name, channel_idx in channels_to_process.items():
            result = process_one_crop_channel(
                registered_tif=registered_tif,
                channel_name=channel_name,
                channel_idx=channel_idx,
                output_root=output_root,
                gpu=gpu,
                force=args.force,
                skip_existing=True,
            )
            if result == "done":
                n_done += 1
            elif result == "skipped":
                n_skip += 1
            else:
                n_err += 1
                msg = f"{registered_tif.parent.name}/{channel_name}: {result}"
                errors.append(msg)
                print(f"  [ERROR] {msg}", file=sys.stderr)

    print(f"\nDone. processed={n_done} skipped={n_skip} errors={n_err}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
