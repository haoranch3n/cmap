#!/usr/bin/env python3
"""
Hoist redundant <volume_stem>/ inside each pipeline stage folder (legacy layout).

For each volume root (directory containing tif_planes/), if stage/<stem>/ exists
and stem matches the volume folder name (or the single subfolder under tif_planes/),
move children up to stage/.

Optionally rename segmentation_2D_stacked -> segmentation_2D_stack.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "tif_planes",
    "segmentation_2D_diameters",
    "segmentation_2D_planes",
    "segmentation_2D_stack",
    "segmentation_2D_stacked",
    "segmentation_3D_masks",
)


def volume_stem(volume_root: Path) -> str | None:
    tp = volume_root / "tif_planes"
    if not tp.is_dir():
        return None
    if (tp / volume_root.name).is_dir():
        return volume_root.name
    subs = [p for p in tp.iterdir() if p.is_dir()]
    if len(subs) == 1:
        return subs[0].name
    return None


def hoist_stage(volume_root: Path, stem: str, dry_run: bool, verbose: bool) -> int:
    n = 0
    for st in STAGES:
        stage = volume_root / st
        inner = stage / stem
        if not inner.is_dir():
            continue
        dest = stage
        for child in sorted(inner.iterdir()):
            target = dest / child.name
            if target.exists():
                print(f"SKIP (exists): {target}", file=sys.stderr)
                continue
            if dry_run:
                print(f"would move {child} -> {target}")
            else:
                if verbose:
                    print(f"move {child} -> {target}")
                shutil.move(str(child), str(target))
            n += 1
        if not dry_run:
            try:
                inner.rmdir()
            except OSError:
                print(f"WARN: not empty, not removing: {inner}", file=sys.stderr)
        else:
            print(f"would rmdir {inner}")
    return n


def rename_stacked_to_stack(volume_root: Path, dry_run: bool) -> None:
    old = volume_root / "segmentation_2D_stacked"
    new = volume_root / "segmentation_2D_stack"
    if not old.is_dir() or new.exists():
        return
    if dry_run:
        print(f"would rename {old} -> {new}")
    else:
        old.rename(new)


def process_output_root(output: Path, dry_run: bool, verbose: bool) -> int:
    total = 0
    for tif_planes in sorted(output.rglob("tif_planes")):
        if not tif_planes.is_dir():
            continue
        volume_root = tif_planes.parent
        stem = volume_stem(volume_root)
        if not stem:
            continue
        print(f"Volume root {volume_root}  stem={stem}")
        total += hoist_stage(volume_root, stem, dry_run, verbose)
        rename_stacked_to_stack(volume_root, dry_run)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Pipeline output directory (default: project output/)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print every move (noisy)")
    args = ap.parse_args()
    out = args.output_root.resolve()
    if not out.is_dir():
        print(f"Not a directory: {out}", file=sys.stderr)
        return 1
    n = process_output_root(out, args.dry_run, args.verbose)
    print(f"Done. Moved {n} item(s)." + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
