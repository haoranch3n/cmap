#!/usr/bin/env python3
"""
Run ``features.crop_cells.run(..., also_full_z=True)`` for every sample directory
that contains ``filtered_642_combined.tif`` under a given output root (default:
``OUTPUT_DIR`` / ``PIPELINE_OUTPUT_DIR`` from ``segmentation.config``).

Each sample is the parent directory of that file (same layout as a single
``--data-dir`` / ``--output-dir`` pair for ``crop_cells.py``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.crop_cells import run  # noqa: E402

try:
    from segmentation.config import OUTPUT_DIR
except ModuleNotFoundError:
    from config import OUTPUT_DIR  # type: ignore


def _sample_dirs(output_root: Path) -> list[Path]:
    root = output_root.resolve()
    if not root.is_dir():
        return []
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in sorted(root.rglob("filtered_642_combined.tif")):
        d = p.parent.resolve()
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"Root to scan (default: {OUTPUT_DIR})",
    )
    ap.add_argument("--force", action="store_true", help="Pass --force to each crop_cells run")
    ap.add_argument("--dry-run", action="store_true", help="List samples only")
    ap.add_argument(
        "--write-list",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write one absolute sample directory per line (for Slurm array) and exit",
    )
    ap.add_argument("--margin-xy", type=int, default=None)
    ap.add_argument("--margin-z", type=int, default=None)
    args = ap.parse_args()

    root = (args.output_root or OUTPUT_DIR).resolve()
    samples = _sample_dirs(root)
    if not samples:
        print(f"No filtered_642_combined.tif found under {root}", file=sys.stderr)
        return 1

    if args.write_list is not None:
        args.write_list.parent.mkdir(parents=True, exist_ok=True)
        lines = "\n".join(str(p.resolve()) for p in samples) + "\n"
        args.write_list.write_text(lines)
        print(f"Wrote {len(samples)} paths to {args.write_list.resolve()}", flush=True)
        return 0

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    print(f"Scan root: {root}\nFound {len(samples)} sample(s).\n", flush=True)
    if args.dry_run:
        for d in samples:
            print(d)
        return 0

    from features.crop_cells import MARGIN_XY_DEFAULT, MARGIN_Z_DEFAULT

    margin_xy = args.margin_xy if args.margin_xy is not None else MARGIN_XY_DEFAULT
    margin_z = args.margin_z if args.margin_z is not None else MARGIN_Z_DEFAULT

    failed = 0
    for i, out_dir in enumerate(samples, 1):
        print(f"\n========== [{i}/{len(samples)}] {out_dir} ==========\n", flush=True)
        rc = run(
            out_dir,
            out_dir,
            margin_xy=margin_xy,
            margin_z=margin_z,
            force=args.force,
            also_full_z=True,
        )
        if rc != 0:
            failed += 1
    if failed:
        print(f"\nCompleted with {failed} failure(s).", flush=True)
        return 1
    print("\nAll samples completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
