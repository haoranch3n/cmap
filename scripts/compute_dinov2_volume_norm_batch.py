#!/usr/bin/env python3
"""Write ``dinov2_volume_norm_bounds.csv`` (per-channel min + 99.99%%ile) for every sample.

Discovers ``filtered_642_combined.tif`` under ``output/`` (each immediate subfolder
except ``cell_qc_all`` / ``tif_planes``), sorted by path. Always prints a
sanity block for the **first** image, then processes all samples (skips when
the CSV already exists unless ``--force``).

Example::

    python scripts/compute_dinov2_volume_norm_batch.py
    python scripts/compute_dinov2_volume_norm_batch.py --output-root /path/to/output --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tifffile  # noqa: E402

from features.dinov2.volume_norm_csv import (  # noqa: E402
    compute_bounds_from_combined_tif,
    default_bounds_csv_path,
    read_bounds_csv,
    write_bounds_csv,
)


def _discover_combined_paths(output_root: Path) -> list[Path]:
    output_root = Path(output_root)
    paths: list[Path] = []
    skip = {"cell_qc_all", "tif_planes"}
    if not output_root.is_dir():
        return paths
    for child in sorted(output_root.iterdir(), key=lambda p: p.name.lower()):
        if child.name in skip:
            continue
        if not child.is_dir() and not (child.is_symlink() and child.resolve().is_dir()):
            continue
        for p in child.rglob("filtered_642_combined.tif"):
            # Keep paths under output/ unresolved so symlinked batches stay
            # relative to output_root (resolve() would jump to other trees).
            paths.append(p)
    paths.sort(key=lambda x: str(x).lower())
    return paths


def _print_first_image_preview(combined: Path) -> None:
    print("\n" + "=" * 72)
    print("FIRST IMAGE (sanity check — same sort order as full batch)")
    print("=" * 72)
    print(f"combined_tif: {combined}")
    try:
        with tifffile.TiffFile(str(combined)) as tf:
            shp = tuple(tf.series[0].shape)
            dt = tf.series[0].dtype
            print(f"shape (Z,C,Y,X): {shp}   dtype: {dt}")
    except Exception as exc:
        print(f"(could not peek TIFF header: {exc})")

    out_csv = default_bounds_csv_path(combined.parent)
    if out_csv.is_file():
        lo, hi = read_bounds_csv(out_csv)
        print(f"existing CSV: {out_csv}")
        print("  (delete or use --force to recompute from TIFF)")
    else:
        lo, hi = None, None

    t0 = time.time()
    lo_n, hi_n = compute_bounds_from_combined_tif(combined, p_high=99.99)
    elapsed = time.time() - t0
    print(f"fresh compute from TIFF: {elapsed:.1f}s")
    ch_names = ("642", "488", "560")
    for i, name in enumerate(ch_names):
        print(
            f"  ch{i} ({name}):  min={float(lo_n[i]):.6g}   p99.99={float(hi_n[i]):.6g}   "
            f"span={float(hi_n[i] - lo_n[i]):.6g}"
        )
    if lo is not None:
        delta = np_max_abs_diff(lo, hi, lo_n, hi_n)
        if delta > 1e-3:
            print(
                f"  NOTE: differs from on-disk CSV by up to {delta:.6g} "
                "(subsampling / float noise); overwrite with --force."
            )
    print("=" * 72 + "\n")


def np_max_abs_diff(
    lo: "object", hi: "object", lo_n, hi_n
) -> float:
    import numpy as np

    return max(
        float(np.max(np.abs(lo.astype(np.float64) - lo_n.astype(np.float64)))),
        float(np.max(np.abs(hi.astype(np.float64) - hi_n.astype(np.float64)))),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "output",
        help="Pipeline output root (default: <repo>/output)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when dinov2_volume_norm_bounds.csv already exists",
    )
    ap.add_argument(
        "--preview-only",
        action="store_true",
        help="Only print the first-image block and exit (no writes)",
    )
    args = ap.parse_args()
    output_root = Path(args.output_root).resolve()

    paths = _discover_combined_paths(output_root)
    if not paths:
        print(f"No filtered_642_combined.tif found under {output_root}", file=sys.stderr)
        return 1

    _print_first_image_preview(paths[0])
    if args.preview_only:
        return 0

    n_ok, n_skip, n_err = 0, 0, 0
    t_all = time.time()
    for idx, combined in enumerate(paths, start=1):
        out_dir = combined.parent
        csv_path = default_bounds_csv_path(out_dir)
        try:
            rel = combined.parent.relative_to(output_root)
        except ValueError:
            rel = combined
        if csv_path.is_file() and not args.force:
            n_skip += 1
            print(f"[{idx}/{len(paths)}] SKIP (exists): {rel}")
            continue
        try:
            t0 = time.time()
            lo, hi = compute_bounds_from_combined_tif(combined, p_high=99.99)
            write_bounds_csv(csv_path, lo, hi, source_tif=str(combined))
            dt = time.time() - t0
            n_ok += 1
            print(
                f"[{idx}/{len(paths)}] OK {dt:.1f}s  {rel}  "
                f"lo=({lo[0]:.4g},{lo[1]:.4g},{lo[2]:.4g}) "
                f"hi=({hi[0]:.4g},{hi[1]:.4g},{hi[2]:.4g})"
            )
        except Exception as exc:
            n_err += 1
            print(f"[{idx}/{len(paths)}] ERROR {rel}: {exc}", file=sys.stderr)

    print(
        f"\nDone in {time.time() - t_all:.1f}s:  wrote={n_ok}  skipped={n_skip}  errors={n_err}  "
        f"total_discovered={len(paths)}"
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
