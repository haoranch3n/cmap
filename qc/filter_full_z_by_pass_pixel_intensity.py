#!/usr/bin/env python3
"""
Copy per-cell TIFFs into a filtered folder using a QC pass column from
`cell_qc/qc_features_filtered.csv`.

Default behavior matches the current workflow: rows with
``pass_qc == 1`` are copied from ``cell_boxing_full_z/`` to
``cell_boxing_full_z_filtered/`` (legacy CSVs without ``pass_qc`` fall back to
``pass_pixel_intensity`` when the default column is used).

For mean-intensity QC (Otsu on per-cell means), use ``--pass-column
pass_intensity`` with ``--source-rel cell_boxing`` and ``--target-rel
cell_boxing_filtered``.

Use ``--pass-column pass_qc`` (or ``pass_intensity`` for mean-only QC) with
``cell_boxing`` → ``cell_boxing_filtered`` when filtered cell IDs must match
``cell_boxing_full_z_filtered`` (same QC column). Pair with ``--prune-target``
to drop stale TIFFs after changing the pass column.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files  # noqa: E402


def _cell_filename(cell_id_raw: str) -> str:
    cell_id = int(float(cell_id_raw))
    return f"cell_{cell_id:04d}.tif"


def run(
    output_dir: Path,
    csv_rel: str = "cell_qc/qc_features_filtered.csv",
    source_rel: str = "cell_boxing_full_z",
    target_rel: str = "cell_boxing_full_z_filtered",
    pass_column: str = "pass_qc",
    overwrite: bool = False,
    prune_target: bool = False,
) -> int:
    output_dir = output_dir.resolve()
    csv_path = output_dir / csv_rel
    src_dir = output_dir / source_rel
    dst_dir = output_dir / target_rel

    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    if not src_dir.is_dir():
        print(f"ERROR: Source directory not found: {src_dir}")
        return 1

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"ERROR: CSV has no rows: {csv_path}")
        return 1
    if "cell_id" not in rows[0]:
        print("ERROR: CSV missing required column: 'cell_id'")
        return 1
    eff_pass = pass_column
    if eff_pass not in rows[0] and eff_pass == "pass_qc" and "pass_pixel_intensity" in rows[0]:
        eff_pass = "pass_pixel_intensity"
    if eff_pass not in rows[0]:
        print(f"ERROR: CSV missing pass column '{pass_column}' (no legacy fallback)")
        return 1

    keep_names: list[str] = []
    for row in rows:
        try:
            keep = int(float(row[eff_pass])) == 1
        except (TypeError, ValueError):
            continue
        if keep:
            try:
                keep_names.append(_cell_filename(row["cell_id"]))
            except (TypeError, ValueError):
                continue

    keep_set = set(keep_names)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if prune_target:
        pruned = 0
        for p in list(dst_dir.glob("cell_*.tif")):
            if p.name not in keep_set:
                p.unlink()
                pruned += 1
        if pruned:
            print(f"Pruned stale TIFFs (not in keep set): {pruned}")

    if not keep_set:
        print(f"No cells with {eff_pass} == 1; nothing to copy.")
        return 0

    copied = 0
    missing = 0
    existing_skipped = 0

    for name in sorted(keep_set):
        src = src_dir / name
        dst = dst_dir / name
        if not src.is_file():
            missing += 1
            continue
        if dst.exists() and not overwrite:
            existing_skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"Sample output dir: {output_dir}")
    print(f"CSV rows: {len(rows)}")
    print(f"Cells passing {eff_pass}==1: {len(keep_set)}")
    print(f"Copied: {copied}")
    print(f"Already existed (skipped): {existing_skipped}")
    print(f"Missing source TIFFs: {missing}")
    print(f"Filtered TIFF folder: {dst_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=None, help="Path to one sample output directory")
    ap.add_argument(
        "--root-dir",
        type=Path,
        action="append",
        default=[],
        help="Batch mode: root containing many sample folders (can pass multiple times)",
    )
    ap.add_argument("--csv-rel", type=str, default=None)
    ap.add_argument("--source-rel", type=str, default=None)
    ap.add_argument("--target-rel", type=str, default=None)
    ap.add_argument(
        "--pass-column",
        type=str,
        default="pass_qc",
        help="CSV column with 0/1 pass flag (e.g. pass_qc, pass_intensity; legacy pass_pixel_intensity)",
    )
    ap.add_argument(
        "--prune-target",
        action="store_true",
        help="Remove cell_*.tif in target that are not in the keep set (e.g. after switching pass column)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files already in target directory")
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            f"Mask variant; sets --csv-rel, --source-rel, --target-rel from "
            f"VARIANT_FILES (default: {DEFAULT_VARIANT}). Per-flag overrides win."
        ),
    )
    args = ap.parse_args()

    if args.output_dir is None and not args.root_dir:
        raise SystemExit("Provide --output-dir for one sample, or --root-dir for batch mode.")

    v = variant_files(args.variant)
    if args.csv_rel is None:
        args.csv_rel = f"{v['cell_qc']}/qc_features_filtered.csv"
    if args.source_rel is None:
        args.source_rel = v["cell_box_full_z"]
    if args.target_rel is None:
        args.target_rel = v["cell_box_full_z_filtered"]
    print(
        f"Variant: {args.variant}  csv={args.csv_rel}  src={args.source_rel}  dst={args.target_rel}"
    )

    if args.output_dir is not None:
        return run(
            output_dir=args.output_dir,
            csv_rel=args.csv_rel,
            source_rel=args.source_rel,
            target_rel=args.target_rel,
            pass_column=args.pass_column,
            overwrite=args.overwrite,
            prune_target=args.prune_target,
        )

    total = 0
    ok = 0
    failed = 0
    for root in args.root_dir:
        root = root.resolve()
        root_total = 0
        root_ok = 0
        root_failed = 0
        csv_paths = sorted(root.glob(f"*/{args.csv_rel}"))
        print(f"\nRoot: {root}")
        if not csv_paths:
            print(f"  No matches for */{args.csv_rel}")
            continue
        for csv_path in csv_paths:
            sample_output_dir = csv_path.parents[1]
            print(f"\n==> Processing {sample_output_dir}")
            rc = run(
                output_dir=sample_output_dir,
                csv_rel=args.csv_rel,
                source_rel=args.source_rel,
                target_rel=args.target_rel,
                pass_column=args.pass_column,
                overwrite=args.overwrite,
                prune_target=args.prune_target,
            )
            total += 1
            root_total += 1
            if rc == 0:
                ok += 1
                root_ok += 1
            else:
                failed += 1
                root_failed += 1
        print(f"\nRoot summary: processed={root_total} ok={root_ok} failed={root_failed}")

    print(f"\nBatch summary: processed={total} ok={ok} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
