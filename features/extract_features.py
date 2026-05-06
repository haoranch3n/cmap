#!/usr/bin/env python3
"""
Extract per-cell QC features from cropped cell TIFs.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files

CELL_BOX_SUBDIR_KEYS = ("cell_box", "cell_box_filtered")

CHANNEL_NAMES = ["642", "488", "560"]
CHANNEL_PAIRS = [(0, 1), (0, 2), (1, 2)]


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
    elif args.output_dir:
        output_dir = args.output_dir.resolve()
        data_dir = output_dir
    else:
        raise SystemExit("Provide either --data-rel or --output-dir or both --data-dir and --output-dir")
    return data_dir, output_dir


def _extract_cell_id(filename: str) -> int | None:
    m = re.match(r"cell_(\d+)", filename)
    return int(m.group(1)) if m else None


def extract_features(crop: np.ndarray) -> dict[str, float]:
    mask = crop[:, 3, :, :] > 0
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        return {}
    feats: dict[str, float] = {"volume": float(n_voxels)}
    channel_means: list[float] = []
    masked_values: list[np.ndarray] = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        vals = crop[:, ch_idx, :, :][mask].astype(np.float64)
        masked_values.append(vals)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        channel_means.append(mean_val)
        feats[f"mean_{ch_name}"] = mean_val
        feats[f"median_{ch_name}"] = float(np.median(vals))
        feats[f"std_{ch_name}"] = std_val
        feats[f"total_{ch_name}"] = float(np.sum(vals))
        feats[f"cv_{ch_name}"] = std_val / mean_val if mean_val > 0 else 0.0
        feats[f"pct95_{ch_name}"] = float(np.percentile(vals, 95))

    corrs: list[float] = []
    for i, j in CHANNEL_PAIRS:
        ci, cj = CHANNEL_NAMES[i], CHANNEL_NAMES[j]
        if len(masked_values[i]) < 2:
            r = 0.0
        else:
            r, _ = pearsonr(masked_values[i], masked_values[j])
            if np.isnan(r):
                r = 0.0
        feats[f"corr_{ci}_{cj}"] = r
        corrs.append(r)
    feats["min_channel_mean"] = float(min(channel_means))
    feats["mean_pairwise_corr"] = float(np.mean(corrs))
    return feats


def _box_dirname(v: dict[str, str], cell_box_subdir_key: str) -> str:
    if cell_box_subdir_key == "cell_box":
        return v["cell_box"]
    if cell_box_subdir_key == "cell_box_filtered":
        return v["cell_box_filtered"]
    raise ValueError(f"cell_box_subdir_key must be one of {CELL_BOX_SUBDIR_KEYS}")


def run(
    output_dir: Path,
    force: bool = False,
    variant: str = DEFAULT_VARIANT,
    cell_box_subdir_key: str = "cell_box",
    cell_ids: set[int] | None = None,
) -> int:
    v = variant_files(variant)
    box_rel = _box_dirname(v, cell_box_subdir_key)
    print(f"Variant: {variant}  (cell_box={box_rel}, cell_qc={v['cell_qc']})")
    box_dir = output_dir / box_rel
    qc_dir = output_dir / v["cell_qc"]
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_csv = qc_dir / "qc_features.csv"
    if out_csv.exists() and not force:
        print(f"SKIP (exists): {out_csv}  (use --force to overwrite)")
        return 0
    if not box_dir.is_dir():
        print(f"ERROR: cell_boxing directory not found: {box_dir}")
        return 1
    cell_tifs = sorted(box_dir.glob("cell_*.tif"))
    if not cell_tifs:
        print(f"ERROR: no cell_*.tif files in {box_dir}")
        return 1

    rows: list[dict] = []
    for idx, tif_path in enumerate(cell_tifs):
        cell_id = _extract_cell_id(tif_path.name)
        if cell_id is None:
            continue
        if cell_ids is not None and cell_id not in cell_ids:
            continue
        crop = tifffile.imread(str(tif_path))
        if crop.ndim != 4 or crop.shape[1] not in (4, 5):
            continue
        feats = extract_features(crop)
        if not feats:
            continue
        feats["cell_id"] = cell_id
        rows.append(feats)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(cell_tifs):
            print(f"  [{idx + 1}/{len(cell_tifs)}] processed")

    if not rows:
        print("No cells processed.")
        return 1
    fieldnames = ["cell_id"] + [k for k in rows[0] if k != "cell_id"]
    with open(str(out_csv), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}  ({len(rows)} cells, {len(fieldnames)} columns)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=f"Mask variant whose cell_box/cell_qc dirs to use (default: {DEFAULT_VARIANT})",
    )
    ap.add_argument(
        "--cell-box-subdir-key",
        choices=CELL_BOX_SUBDIR_KEYS,
        default="cell_box",
        help="Subfolder with cell_*.tif (default cell_box; use cell_box_filtered for QC-pruned crops).",
    )
    ap.add_argument(
        "--cell-id",
        type=int,
        action="append",
        default=None,
        help="Only process these cell ids (repeatable). Omit for all TIFFs in folder.",
    )
    args = ap.parse_args()
    _, output_dir = _resolve_dirs(args)
    cid_set = set(args.cell_id) if args.cell_id else None
    return run(
        output_dir,
        force=args.force,
        variant=args.variant,
        cell_box_subdir_key=args.cell_box_subdir_key,
        cell_ids=cid_set,
    )


if __name__ == "__main__":
    sys.exit(main())

