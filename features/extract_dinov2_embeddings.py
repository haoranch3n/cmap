#!/usr/bin/env python3
"""Extract per-cell DINOv2 ViT-B/14 embeddings from cropped 3D cells.

Mirrors the CLI shape of ``features/extract_features.py`` so it slots cleanly
into ``pipelines/run_analysis_pipeline.py``.  Reads
``output/<sample>/<cell-boxing-dir>/cell_*.tif`` (default: ``cell_boxing``) and writes
``output/<sample>/<cell-qc-dir>/dinov2_embeddings.npy`` (default: ``cell_qc``) plus a matching
``dinov2_embeddings.csv``. Use ``--cell-boxing-dir cell_boxing_filtered`` and
``--cell-qc-dir cell_qc_filtered`` for filtered crops without touching the main QC tree.

Volume-wide normalization uses ``dinov2_volume_norm_bounds.csv`` in the
sample output directory (see ``features/compute_dinov2_volume_norm_csv.py``),
so the large combined TIFF is not reloaded for every extraction run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - fallback when run from segmentation/
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT  # type: ignore[no-redef]

from features.dinov2 import DinoV2Config, extract_sample
from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files


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


def _build_config(args) -> DinoV2Config:
    return DinoV2Config(
        model_name=args.model_name,
        target_size=args.target_size,
        norm_p_low=args.p_low,
        norm_p_high=args.p_high,
        use_imagenet_stats=args.imagenet_stats,
        apply_mask=args.apply_mask,
        empty_slice_nonzero_frac=args.empty_frac,
        batch_size=args.batch_size,
        device=args.device,
        save_slice_embeddings=args.save_slice_embeddings,
        extraction_mode=args.extraction_mode,
        norm_scope=args.norm_scope,
        centroid_core_quantile=args.centroid_core_q,
        centroid_use_intensity_weights=not args.centroid_no_intensity_weights,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")

    ap.add_argument("--model-name", type=str, default="dinov2_vitb14",
                    help="torch.hub model name (default: dinov2_vitb14, 768-D CLS token)")
    ap.add_argument("--target-size", type=int, default=224,
                    help="Spatial size each slice is resized to. Must be a multiple of 14.")
    ap.add_argument("--p-low", type=float, default=1.0,
                    help="Lower percentile for per-slice normalization (ignored when --norm-scope volume)")
    ap.add_argument("--p-high", type=float, default=99.0,
                    help="Upper percentile for per-slice normalization (ignored when --norm-scope volume)")
    ap.add_argument("--imagenet-stats", action="store_true",
                    help="Apply ImageNet mean/std after [0,1] scaling (off by default for microscopy)")
    ap.add_argument("--apply-mask", action="store_true",
                    help="Zero pixels outside Primary_Cell_Mask before normalization")
    ap.add_argument("--empty-frac", type=float, default=0.01,
                    help="Skip a z-slice if 642 nonzero fraction is below this (default: 0.01; 0 disables)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Slices encoded per forward pass")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Torch device (e.g. cuda, cuda:0, cpu)")
    ap.add_argument("--save-slice-embeddings", action="store_true",
                    help="Also dump per-slice embeddings to dinov2_embeddings_per_slice.npz")
    ap.add_argument(
        "--extraction-mode",
        choices=("z_mean", "orthogonal_concat"),
        default="z_mean",
        help="z_mean: mean-pool z-slice embeddings. orthogonal_concat: three planes through robust centroid.",
    )
    ap.add_argument(
        "--norm-scope",
        choices=("slice", "volume"),
        default="slice",
        help="slice: per-slice percentiles. volume: min + 99.99%% from dinov2_volume_norm_bounds.csv.",
    )
    ap.add_argument(
        "--volume-bounds-csv",
        type=Path,
        default=None,
        help="Override path to bounds CSV (default: <output_dir>/dinov2_volume_norm_bounds.csv)",
    )
    ap.add_argument(
        "--recompute-volume-norm",
        action="store_true",
        help="Load filtered_642_combined.tif once, write the bounds CSV, then extract.",
    )
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Process at most this many cell TIFFs (after sort; for smoke tests)")
    ap.add_argument(
        "--cell-id",
        type=int,
        action="append",
        default=None,
        help="Only process this cell id (repeatable). If set, --max-cells is ignored.",
    )
    ap.add_argument(
        "--centroid-core-q",
        type=float,
        default=0.75,
        help="Distance-transform core quantile for centroid (orthogonal mode).",
    )
    ap.add_argument(
        "--centroid-no-intensity-weights",
        action="store_true",
        help="Use unweighted COM on the DT core (orthogonal mode).",
    )
    ap.add_argument(
        "--cell-boxing-dir",
        type=str,
        default=None,
        help="Subfolder of output_dir with cell_*.tif (e.g. cell_boxing_filtered). "
             "Defaults to the cell_box dir for --variant.",
    )
    ap.add_argument(
        "--cell-qc-dir",
        type=str,
        default=None,
        help="Subfolder of output_dir for dinov2 outputs (e.g. cell_qc_filtered). "
             "Defaults to the cell_qc dir for --variant.",
    )
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            f"Mask variant whose cell_box / cell_qc dirs, combined TIFF and bounds CSV "
            f"to use (default: {DEFAULT_VARIANT}). Per-flag overrides "
            "(--cell-boxing-dir, --cell-qc-dir, --volume-bounds-csv) take precedence."
        ),
    )
    ap.add_argument("--verbose", action="store_true", help="Enable INFO logging")

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _, output_dir = _resolve_dirs(args)
    cfg = _build_config(args)
    only_ids = set(args.cell_id) if args.cell_id else None
    max_cells = None if only_ids else args.max_cells

    v = variant_files(args.variant)
    cell_boxing_dirname = args.cell_boxing_dir or v["cell_box"]
    cell_qc_dirname = args.cell_qc_dir or v["cell_qc"]
    combined_filename = v["combined"]
    bounds_csv_filename = v["dinov2_volume_bounds"]
    print(
        f"Variant: {args.variant}  cell_box={cell_boxing_dirname}  "
        f"cell_qc={cell_qc_dirname}  combined={combined_filename}  "
        f"bounds_csv={bounds_csv_filename}"
    )

    return extract_sample(
        output_dir,
        cfg,
        force=args.force,
        recompute_volume_norm=args.recompute_volume_norm,
        volume_bounds_csv=args.volume_bounds_csv,
        max_cells=max_cells,
        only_cell_ids=only_ids,
        cell_boxing_dirname=cell_boxing_dirname,
        cell_qc_dirname=cell_qc_dirname,
        combined_filename=combined_filename,
        bounds_csv_filename=bounds_csv_filename,
    )


if __name__ == "__main__":
    sys.exit(main())
