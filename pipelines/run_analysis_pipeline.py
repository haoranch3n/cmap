#!/usr/bin/env python3
"""
Top-level orchestration across sibling modules.

Segmentation remains raw input -> mask outputs.
Downstream steps are explicit, optional modules.

Passthrough flags (``--passthrough`` and everything after it) are forwarded to
postprocess, ``qc/filter_by_intensity.py``, ``qc/apply_qc_pass_to_label_mask.py``,
``features/crop_cells.py``, ``features/extract_features.py``, ``qc/merge_qc_features_filtered.py``,
and (when enabled) DINOv2 steps. Do **not** put ``--also-full-z`` in passthrough
(postprocess CLIs do not accept it). Use the orchestrator flag instead::

    python pipelines/run_analysis_pipeline.py --also-full-z --passthrough --data-rel <rel>

To also write ``filtered_560.tif`` / ``filtered_488.tif`` (triple-overlap, same
rule as ``filtered_642.tif``), pass ``--export-all-anchors`` inside passthrough
to ``postprocess/filter_642_mask.py``::

    python pipelines/run_analysis_pipeline.py --passthrough --data-rel <rel> --export-all-anchors

That appends ``--also-full-z`` only to ``features/crop_cells.py``, writing
``cell_boxing_full_z/`` under the same sample output directory as ``cell_boxing/``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from postprocess import VALID_VARIANTS, variant_files  # noqa: E402

VARIANT_CHOICES = (*VALID_VARIANTS, "both")


def _run(step_name: str, rel_cmd: list[str]) -> int:
    print(f"\n>>> {step_name}: {' '.join(rel_cmd)}", flush=True)
    return subprocess.call(rel_cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-segmentation", action="store_true")
    ap.add_argument("--run-visualization", action="store_true")
    ap.add_argument(
        "--run-dinov2",
        action="store_true",
        help="Also run features/extract_dinov2_embeddings.py after qc.filter",
    )
    ap.add_argument(
        "--run-dinov2-vis",
        action="store_true",
        help="After all per-sample steps, run visualization/dinov2_visualize.py "
             "to pool embeddings into cell_qc_all/ (implies --run-dinov2)",
    )
    ap.add_argument(
        "--dinov2-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra flags forwarded to extract_dinov2_embeddings.py (e.g. --apply-mask)",
    )
    ap.add_argument(
        "--also-full-z",
        action="store_true",
        help="Forward --also-full-z only to features/crop_cells.py (full-Z cell_boxing_full_z/).",
    )
    ap.add_argument(
        "--mask-variant",
        choices=VARIANT_CHOICES,
        default="filtered_642",
        help=(
            "Run the downstream chain for one mask variant or both. "
            "filtered_642 (default): unchanged. union_488_560: replaces "
            "postprocess.filter+combine with postprocess.union and threads "
            "--variant union_488_560 into all downstream stages. both: runs "
            "filtered_642 then union_488_560 (segmentation runs once)."
        ),
    )
    ap.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        default=[],
        help="Args after this flag go to postprocess, QC, crop_cells, extract_features, merge, "
        "and DINOv2 steps (e.g. --data-rel <rel>). Use --also-full-z separately for full-Z crops.",
    )
    args = ap.parse_args()

    passthrough = args.passthrough or []
    crop_passthrough = list(passthrough)
    if args.also_full_z:
        crop_passthrough.append("--also-full-z")
    dinov2_extra = args.dinov2_extra or []
    run_dinov2 = args.run_dinov2 or args.run_dinov2_vis

    variants_to_run = (
        list(VALID_VARIANTS)
        if args.mask_variant == "both"
        else [args.mask_variant]
    )

    if not args.skip_segmentation:
        rc = _run("segmentation", [sys.executable, "segmentation/run_segmentation.py"])
        if rc != 0:
            return rc

    for variant in variants_to_run:
        print(f"\n=== Variant chain: {variant} ===", flush=True)
        if variant == "filtered_642":
            rc = _run(
                "postprocess.filter",
                [sys.executable, "postprocess/filter_642_mask.py", *passthrough],
            )
            if rc != 0:
                return rc
            rc = _run(
                "postprocess.combine",
                [sys.executable, "postprocess/combine_with_mask.py", *passthrough],
            )
            if rc != 0:
                return rc
        else:  # union_488_560
            rc = _run(
                "postprocess.union",
                [sys.executable, "postprocess/union_488_560_mask.py", *passthrough],
            )
            if rc != 0:
                return rc
            # union_488_560_mask.py writes the combined TIFF itself; no separate combine step.

        variant_flag = ["--variant", variant]
        v = variant_files(variant)
        pass_mask_name = v["pass_otsu_shape"]

        rc = _run(
            "qc.filter_by_intensity_vol",
            [
                sys.executable,
                "qc/filter_by_intensity.py",
                *variant_flag,
                "--from-volume",
                *passthrough,
            ],
        )
        if rc != 0:
            return rc

        rc = _run(
            "qc.apply_qc_pass",
            [
                sys.executable,
                "qc/apply_qc_pass_to_label_mask.py",
                *variant_flag,
                *passthrough,
            ],
        )
        if rc != 0:
            return rc

        rc = _run(
            "features.crop",
            [
                sys.executable,
                "features/crop_cells.py",
                *variant_flag,
                "--label-mask-name",
                pass_mask_name,
                "--output-subdir-key",
                "cell_box_filtered",
                *crop_passthrough,
            ],
        )
        if rc != 0:
            return rc

        rc = _run(
            "features.extract",
            [
                sys.executable,
                "features/extract_features.py",
                *variant_flag,
                "--cell-box-subdir-key",
                "cell_box_filtered",
                *passthrough,
            ],
        )
        if rc != 0:
            return rc

        rc = _run(
            "qc.merge_qc_features_filtered",
            [
                sys.executable,
                "qc/merge_qc_features_filtered.py",
                *variant_flag,
                *passthrough,
            ],
        )
        if rc != 0:
            return rc

        if run_dinov2:
            rc = _run(
                "features.dinov2_volume_norm",
                [
                    sys.executable,
                    "features/compute_dinov2_volume_norm_csv.py",
                    *variant_flag,
                    *passthrough,
                ],
            )
            if rc != 0:
                return rc
            rc = _run(
                "features.dinov2",
                [
                    sys.executable,
                    "features/extract_dinov2_embeddings.py",
                    *variant_flag,
                    "--cell-boxing-dir",
                    v["cell_box_filtered"],
                    *passthrough,
                    *dinov2_extra,
                ],
            )
            if rc != 0:
                return rc

        if args.run_visualization:
            rc = _run(
                "visualization.tsne",
                [sys.executable, "visualization/tsne_visualize.py", *passthrough],
            )
            if rc != 0:
                return rc

        if args.run_dinov2_vis:
            rc = _run(
                "visualization.dinov2",
                [sys.executable, "visualization/dinov2_visualize.py", *variant_flag],
            )
            if rc != 0:
                return rc

    print("\nPipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

