#!/usr/bin/env python3
"""
Top-level orchestration across sibling modules.

Segmentation remains raw input -> mask outputs.
Downstream steps are explicit, optional modules.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(step_name: str, rel_cmd: list[str]) -> int:
    print(f"\n>>> {step_name}: {' '.join(rel_cmd)}", flush=True)
    return subprocess.call(rel_cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-segmentation", action="store_true")
    ap.add_argument("--run-visualization", action="store_true")
    ap.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    passthrough = args.passthrough or []

    if not args.skip_segmentation:
        rc = _run("segmentation", [sys.executable, "segmentation/run_segmentation.py"])
        if rc != 0:
            return rc

    rc = _run("postprocess.filter", [sys.executable, "postprocess/filter_642_mask.py", *passthrough])
    if rc != 0:
        return rc

    rc = _run("postprocess.combine", [sys.executable, "postprocess/combine_with_mask.py", *passthrough])
    if rc != 0:
        return rc

    rc = _run("features.crop", [sys.executable, "features/crop_cells.py", *passthrough])
    if rc != 0:
        return rc

    rc = _run("features.extract", [sys.executable, "features/extract_features.py", *passthrough])
    if rc != 0:
        return rc

    rc = _run("qc.filter", [sys.executable, "qc/filter_by_intensity.py", *passthrough])
    if rc != 0:
        return rc

    if args.run_visualization:
        rc = _run(
            "visualization.tsne",
            [sys.executable, "visualization/tsne_visualize.py", *passthrough],
        )
        if rc != 0:
            return rc

    print("\nPipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

