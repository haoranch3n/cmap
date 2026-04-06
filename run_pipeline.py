#!/usr/bin/env python3
"""
Run end-to-end: preprocess → 2D Cellpose → Z-stack masks → 3D label volume.

Usage (from project root):
  python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

from pipeline_config import SEGMENTATION_3D_DIR

ROOT = Path(__file__).resolve().parent

STEPS = [
    ["python3", "preprocessing/separate_channels.py"],
    ["python3", "multiscale_cellpose/segmentation_cellpose_2d.py"],
    ["python3", "cellcomposor/stack_2D_planes.py"],
    ["python3", "cellcomposor/create_3D_cells.py"],
]


def main() -> int:
    for cmd in STEPS:
        print(f"\n>>> {' '.join(cmd)}\n", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"Step failed: {cmd}", file=sys.stderr)
            return r.returncode
    print(f"\nPipeline finished. 3D masks: {SEGMENTATION_3D_DIR}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
