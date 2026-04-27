#!/usr/bin/env python3
"""
Run segmentation end-to-end: preprocess -> 2D Cellpose -> stack -> 3D labels.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from config import SEGMENTATION_3D_DIR
except ModuleNotFoundError:
    from segmentation.config import SEGMENTATION_3D_DIR

ROOT = Path(__file__).resolve().parent

STEPS = [
    [sys.executable, "preprocess/separate_channels.py"],
    [sys.executable, "cellpose2d/segmentation_cellpose_2d.py"],
    [sys.executable, "assemble3d/stack_2d_planes.py"],
    [sys.executable, "assemble3d/create_3d_cells.py"],
]


def main() -> int:
    for cmd in STEPS:
        print(f"\n>>> {' '.join(cmd)}\n", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            print(f"Step failed: {cmd}", file=sys.stderr)
            return rc
    print(f"\nSegmentation finished. 3D masks: {SEGMENTATION_3D_DIR}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

