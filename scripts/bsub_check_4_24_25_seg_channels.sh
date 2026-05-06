#!/usr/bin/env bash
# LSF: quick scan of 4_24_25 (or any) dataset — 642/488/560 indexed mask max labels
# and presence of *_3D_indexed_filtered.tif per channel. Runs on a compute node.
#
# No LSF email: do not use #BSUB -B / #BSUB -N (job begin/end mail).
#
# Submit from repo root:
#   bsub < scripts/bsub_check_4_24_25_seg_channels.sh
#
# Optional:
#   CMAP_CHECK_BASE  — root containing sample dirs (default: 4_24_25 output tree below)

#BSUB -J chk_42425_seg
#BSUB -q standard
#BSUB -n 1
#BSUB -R "rusage[mem=8000] span[hosts=1]"
#BSUB -W 60
#BSUB -o logs/chk_42425_seg_%J.out
#BSUB -e logs/chk_42425_seg_%J.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cmap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${LS_SUBCWD:-}" ]]; then
  PROJECT_ROOT="$(cd "$LS_SUBCWD" && pwd)"
elif [[ -n "${LS_EXECCWD:-}" ]]; then
  PROJECT_ROOT="$(cd "$LS_EXECCWD" && pwd)"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1
mkdir -p logs

DEFAULT_BASE="${PROJECT_ROOT}/segmentation_multiscale_cellpose_3D/output/4_24_25_CGN_6_10_2"
export CMAP_CHECK_BASE="${CMAP_CHECK_BASE:-$DEFAULT_BASE}"

echo "=== $(date -Is) seg channel check ==="
echo "CMAP_CHECK_BASE=$CMAP_CHECK_BASE"
echo ""

python3 -u << PY
import os
import sys

import numpy as np
import tifffile as tf

base = os.environ["CMAP_CHECK_BASE"]
if not os.path.isdir(base):
    print(f"ERROR: not a directory: {base}", file=sys.stderr)
    sys.exit(1)


def max_memmap(path: str):
    if not os.path.isfile(path):
        return None
    m = tf.memmap(path, mode="r")
    return int(np.max(m)), m.shape, str(m.dtype)


def has_filtered(sample: str, nm: str) -> bool:
    stem = f"{nm}_crop_3D_indexed_filtered.tif"
    p = os.path.join(base, sample, f"{nm}_crop", "segmentation_3D_masks", stem)
    return os.path.isfile(p)


rows = []
for name in sorted(os.listdir(base)):
    p = os.path.join(base, name)
    if not os.path.isdir(p):
        continue
    p488 = os.path.join(p, "488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif")
    p560 = os.path.join(p, "560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif")
    p642 = os.path.join(p, "642nm_crop/segmentation_3D_masks/642nm_crop_3D_indexed.tif")
    a488 = max_memmap(p488)
    a560 = max_memmap(p560)
    a642 = max_memmap(p642)
    f642, f488, f560 = [has_filtered(name, x) for x in ("642nm", "488nm", "560nm")]
    rows.append((name, a488, a560, a642, f642, f488, f560))

print(f"{'sample':<42} {'488_max':>8} {'560_max':>8} {'642_max':>8}  filt642 filt488 filt560")
for name, a488, a560, a642, f642, f488, f560 in rows:
    def fm(a):
        if a is None:
            return "   —   "
        mx, sh, dt = a
        return f"{mx:8d}"
    print(
        f"{name:<42} {fm(a488)} {fm(a560)} {fm(a642)}  "
        f"{'Y' if f642 else '.'}{' '}{'Y' if f488 else '.'}{' '}{'Y' if f560 else '.'}"
    )

print()
print("--- anomalies (488 max label 0, or much lower than 642) ---")
for name, a488, a560, a642, f642, f488, f560 in rows:
    if a488 is None:
        print(f"  {name}: missing 488 indexed mask")
        continue
    mx488, _, _ = a488
    mx642 = a642[0] if a642 else 0
    if mx488 == 0:
        print(f"  {name}: 488 mask all zeros")
    elif mx642 and mx488 < max(5, mx642 // 100):
        print(f"  {name}: 488 max label {mx488} vs 642 max label {mx642}")
    if not f488:
        print(f"  {name}: missing 488nm *_3D_indexed_filtered.tif (post/filter step incomplete?)")

print()
print("=== done ===")
PY
