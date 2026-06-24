"""Check original-image intensity inside each missing-cell box.

For every missing-cell annotation we read only the three original intensity
channels (642/488/560) at the box's Z slice and compare the box region to the
whole-slice statistics. High relative intensity suggests a real cell that
segmentation missed; near-background intensity suggests an empty region.

Light I/O (a few TIFF pages per box) -> safe to run directly.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
ANN = ROOT / (
    "test_data/cell_exemption_annotations/"
    "cell_exemption_annotations_hchen19_20260624_184451.csv"
)
N_CHANNELS = 5
INTENSITY = [(0, "642"), (1, "488"), (2, "560")]


def read_channel_slice(tf, z, c, n_z):
    z = max(0, min(n_z - 1, z))
    page = z * N_CHANNELS + c
    return tf.asarray(key=page)  # (Y, X)


def main():
    rows = [r for r in csv.DictReader(open(ANN))
            if r["annotation_type"] == "missing_cell"]
    by_img = defaultdict(list)
    for r in rows:
        by_img[(r["sample"], r["image_path"])].append(r)

    for (sample, image_path), srows in by_img.items():
        print(f"\n================ {sample} ================")
        p = Path(image_path)
        if not p.exists():
            print(f"  image missing: {p}")
            continue
        with tifffile.TiffFile(str(p)) as tf:
            n_z, _, Y, X = tf.series[0].shape
            for r in srows:
                z = int(float(r["z_index"]))
                y0, y1, x0, x1 = (int(round(float(r[k]))) for k in
                                  ("y0", "y1", "x0", "x1"))
                y0c, y1c = max(0, y0), min(Y, y1)
                x0c, x1c = max(0, x0), min(X, x1)
                print(f"  [{r['cell_id']}] z={z} box y{y0}:{y1} x{x0}:{x1}")
                for c, name in INTENSITY:
                    sl = read_channel_slice(tf, z, c, n_z).astype(np.float32)
                    box = sl[y0c:y1c, x0c:x1c]
                    if box.size == 0:
                        print(f"      {name}: empty box")
                        continue
                    s_med = float(np.median(sl))
                    s_p95 = float(np.percentile(sl, 95))
                    s_p99 = float(np.percentile(sl, 99))
                    b_med = float(np.median(box))
                    b_mean = float(box.mean())
                    b_p90 = float(np.percentile(box, 90))
                    b_max = float(box.max())
                    frac_hot = float((box > s_p95).mean()) * 100.0
                    ratio = (b_mean / s_med) if s_med > 0 else float("inf")
                    verdict = ("HIGH" if (b_p90 > s_p99 or frac_hot > 10.0)
                               else ("some" if frac_hot > 2.0 else "low"))
                    print(f"      {name}: box mean={b_mean:.0f} med={b_med:.0f} "
                          f"p90={b_p90:.0f} max={b_max:.0f} | "
                          f"slice med={s_med:.0f} p95={s_p95:.0f} p99={s_p99:.0f} "
                          f"| {frac_hot:.1f}% > slice-p95, mean/med={ratio:.2f} "
                          f"-> {verdict}")


if __name__ == "__main__":
    main()
