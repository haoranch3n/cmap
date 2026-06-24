"""Verify exemption annotations against the actual label channel of each image.

For every existing-cell annotation we confirm that the segmentation label
(`cell_id`) really lives inside the recorded bounding box, and report the
label's true extent. For every missing-cell box we report how much existing
labelled signal sits inside it (should be ~0 for a genuine missing cell).

This reads the label channel (channel index 3) of multi-GB combined TIFFs, so
it is meant to run under bsub, not on the login node.
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
LABEL_CHANNEL = 3
N_CHANNELS = 5


def load_label_volume(image_path: Path) -> np.ndarray:
    """Read only the label channel as a (Z, Y, X) int32 volume."""
    with tifffile.TiffFile(str(image_path)) as tf:
        shape = tf.series[0].shape  # (Z, C, Y, X)
        z = shape[0]
        keys = list(range(LABEL_CHANNEL, z * N_CHANNELS, N_CHANNELS))
        vol = tf.asarray(key=keys)  # (Z, Y, X)
    return np.rint(vol).astype(np.int32, copy=False)


def main():
    rows = list(csv.DictReader(open(ANN)))
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[(r["sample"], r["image_path"])].append(r)

    for (sample, image_path), srows in by_sample.items():
        print(f"\n================ {sample} ================")
        p = Path(image_path)
        if not p.exists():
            print(f"  image missing: {p}")
            continue
        lab = load_label_volume(p)
        Z, Y, X = lab.shape
        print(f"  label volume shape (Z,Y,X) = {lab.shape}")

        for r in srows:
            if r["annotation_type"] != "existing_cell":
                continue
            cid = int(r["cell_id"])
            z0, z1, y0, y1, x0, x1 = (int(float(r[k])) for k in
                                      ("z0", "z1", "y0", "y1", "x0", "x1"))
            mask = lab == cid
            total = int(mask.sum())
            sub = mask[z0:z1, y0:y1, x0:x1]
            inside = int(sub.sum())
            if total == 0:
                print(f"  [existing] cell {cid:>3} {r['label']:>4}: "
                      f"LABEL NOT FOUND in volume")
                continue
            zz, yy, xx = np.where(mask)
            tb = (int(zz.min()), int(zz.max()) + 1,
                  int(yy.min()), int(yy.max()) + 1,
                  int(xx.min()), int(xx.max()) + 1)
            frac_in = inside / total
            box_ok = (tb[0] >= z0 and tb[1] <= z1 and tb[2] >= y0 and
                      tb[3] <= y1 and tb[4] >= x0 and tb[5] <= x1)
            print(f"  [existing] cell {cid:>3} {r['label']:>4}: "
                  f"voxels={total}, {frac_in*100:.1f}% inside box, "
                  f"true_bbox(z,y,x)={tb} {'OK' if box_ok else 'OUTSIDE-BOX'}")

        for r in srows:
            if r["annotation_type"] != "missing_cell":
                continue
            z = int(float(r["z_index"]))
            y0, y1, x0, x1 = (int(round(float(r[k]))) for k in
                              ("y0", "y1", "x0", "x1"))
            z = max(0, min(Z - 1, z))
            patch = lab[z, max(0, y0):min(Y, y1), max(0, x0):min(X, x1)]
            nz = int((patch > 0).sum())
            area = patch.size
            ids, counts = np.unique(patch[patch > 0], return_counts=True)
            top = ""
            if ids.size:
                order = np.argsort(counts)[::-1][:3]
                top = ", ".join(f"id{int(ids[i])}:{int(counts[i])}px"
                                for i in order)
            print(f"  [missing ] {r['cell_id']} z={z} box y{y0}:{y1} x{x0}:{x1}: "
                  f"{nz}/{area} ({(nz/area*100) if area else 0:.1f}%) labelled "
                  f"{'-> ' + top if top else '(empty: genuinely missing)'}")


if __name__ == "__main__":
    main()
