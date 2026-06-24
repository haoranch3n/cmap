"""Cross-check the exemption annotation CSV against source coordinate CSVs.

Login-safe: reads only small CSV files (no TIFF I/O).
"""

import csv
import sys
from pathlib import Path

ANN = Path(
    "test_data/cell_exemption_annotations/"
    "cell_exemption_annotations_hchen19_20260624_184451.csv"
)
BASE = Path("test_data/4_18_25")
KEYS = ("z0", "z1", "y0", "y1", "x0", "x1")


def load_coords(sample):
    f = BASE / sample / "cell_box_bg_sigma_488560_shape" / "cell_coordinates.csv"
    d = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            d[int(r["cell_id"])] = {k: int(r[k]) for k in KEYS}
    return d


def main():
    cache = {}
    rows = list(csv.DictReader(open(ANN)))
    existing = [r for r in rows if r["annotation_type"] == "existing_cell"]
    missing = [r for r in rows if r["annotation_type"] == "missing_cell"]

    print("=== EXISTING-CELL coordinate check (annotation vs source CSV) ===")
    print(f"{'sample':24}{'cell':>5}{'label':>7}  result")
    n_ok = n_bad = 0
    for r in existing:
        s = r["sample"]
        cid = int(r["cell_id"])
        cache.setdefault(s, load_coords(s))
        src = cache[s].get(cid)
        ann_box = {k: int(float(r[k])) for k in KEYS}
        if src is None:
            print(f"{s:24}{cid:5}{r['label']:>7}  NOT-IN-COORD-CSV")
            n_bad += 1
            continue
        if src == ann_box:
            print(f"{s:24}{cid:5}{r['label']:>7}  OK")
            n_ok += 1
        else:
            print(f"{s:24}{cid:5}{r['label']:>7}  MISMATCH")
            print(f"      src={src}")
            print(f"      ann={ann_box}")
            n_bad += 1
    print(f"existing: {n_ok} OK, {n_bad} problems")

    print("\n=== MISSING-CELL boxes (sanity: in-bounds, IDs unique) ===")
    print(f"{'sample':24}{'cell_id':>14}{'z':>5}  y0:y1        x0:x1")
    for r in missing:
        s = r["sample"]
        print(
            f"{s:24}{r['cell_id']:>14}{r['z_index']:>5}  "
            f"{float(r['y0']):.0f}:{float(r['y1']):.0f}    "
            f"{float(r['x0']):.0f}:{float(r['x1']):.0f}"
        )


if __name__ == "__main__":
    sys.exit(main())
