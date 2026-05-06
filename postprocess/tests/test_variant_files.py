"""Tests for ``postprocess.VARIANT_FILES`` + ``features/crop_cells.py --variant``.

Run from repo root::

    pytest postprocess/tests/test_variant_files.py -q

Or directly (no pytest)::

    python postprocess/tests/test_variant_files.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from postprocess import (  # noqa: E402
    DEFAULT_VARIANT,
    VALID_VARIANTS,
    VARIANT_FILES,
    variant_files,
)


# ---------------------------------------------------------------------------
# VARIANT_FILES schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "mask",
    "combined",
    "cell_box",
    "cell_box_full_z",
    "cell_box_filtered",
    "cell_box_full_z_filtered",
    "cell_qc",
    "pass_otsu_shape",
    "dinov2_volume_bounds",
}


def test_variant_files_keys_match_for_all_variants():
    assert set(VARIANT_FILES.keys()) == set(VALID_VARIANTS)
    assert DEFAULT_VARIANT in VALID_VARIANTS
    for v, mapping in VARIANT_FILES.items():
        missing = REQUIRED_KEYS - set(mapping.keys())
        extra = set(mapping.keys()) - REQUIRED_KEYS
        assert not missing, f"variant {v!r} missing keys: {sorted(missing)}"
        assert not extra, f"variant {v!r} has extra keys: {sorted(extra)}"


def test_variant_files_filtered_642_values_are_legacy_strings():
    v = variant_files("filtered_642")
    assert v["mask"] == "filtered_642.tif"
    assert v["combined"] == "filtered_642_combined.tif"
    assert v["cell_box"] == "cell_boxing"
    assert v["cell_box_full_z"] == "cell_boxing_full_z"
    assert v["cell_box_filtered"] == "cell_boxing_filtered"
    assert v["cell_box_full_z_filtered"] == "cell_boxing_full_z_filtered"
    assert v["cell_qc"] == "cell_qc"
    assert v["pass_otsu_shape"] == "filtered_642_pass_otsu_shape.tif"
    assert v["dinov2_volume_bounds"] == "dinov2_volume_norm_bounds.csv"


def test_variant_files_union_488_560_values():
    v = variant_files("union_488_560")
    assert v["mask"] == "union_488_560.tif"
    assert v["combined"] == "union_488_560_combined.tif"
    assert v["cell_box"] == "cell_boxing_union_488_560"
    assert v["cell_box_full_z"] == "cell_boxing_full_z_union_488_560"
    assert v["cell_box_filtered"] == "cell_box_otsu_shape_filtered"
    assert v["cell_box_full_z_filtered"] == "cell_box_full_z_otsu_shape_filtered"
    assert v["cell_qc"] == "cell_qc_union_488_560"
    assert v["pass_otsu_shape"] == "union_488_560_otsu_shape.tif"
    assert v["dinov2_volume_bounds"] == "dinov2_volume_norm_bounds_union_488_560.csv"


def test_variant_files_unknown_variant_raises():
    try:
        variant_files("does_not_exist")
    except ValueError as exc:
        assert "Unknown mask variant" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown variant")


# ---------------------------------------------------------------------------
# crop_cells.py --variant union_488_560 synthetic fixture
# ---------------------------------------------------------------------------


def _build_union_fixture(output_dir: Path) -> None:
    """Write a tiny union_488_560.tif + union_488_560_combined.tif under
    ``output_dir`` so ``features/crop_cells.py --variant union_488_560`` can
    process it without any segmentation step.
    """
    z, y, x = 4, 24, 24
    mask = np.zeros((z, y, x), dtype=np.uint16)
    mask[1:3, 6:11, 6:11] = 1
    mask[1:3, 14:19, 14:19] = 2

    combined = np.zeros((z, 4, y, x), dtype=np.float32)
    rng = np.random.default_rng(0)
    for c in range(3):
        combined[:, c] = rng.uniform(low=10.0, high=200.0, size=(z, y, x))
    combined[:, 3] = mask.astype(np.float32)

    tifffile.imwrite(
        str(output_dir / "union_488_560_combined.tif"),
        combined,
        imagej=True,
        photometric="minisblack",
        metadata={"axes": "ZCYX", "mode": "grayscale"},
    )
    tifffile.imwrite(
        str(output_dir / "union_488_560.tif"),
        mask,
        compression="zlib",
    )


def test_crop_cells_variant_union_488_560_writes_box_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "sample01"
    output_dir.mkdir(parents=True)
    _build_union_fixture(output_dir)

    crop_py = _REPO / "features" / "crop_cells.py"
    assert crop_py.is_file(), crop_py
    rc = subprocess.call(
        [
            sys.executable,
            str(crop_py),
            "--variant",
            "union_488_560",
            "--data-dir",
            str(output_dir),
            "--output-dir",
            str(output_dir),
            "--margin-xy",
            "2",
            "--margin-z",
            "0",
            "--force",
        ],
        cwd=str(_REPO),
    )
    assert rc == 0, f"crop_cells.py returned {rc}"

    box_dir = output_dir / "cell_boxing_union_488_560"
    assert box_dir.is_dir(), f"missing box dir: {box_dir}"
    cells = sorted(box_dir.glob("cell_*.tif"))
    assert len(cells) == 2, f"expected 2 cells, got {len(cells)}: {cells}"

    for p in cells:
        arr = tifffile.imread(str(p))
        assert arr.ndim == 4 and arr.shape[1] >= 4, f"unexpected shape {arr.shape} for {p}"

    # The legacy 642 paths must NOT be touched by the union variant.
    assert not (output_dir / "cell_boxing").exists()


# ---------------------------------------------------------------------------
# Test runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import traceback

    funcs = [
        test_variant_files_keys_match_for_all_variants,
        test_variant_files_filtered_642_values_are_legacy_strings,
        test_variant_files_union_488_560_values,
        test_variant_files_unknown_variant_raises,
    ]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:  # pragma: no cover
            traceback.print_exc()
            print(f"  FAIL  {fn.__name__}")
            failed += 1

    with tempfile.TemporaryDirectory() as td:
        try:
            test_crop_cells_variant_union_488_560_writes_box_dir(Path(td))
            print("  PASS  test_crop_cells_variant_union_488_560_writes_box_dir")
        except Exception:  # pragma: no cover
            traceback.print_exc()
            print("  FAIL  test_crop_cells_variant_union_488_560_writes_box_dir")
            failed += 1

    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print("All tests passed.")
