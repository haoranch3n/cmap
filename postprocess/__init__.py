"""Post-segmentation mask utilities.

This package centralises the per-sample artefact names for each downstream
"mask variant" so consumers (``features/``, ``qc/``, ``visualization/``,
``pipelines/``) resolve filenames from one place instead of hardcoding them.

Two variants currently exist:

- ``filtered_642`` -- the original 642-anchored chain produced by
  ``postprocess/filter_642_mask.py``. Defaults of every CLI in the repo stay
  on this variant so existing behaviour is unchanged.
- ``union_488_560`` -- the strict-by-construction overlap of the 488 and 560
  indexed masks produced by ``postprocess/union_488_560_mask.py`` (gates 1+2+3
  in the strict-overlap merge). The canonical mask file is
  ``union_488_560_strict_overlap.tif``.

For each variant, the ``pass_otsu_shape`` mapping key is the QC-pass label mask
basename written by ``qc/apply_qc_pass_to_label_mask.py`` (pixel Otsu + shape
gate when enabled in ``qc/filter_by_intensity.py --from-volume``). For
``union_488_560`` the on-disk name is
``union_488_560_strict_overlap_otsu_shape.tif``; filtered crops live under
``cell_box_otsu_shape_filtered/`` (and full-Z under
``cell_box_full_z_otsu_shape_filtered/``) for DINOv2 inputs.

The legacy non-strict artifact ``union_488_560.tif`` is preserved on disk for
provenance but is no longer the active variant input. ``combined`` continues
to point at ``union_488_560_combined.tif`` because consumers
(``qc/filter_by_intensity.py``, ``features/crop_cells.py``,
``qc/bg_threshold_explore.py``) only read its 642/488/560 signal channels
(0..2); the channel-3 mask plane is unused, so the strict-mask + legacy-
combined pairing is signal-identical.
"""

from __future__ import annotations

VARIANT_FILES: dict[str, dict[str, str]] = {
    "filtered_642": {
        "mask": "filtered_642.tif",
        "combined": "filtered_642_combined.tif",
        "cell_box": "cell_boxing",
        "cell_box_full_z": "cell_boxing_full_z",
        "cell_box_filtered": "cell_boxing_filtered",
        "cell_box_full_z_filtered": "cell_boxing_full_z_filtered",
        "cell_qc": "cell_qc",
        "pass_otsu_shape": "filtered_642_pass_otsu_shape.tif",
        "dinov2_volume_bounds": "dinov2_volume_norm_bounds.csv",
    },
    "union_488_560": {
        "mask": "union_488_560_strict_overlap.tif",
        "combined": "union_488_560_combined.tif",
        "cell_box": "cell_boxing_union_488_560",
        "cell_box_full_z": "cell_boxing_full_z_union_488_560",
        "cell_box_filtered": "cell_box_otsu_shape_filtered",
        "cell_box_full_z_filtered": "cell_box_full_z_otsu_shape_filtered",
        "cell_qc": "cell_qc_union_488_560",
        "pass_otsu_shape": "union_488_560_strict_overlap_otsu_shape.tif",
        "dinov2_volume_bounds": "dinov2_volume_norm_bounds_union_488_560.csv",
        "cell_box_bg_sigma_shape":             "cell_box_bg_sigma_shape",
        "cell_box_full_z_bg_sigma_shape":      "cell_box_full_z_bg_sigma_shape",
        "cell_qc_bg_sigma_shape":              "cell_qc_bg_sigma_shape",
        "cell_box_bg_sigma_488560_shape":      "cell_box_bg_sigma_488560_shape",
        "cell_box_full_z_bg_sigma_488560_shape": "cell_box_full_z_bg_sigma_488560_shape",
        "cell_qc_bg_sigma_488560_shape":       "cell_qc_bg_sigma_488560_shape",
    },
}

VALID_VARIANTS: tuple[str, ...] = tuple(VARIANT_FILES.keys())

DEFAULT_VARIANT: str = "filtered_642"


def variant_files(variant: str) -> dict[str, str]:
    """Return the artefact-name mapping for ``variant``; raises on unknown."""
    if variant not in VARIANT_FILES:
        raise ValueError(
            f"Unknown mask variant {variant!r}; expected one of {VALID_VARIANTS}"
        )
    return VARIANT_FILES[variant]


__all__ = ["VARIANT_FILES", "VALID_VARIANTS", "DEFAULT_VARIANT", "variant_files"]
