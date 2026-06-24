"""Data models for CMAP cell exemption annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


COMBINED_TIFF_NAME = "cell_box_bg_sigma_488560_shape_combined.tif"
CELL_BOX_SUBDIR = "cell_box_bg_sigma_488560_shape"
COORDINATE_CSV_NAME = "cell_coordinates.csv"
ANNOTATION_CSV_PREFIX = "cell_exemption_annotations"
ANNOTATION_CSV_GLOB = f"{ANNOTATION_CSV_PREFIX}*.csv"

ANNOTATION_COLUMNS = [
    "image_name",
    "cell_id",
    "annotation_type",
    "label",
    "batch",
    "sample",
    "image_path",
    "z0",
    "z1",
    "y0",
    "y1",
    "x0",
    "x1",
    "z_index",
    "center_x",
    "center_y",
    "width",
    "height",
    "reviewer",
    "notes",
    "created_at",
    "updated_at",
]

EXISTING_CELL = "existing_cell"
MISSING_CELL = "missing_cell"

GOOD_LABEL = "good"
BAD_LABEL = "bad"
MISSING_LABEL = "missing"
UNANNOTATED_LABEL = "unannotated"


@dataclass(frozen=True)
class SampleRecord:
    """One reconstructed CMAP sample volume and its coordinate table."""

    batch: str
    sample: str
    sample_dir: Path
    image_path: Path
    coordinate_csv: Path

    @property
    def image_name(self) -> str:
        return self.image_path.name


@dataclass(frozen=True)
class CellBoxRecord:
    """Half-open 3D crop coordinates for one existing segmented cell."""

    cell_id: int
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def center_zyx(self) -> tuple[float, float, float]:
        return (
            (self.z0 + self.z1) / 2.0,
            (self.y0 + self.y1) / 2.0,
            (self.x0 + self.x1) / 2.0,
        )

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class AnnotationRow:
    """Serializable annotation row.

    `cell_id` is a string so missing cells can use IDs such as
    `missing_0001` without colliding with numeric segmentation labels.
    """

    image_name: str
    cell_id: str
    annotation_type: str
    label: str
    batch: str
    sample: str
    image_path: str
    z0: str = ""
    z1: str = ""
    y0: str = ""
    y1: str = ""
    x0: str = ""
    x1: str = ""
    z_index: str = ""
    center_x: str = ""
    center_y: str = ""
    width: str = ""
    height: str = ""
    reviewer: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.image_path, self.annotation_type, self.cell_id)

    def as_dict(self) -> dict[str, str]:
        return {col: str(getattr(self, col)) for col in ANNOTATION_COLUMNS}

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "AnnotationRow":
        data = {col: "" for col in ANNOTATION_COLUMNS}
        for col in ANNOTATION_COLUMNS:
            value = values.get(col, "")
            data[col] = "" if value is None else str(value)
        return cls(**data)

    @classmethod
    def for_existing_cell(
        cls,
        sample: SampleRecord,
        box: CellBoxRecord,
        label: str,
        *,
        reviewer: str = "",
        notes: str = "",
        created_at: str = "",
        updated_at: str = "",
        image_path_key: str | None = None,
    ) -> "AnnotationRow":
        key = image_path_key or str(sample.image_path)
        return cls(
            image_name=sample.image_name,
            cell_id=str(box.cell_id),
            annotation_type=EXISTING_CELL,
            label=label,
            batch=sample.batch,
            sample=sample.sample,
            image_path=key,
            z0=str(box.z0),
            z1=str(box.z1),
            y0=str(box.y0),
            y1=str(box.y1),
            x0=str(box.x0),
            x1=str(box.x1),
            reviewer=reviewer,
            notes=notes,
            created_at=created_at,
            updated_at=updated_at,
        )


def rows_to_dataframe(rows: Iterable[AnnotationRow]) -> pd.DataFrame:
    """Convert annotation rows to a stable-column DataFrame."""

    return pd.DataFrame([row.as_dict() for row in rows], columns=ANNOTATION_COLUMNS)


def dataframe_to_rows(df: pd.DataFrame) -> list[AnnotationRow]:
    """Convert a DataFrame into annotation rows, tolerating older partial CSVs."""

    rows: list[AnnotationRow] = []
    for _, row in df.iterrows():
        rows.append(AnnotationRow.from_mapping(row.to_dict()))
    return rows
