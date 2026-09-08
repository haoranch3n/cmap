from __future__ import annotations

from pathlib import Path

from cmap_cell_exemption_plugin.data_io import (
    annotation_csv_path,
    discover_samples,
    load_annotation_folder,
    load_cell_boxes,
    path_key,
    save_annotations_atomic,
)
from cmap_cell_exemption_plugin.models import (
    BAD_LABEL,
    CELL_BOX_SUBDIR,
    COMBINED_TIFF_NAME,
    COORDINATE_CSV_NAME,
    EXISTING_CELL,
    MISSING_CELL,
    AnnotationRow,
    CellBoxRecord,
    SampleRecord,
)


def _make_sample(root: Path, batch: str = "batchA", sample: str = "Sample1") -> Path:
    sample_dir = root / batch / sample
    box_dir = sample_dir / CELL_BOX_SUBDIR
    box_dir.mkdir(parents=True)
    (sample_dir / COMBINED_TIFF_NAME).write_bytes(b"placeholder")
    (box_dir / COORDINATE_CSV_NAME).write_text(
        "cell_id,z0,z1,y0,y1,x0,x1\n"
        "2,0,3,10,20,30,40\n"
        "1,1,4,11,21,31,41\n"
    )
    return sample_dir


def test_discover_samples_and_load_boxes(tmp_path: Path) -> None:
    sample_dir = _make_sample(tmp_path)
    samples = discover_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].batch == "batchA"
    assert samples[0].sample == "Sample1"
    assert samples[0].sample_dir == sample_dir

    boxes = load_cell_boxes(samples[0].coordinate_csv)
    assert [box.cell_id for box in boxes] == [1, 2]
    assert boxes[0] == CellBoxRecord(1, 1, 4, 11, 21, 31, 41)


def test_save_and_load_annotation_folder(tmp_path: Path) -> None:
    sample_dir = _make_sample(tmp_path)
    sample = discover_samples(tmp_path)[0]
    row = AnnotationRow.for_existing_cell(
        sample,
        CellBoxRecord(1, 1, 4, 11, 21, 31, 41),
        BAD_LABEL,
        reviewer="tester",
        notes="mask artifact",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        image_path_key=str(sample_dir / COMBINED_TIFF_NAME),
    )
    missing = AnnotationRow(
        image_name=COMBINED_TIFF_NAME,
        cell_id="missing_0001",
        annotation_type=MISSING_CELL,
        label="missing",
        batch="batchA",
        sample="Sample1",
        image_path=str(sample_dir / COMBINED_TIFF_NAME),
        z0="5",
        z1="6",
        y0="10",
        y1="20",
        x0="30",
        x1="40",
        z_index="5",
        center_x="35",
        center_y="15",
        width="10",
        height="10",
    )
    out = annotation_csv_path(tmp_path / "ann", "stamp")
    save_annotations_atomic(out, [row, missing])

    loaded = load_annotation_folder(tmp_path / "ann")
    assert len(loaded) == 2
    assert (str(sample_dir / COMBINED_TIFF_NAME), EXISTING_CELL, "1") in loaded
    assert (str(sample_dir / COMBINED_TIFF_NAME), MISSING_CELL, "missing_0001") in loaded
    assert loaded[(str(sample_dir / COMBINED_TIFF_NAME), EXISTING_CELL, "1")].notes == "mask artifact"


def test_annotation_columns_start_with_image_name_cell_id(tmp_path: Path) -> None:
    sample = SampleRecord(
        batch="batchA",
        sample="Sample1",
        sample_dir=tmp_path,
        image_path=tmp_path / COMBINED_TIFF_NAME,
        coordinate_csv=tmp_path / CELL_BOX_SUBDIR / COORDINATE_CSV_NAME,
    )
    row = AnnotationRow.for_existing_cell(
        sample,
        CellBoxRecord(7, 0, 1, 2, 3, 4, 5),
        BAD_LABEL,
    )
    out = tmp_path / "rows.csv"
    save_annotations_atomic(out, [row])
    header = out.read_text().splitlines()[0].split(",")
    assert header[:2] == ["image_name", "cell_id"]


def test_path_key_normalizes_mount_alias() -> None:
    jude = (
        "/research_jude/rgs01_jude/dept/DNB/core_operations/"
        "ImageAnalysis/Core/Haoran/cmap/foo.tif"
    )
    dept = (
        "/research/dept/dnb/core_operations/"
        "ImageAnalysis/Core/Haoran/cmap/foo.tif"
    )
    assert path_key(jude) == path_key(dept)
