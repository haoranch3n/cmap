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


def _minimal_row(cell_id: str = "1") -> AnnotationRow:
    """Smallest annotation row that round-trips through the CSV writer."""
    return AnnotationRow(
        image_name=COMBINED_TIFF_NAME,
        cell_id=cell_id,
        annotation_type=MISSING_CELL,
        label="missing",
        batch="batchA",
        sample="Sample1",
        image_path=f"/fake/{cell_id}/{COMBINED_TIFF_NAME}",
    )


def test_annotation_path_files_session_under_reviewer_then_date(tmp_path: Path) -> None:
    out = annotation_csv_path(tmp_path / "ann", "20260908_223023", "hchen19")
    assert out == tmp_path / "ann" / "hchen19" / "20260908" / (
        "cell_exemption_annotations_223023.csv"
    )


def test_same_day_sessions_share_a_date_dir_but_not_a_file(tmp_path: Path) -> None:
    a = annotation_csv_path(tmp_path / "ann", "20260908_090000", "hchen19")
    b = annotation_csv_path(tmp_path / "ann", "20260908_223023", "hchen19")
    assert a.parent == b.parent
    assert a.name != b.name


def test_annotation_path_without_reviewer_uses_unknown_dir(tmp_path: Path) -> None:
    out = annotation_csv_path(tmp_path / "ann", "20260908_223023", "")
    assert out.parent == tmp_path / "ann" / "unknown" / "20260908"


def test_reviewer_name_cannot_escape_the_annotation_root(tmp_path: Path) -> None:
    root = (tmp_path / "ann").resolve()
    for hostile in ("../..", "/etc", "a/../../b"):
        out = annotation_csv_path(root, "20260908_223023", hostile).resolve()
        assert root in out.parents, f"{hostile!r} escaped to {out}"


def test_load_annotation_folder_finds_nested_sessions(tmp_path: Path) -> None:
    root = tmp_path / "ann"
    save_annotations_atomic(
        annotation_csv_path(root, "20260907_100000", "alice"), [_minimal_row("1")]
    )
    save_annotations_atomic(
        annotation_csv_path(root, "20260908_110000", "bob"), [_minimal_row("2")]
    )
    loaded = load_annotation_folder(root)
    assert len(loaded) == 2


def test_load_annotation_folder_still_reads_pre_migration_flat_csvs(tmp_path: Path) -> None:
    root = tmp_path / "ann"
    root.mkdir()
    save_annotations_atomic(root / "cell_exemption_annotations_legacy.csv", [_minimal_row("9")])
    loaded = load_annotation_folder(root)
    assert len(loaded) == 1
