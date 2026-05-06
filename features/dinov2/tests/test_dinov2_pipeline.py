"""Synthetic-data tests for the DINOv2 cell-embedding pipeline.

The DINOv2 encoder is replaced with a deterministic stub so the tests run
on CPU without ``torch.hub`` access.

Run with::

    pytest features/dinov2/tests/test_dinov2_pipeline.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
import torch.nn as nn

_FEATURES_PARENT = Path(__file__).resolve().parents[3]
if str(_FEATURES_PARENT) not in sys.path:
    sys.path.insert(0, str(_FEATURES_PARENT))

from features.dinov2 import (  # noqa: E402  (path bootstrap)
    DinoV2Config,
    DinoV2Encoder,
    extract_one_cell,
    extract_sample,
    mean_pool,
)
from features.dinov2.crop_io import read_cell_crop  # noqa: E402
from features.dinov2.transforms import (  # noqa: E402
    apply_primary_mask,
    fixed_channel_normalize,
    is_empty_slice,
    percentile_normalize,
    prepare_batch,
    prepare_planes_batch,
)
from features.dinov2.volume_norm_csv import (  # noqa: E402
    read_bounds_csv,
    write_bounds_csv,
)


# ---------------------------------------------------------------------------
# Stub encoder
# ---------------------------------------------------------------------------

EMBED_DIM = 768


class _StubViT(nn.Module):
    """Stand-in for the torch.hub DINOv2 model.

    Produces a deterministic 768-D vector per image using a fixed linear
    projection of the average pixel value across channels.  This keeps the
    output dependent on the input so tests can sanity-check shapes and basic
    behaviour, while staying fast and CPU-only.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        gen = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.randn(3, embed_dim, generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, 3) -> (B, embed_dim)
        per_channel = x.mean(dim=(2, 3))
        return per_channel @ self.proj


@pytest.fixture
def stub_cfg() -> DinoV2Config:
    return DinoV2Config(
        target_size=28,             # multiple of 14, keeps the test tiny
        batch_size=4,
        device="cpu",
        empty_slice_nonzero_frac=0.01,
        apply_mask=False,
    )


@pytest.fixture
def stub_encoder(stub_cfg: DinoV2Config) -> DinoV2Encoder:
    return DinoV2Encoder(stub_cfg, model=_StubViT())


# ---------------------------------------------------------------------------
# Synthetic crop helpers
# ---------------------------------------------------------------------------

def _make_synthetic_crop(z: int = 6, y: int = 32, x: int = 32, seed: int = 0) -> np.ndarray:
    """Build a (Z, 5, Y, X) cell crop in the format crop_cells.py writes.

    Channels: [642, 488, 560, Primary_Cell_Mask, Mask_Boundary].
    The first z plane is intentionally empty so we can test the empty-slice
    filter, and the primary mask is a centred square so apply_mask has visible
    effect.
    """
    rng = np.random.default_rng(seed)
    intensities = rng.uniform(50, 1500, size=(z, 3, y, x)).astype(np.float32)
    intensities[0] = 0.0  # empty top slice

    mask = np.zeros((z, y, x), dtype=np.float32)
    cy, cx = y // 2, x // 2
    half = min(y, x) // 4
    mask[:, cy - half: cy + half, cx - half: cx + half] = 1.0
    boundary = np.zeros_like(mask)
    boundary[:, cy - half, cx - half: cx + half] = 1.0

    crop = np.concatenate(
        [intensities, mask[:, None], boundary[:, None]],
        axis=1,
    ).astype(np.float32)
    return crop


def _write_cell_tif(box_dir: Path, cell_id: int, crop: np.ndarray) -> Path:
    box_dir.mkdir(parents=True, exist_ok=True)
    path = box_dir / f"cell_{cell_id:04d}.tif"
    tifffile.imwrite(
        str(path),
        crop,
        imagej=True,
        photometric="minisblack",
        metadata={"axes": "ZCYX"},
    )
    return path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_read_cell_crop_returns_three_intensity_channels(tmp_path: Path) -> None:
    crop = _make_synthetic_crop()
    path = _write_cell_tif(tmp_path / "cell_boxing", 1, crop)

    intensity, mask = read_cell_crop(path)

    assert intensity.shape == (crop.shape[0], 3, crop.shape[2], crop.shape[3])
    assert intensity.dtype == np.float32
    assert mask is not None
    assert mask.dtype == np.bool_
    assert mask.shape == (crop.shape[0], crop.shape[2], crop.shape[3])
    np.testing.assert_array_equal(mask, crop[:, 3] > 0)


def test_percentile_normalize_outputs_unit_range() -> None:
    rng = np.random.default_rng(0)
    plane = rng.uniform(50, 5000, size=(3, 16, 16)).astype(np.float32)
    out = percentile_normalize(plane, p_low=1.0, p_high=99.0)
    assert out.shape == plane.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    # Per-channel, not global: each channel should saturate to ~1 somewhere.
    for c in range(3):
        assert out[c].max() == pytest.approx(1.0, abs=1e-5)


def test_is_empty_slice_flags_zero_plane() -> None:
    plane = np.zeros((3, 8, 8), dtype=np.float32)
    assert is_empty_slice(plane, threshold=0.01) is True
    plane[0, 0, 0] = 1.0
    nonzero_frac = 1 / (8 * 8)
    assert is_empty_slice(plane, threshold=nonzero_frac * 2) is True  # below threshold
    assert is_empty_slice(plane, threshold=nonzero_frac / 2) is False  # above threshold
    # threshold=0 disables the filter
    assert is_empty_slice(np.zeros((3, 8, 8), dtype=np.float32), threshold=0.0) is False


def test_apply_primary_mask_zeros_outside_mask() -> None:
    plane = np.ones((3, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    out = apply_primary_mask(plane, mask)
    assert out.shape == plane.shape
    assert out[:, 0, 0].sum() == 0
    np.testing.assert_array_equal(out[:, 1:3, 1:3], 1.0)


def test_fixed_channel_normalize_matches_percentile_on_uniform() -> None:
    plane = np.full((3, 8, 8), 100.0, dtype=np.float32)
    plane[0, 0, 0] = 200.0
    lo = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    hi = np.array([200.0, 100.0, 100.0], dtype=np.float32)
    out = fixed_channel_normalize(plane, lo, hi)
    assert out[0, 0, 0] == pytest.approx(1.0)
    assert out[0, 1, 1] == pytest.approx(0.5)


def test_volume_bounds_csv_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "bounds.csv"
    lo = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    hi = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    write_bounds_csv(p, lo, hi, source_tif="/fake.tif")
    lo2, hi2 = read_bounds_csv(p)
    np.testing.assert_allclose(lo2, lo)
    np.testing.assert_allclose(hi2, hi)


def test_prepare_batch_drops_empty_slices_and_resizes() -> None:
    crop = _make_synthetic_crop(z=4, y=32, x=32)
    intensity = crop[:, :3]
    mask = crop[:, 3] > 0
    batch, valid = prepare_batch(
        intensity, mask,
        p_low=1.0, p_high=99.0,
        target_size=28,
        apply_mask=True,
        use_imagenet_stats=False,
        empty_slice_nonzero_frac=0.01,
        device="cpu",
    )
    # First slice is empty, three should remain
    assert valid == [1, 2, 3]
    assert batch.shape == (3, 3, 28, 28)
    assert batch.dtype == torch.float32


def test_encode_slices_batch_matches_sequential(stub_encoder: DinoV2Encoder) -> None:
    """Batched forward (B=3) equals three single-image forwards for the stub ViT."""
    torch.manual_seed(1)
    batch = torch.randn(3, 3, 28, 28, dtype=torch.float32)
    batched = stub_encoder.encode_slices(batch).cpu()
    rows = [stub_encoder.encode_slices(batch[i : i + 1]).cpu() for i in range(3)]
    stacked = torch.cat(rows, dim=0)
    torch.testing.assert_close(batched, stacked, rtol=0, atol=0)


def test_mean_pool_returns_correct_shape() -> None:
    embeds = torch.randn(5, 768)
    pooled = mean_pool(embeds)
    assert pooled.shape == (768,)
    torch.testing.assert_close(pooled, embeds.mean(dim=0))


def test_mean_pool_rejects_empty() -> None:
    with pytest.raises(ValueError):
        mean_pool(torch.empty(0, 768))


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

def test_extract_one_cell_skips_all_empty_volume(
    tmp_path: Path,
    stub_cfg: DinoV2Config,
    stub_encoder: DinoV2Encoder,
) -> None:
    crop = np.zeros((3, 5, 16, 16), dtype=np.float32)
    path = _write_cell_tif(tmp_path / "cell_boxing", 7, crop)
    result = extract_one_cell(path, stub_encoder, stub_cfg)
    assert result is None


def test_extract_one_cell_orthogonal_concat_2304d(
    tmp_path: Path,
    stub_encoder: DinoV2Encoder,
) -> None:
    cfg = DinoV2Config(
        target_size=28,
        batch_size=4,
        device="cpu",
        extraction_mode="orthogonal_concat",
        empty_slice_nonzero_frac=0.0,
    )
    crop = _make_synthetic_crop(z=8, y=32, x=32, seed=2)
    path = _write_cell_tif(tmp_path / "cell_boxing", 11, crop)
    result = extract_one_cell(path, stub_encoder, cfg, volume_lo_hi=None)
    assert result is not None
    assert result.embedding.shape == (3 * EMBED_DIM,)
    assert result.centroid_zyx is not None


def test_prepare_planes_batch_three_views() -> None:
    rng = np.random.default_rng(0)
    z, y, x = 6, 16, 20
    intensity = rng.uniform(10, 200, size=(z, 3, y, x)).astype(np.float32)
    masks = [np.ones((y, x), dtype=bool) for _ in range(3)]
    planes = [
        intensity[2],
        np.transpose(intensity[:, :, 7, :], (1, 0, 2)),
        np.transpose(intensity[:, :, :, 10], (1, 0, 2)),
    ]
    batch = prepare_planes_batch(
        planes,
        masks,
        p_low=1.0,
        p_high=99.0,
        target_size=28,
        apply_mask=False,
        use_imagenet_stats=False,
        empty_slice_nonzero_frac=0.0,
        device="cpu",
        volume_lo_hi=None,
    )
    assert batch is not None
    assert batch.shape == (3, 3, 28, 28)


def test_extract_one_cell_produces_768d_embedding(
    tmp_path: Path,
    stub_cfg: DinoV2Config,
    stub_encoder: DinoV2Encoder,
) -> None:
    crop = _make_synthetic_crop(z=4, y=32, x=32, seed=1)
    path = _write_cell_tif(tmp_path / "cell_boxing", 3, crop)
    result = extract_one_cell(path, stub_encoder, stub_cfg)
    assert result is not None
    assert result.cell_id == 3
    assert result.embedding.shape == (EMBED_DIM,)
    assert result.embedding.dtype == np.float32
    assert result.n_valid_slices == 3  # the first z plane is empty
    assert result.original_shape == (4, 32, 32)


def test_extract_sample_writes_expected_files(
    tmp_path: Path,
    stub_cfg: DinoV2Config,
    stub_encoder: DinoV2Encoder,
) -> None:
    box_dir = tmp_path / "cell_boxing"
    _write_cell_tif(box_dir, 1, _make_synthetic_crop(seed=10))
    _write_cell_tif(box_dir, 2, _make_synthetic_crop(seed=11))

    rc = extract_sample(tmp_path, stub_cfg, encoder=stub_encoder, force=True)
    assert rc == 0

    qc_dir = tmp_path / "cell_qc"
    npy = qc_dir / "dinov2_embeddings.npy"
    csv = qc_dir / "dinov2_embeddings.csv"
    log = qc_dir / "dinov2_extract_summary.txt"
    assert npy.is_file() and csv.is_file() and log.is_file()

    arr = np.load(str(npy))
    assert arr.shape == (2, EMBED_DIM)
    assert arr.dtype == np.float32

    import csv as _csv
    with open(csv) as fh:
        rows = list(_csv.DictReader(fh))
    assert [int(r["cell_id"]) for r in rows] == [1, 2]
    assert {int(r["embedding_dim"]) for r in rows} == {EMBED_DIM}
    assert all(int(r["n_valid_slices"]) > 0 for r in rows)


def test_extract_sample_with_save_slice_embeddings(tmp_path: Path) -> None:
    cfg = DinoV2Config(
        target_size=28,
        batch_size=4,
        device="cpu",
        save_slice_embeddings=True,
    )
    encoder = DinoV2Encoder(cfg, model=_StubViT())
    box_dir = tmp_path / "cell_boxing"
    _write_cell_tif(box_dir, 5, _make_synthetic_crop(seed=20))
    rc = extract_sample(tmp_path, cfg, encoder=encoder, force=True)
    assert rc == 0
    npz = tmp_path / "cell_qc" / "dinov2_embeddings_per_slice.npz"
    assert npz.is_file()
    with np.load(str(npz)) as data:
        assert "cell_0005" in data
        assert data["cell_0005"].shape[1] == EMBED_DIM


def test_extract_sample_filtered_box_dir_writes_to_filtered_qc(
    tmp_path: Path,
    stub_encoder: DinoV2Encoder,
) -> None:
    fdir = tmp_path / "cell_boxing_filtered"
    qdir = tmp_path / "cell_qc_filtered"
    _write_cell_tif(fdir, 1, _make_synthetic_crop(z=5, y=24, x=24, seed=7))
    lo = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    hi = np.array([2000.0, 2000.0, 2000.0], dtype=np.float32)
    write_bounds_csv(tmp_path / "dinov2_volume_norm_bounds.csv", lo, hi, source_tif="x")
    cfg = DinoV2Config(
        target_size=28,
        batch_size=4,
        device="cpu",
        norm_scope="volume",
        empty_slice_nonzero_frac=0.0,
    )
    rc = extract_sample(
        tmp_path,
        cfg,
        encoder=stub_encoder,
        force=True,
        cell_boxing_dirname="cell_boxing_filtered",
        cell_qc_dirname="cell_qc_filtered",
    )
    assert rc == 0
    assert (qdir / "dinov2_embeddings.npy").is_file()
    assert not (tmp_path / "cell_qc" / "dinov2_embeddings.npy").exists()


def test_extract_sample_volume_norm_reads_csv_only(
    tmp_path: Path,
    stub_encoder: DinoV2Encoder,
) -> None:
    """Volume bounds from CSV; no need for filtered_642_combined.tif."""
    box_dir = tmp_path / "cell_boxing"
    _write_cell_tif(box_dir, 1, _make_synthetic_crop(z=5, y=24, x=24, seed=99))
    lo = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    hi = np.array([2000.0, 2000.0, 2000.0], dtype=np.float32)
    write_bounds_csv(
        tmp_path / "dinov2_volume_norm_bounds.csv",
        lo,
        hi,
        source_tif="synthetic",
    )
    cfg = DinoV2Config(
        target_size=28,
        batch_size=4,
        device="cpu",
        norm_scope="volume",
        empty_slice_nonzero_frac=0.0,
    )
    rc = extract_sample(tmp_path, cfg, encoder=stub_encoder, force=True)
    assert rc == 0
    arr = np.load(str(tmp_path / "cell_qc" / "dinov2_embeddings.npy"))
    assert arr.shape == (1, EMBED_DIM)


def test_extract_sample_skip_existing(tmp_path: Path, stub_cfg: DinoV2Config) -> None:
    encoder = DinoV2Encoder(stub_cfg, model=_StubViT())
    box_dir = tmp_path / "cell_boxing"
    _write_cell_tif(box_dir, 1, _make_synthetic_crop(seed=1))

    rc = extract_sample(tmp_path, stub_cfg, encoder=encoder, force=True)
    assert rc == 0

    npy = tmp_path / "cell_qc" / "dinov2_embeddings.npy"
    mtime = npy.stat().st_mtime

    rc = extract_sample(tmp_path, stub_cfg, encoder=encoder, force=False)
    assert rc == 0
    assert npy.stat().st_mtime == mtime  # not overwritten


def test_extract_one_cell_apply_mask_changes_embedding(
    tmp_path: Path,
) -> None:
    base = DinoV2Config(target_size=28, batch_size=4, device="cpu")
    masked = DinoV2Config(target_size=28, batch_size=4, device="cpu", apply_mask=True)
    encoder_a = DinoV2Encoder(base, model=_StubViT())
    encoder_b = DinoV2Encoder(masked, model=_StubViT())
    crop = _make_synthetic_crop(z=4, y=32, x=32, seed=42)
    path = _write_cell_tif(tmp_path / "cell_boxing", 9, crop)

    r_a = extract_one_cell(path, encoder_a, base)
    r_b = extract_one_cell(path, encoder_b, masked)
    assert r_a is not None and r_b is not None
    # The mask zeros most of the slice, so the embedding must differ.
    assert not np.allclose(r_a.embedding, r_b.embedding)
