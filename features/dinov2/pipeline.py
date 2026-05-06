"""High-level DINOv2 extraction routines.

This module is the integration point between the existing CMAP cell-crop
layout (``output/<sample>/cell_boxing/cell_*.tif``) and the frozen DINOv2
encoder.  The output mirrors the structure of ``cell_qc/qc_features.csv`` so
the napari plugin and pooled visualization scripts can join on
``(sample, cell_id)`` without any extra plumbing.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .aggregate import mean_pool
from .centroid import compute_robust_centroid, slice_orthogonal_planes
from .config import DinoV2Config
from .crop_io import list_cell_crops, parse_cell_id, read_cell_crop
from .encoder import DinoV2Encoder
from .transforms import prepare_batch, prepare_planes_batch
from .volume_norm_csv import (
    compute_bounds_from_combined_tif,
    default_bounds_csv_path,
    read_bounds_csv,
    write_bounds_csv,
)

logger = logging.getLogger(__name__)

EMBEDDINGS_NPY = "dinov2_embeddings.npy"
EMBEDDINGS_CSV = "dinov2_embeddings.csv"
PER_SLICE_NPZ = "dinov2_embeddings_per_slice.npz"
EXTRACT_LOG = "dinov2_extract_summary.txt"


@dataclass
class CellResult:
    """Per-cell output of :func:`extract_one_cell`."""

    cell_id: int
    embedding: np.ndarray  # shape (D,) or (3 * D,) in orthogonal_concat mode
    n_valid_slices: int
    valid_indices: list[int]
    slice_embeddings: np.ndarray | None  # (N_valid, D) or (3, D) when requested
    original_shape: tuple[int, int, int]  # (Z, Y, X)
    centroid_zyx: tuple[float, float, float] | None = None


def _volume_lo_hi_for_sample(
    output_dir: Path,
    cfg: DinoV2Config,
    *,
    recompute_volume_norm: bool,
    volume_bounds_csv: Path | None,
    combined_filename: str = "filtered_642_combined.tif",
    bounds_csv_filename: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if cfg.norm_scope != "volume":
        return None
    combined = output_dir / combined_filename
    csv_path = (
        Path(volume_bounds_csv)
        if volume_bounds_csv
        else default_bounds_csv_path(output_dir, filename=bounds_csv_filename)
    )
    if recompute_volume_norm:
        if not combined.is_file():
            print(f"ERROR: --recompute-volume-norm requires {combined}")
            raise FileNotFoundError(str(combined))
        lo, hi = compute_bounds_from_combined_tif(combined, p_high=99.99)
        write_bounds_csv(csv_path, lo, hi, source_tif=str(combined))
        print(f"Wrote volume norm bounds CSV: {csv_path}")
    if not csv_path.is_file():
        print(
            f"ERROR: volume normalization CSV not found: {csv_path}\n"
            "  Run once per sample (loads the combined TIFF only once):\n"
            f"    python features/compute_dinov2_volume_norm_csv.py "
            f"--output-dir {output_dir}\n"
            "  Or pass --recompute-volume-norm to extract_dinov2_embeddings.py "
            "(reads TIFF once, writes the CSV, then runs extraction)."
        )
        raise FileNotFoundError(str(csv_path))
    return read_bounds_csv(csv_path)


def extract_one_cell(
    crop_path: Path,
    encoder: DinoV2Encoder,
    cfg: DinoV2Config,
    *,
    volume_lo_hi: tuple[np.ndarray, np.ndarray] | None = None,
) -> CellResult | None:
    """Run DINOv2 on one ``cell_<NNNN>.tif`` (z-mean or orthogonal_concat)."""
    cell_id = parse_cell_id(crop_path.name)
    if cell_id is None:
        logger.warning("Skipping %s: cannot parse cell id", crop_path.name)
        return None
    try:
        intensity, mask = read_cell_crop(crop_path)
    except Exception as exc:
        logger.warning("Skipping %s: %s", crop_path.name, exc)
        return None

    z, _, y, x = intensity.shape
    vlohi = volume_lo_hi if cfg.norm_scope == "volume" else None

    if cfg.extraction_mode == "orthogonal_concat":
        if mask is None:
            mask = np.any(intensity > 0, axis=1)
        cz, cy, cx = compute_robust_centroid(
            mask,
            intensity,
            core_quantile=cfg.centroid_core_quantile,
            use_intensity_weights=cfg.centroid_use_intensity_weights,
        )
        planes, mask_planes = slice_orthogonal_planes(
            intensity, mask, cz, cy, cx, apply_mask=cfg.apply_mask
        )
        batch = prepare_planes_batch(
            planes,
            mask_planes,
            p_low=cfg.norm_p_low,
            p_high=cfg.norm_p_high,
            target_size=cfg.target_size,
            apply_mask=cfg.apply_mask,
            use_imagenet_stats=cfg.use_imagenet_stats,
            empty_slice_nonzero_frac=cfg.empty_slice_nonzero_frac,
            device=encoder.device,
            volume_lo_hi=vlohi,
        )
        if batch is None:
            logger.info("Skipping %s: an orthogonal plane was empty", crop_path.name)
            return None
        slice_embeds = encoder.encode_slices(batch)
        pooled = (
            torch.cat([slice_embeds[0], slice_embeds[1], slice_embeds[2]], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        per_slice = None
        if cfg.save_slice_embeddings:
            per_slice = slice_embeds.detach().cpu().numpy().astype(np.float32, copy=False)
        return CellResult(
            cell_id=cell_id,
            embedding=pooled,
            n_valid_slices=3,
            valid_indices=[0, 1, 2],
            slice_embeddings=per_slice,
            original_shape=(int(z), int(y), int(x)),
            centroid_zyx=(cz, cy, cx),
        )

    batch, valid_indices = prepare_batch(
        intensity,
        mask,
        p_low=cfg.norm_p_low,
        p_high=cfg.norm_p_high,
        target_size=cfg.target_size,
        apply_mask=cfg.apply_mask,
        use_imagenet_stats=cfg.use_imagenet_stats,
        empty_slice_nonzero_frac=cfg.empty_slice_nonzero_frac,
        device=encoder.device,
        volume_lo_hi=vlohi,
    )
    if batch.shape[0] == 0:
        logger.info("Skipping %s: every z-slice filtered as empty", crop_path.name)
        return None

    chunks: list[torch.Tensor] = []
    for start in range(0, batch.shape[0], cfg.batch_size):
        end = min(start + cfg.batch_size, batch.shape[0])
        chunks.append(encoder.encode_slices(batch[start:end]))
    slice_embeds = torch.cat(chunks, dim=0)
    pooled = mean_pool(slice_embeds).detach().cpu().numpy().astype(np.float32, copy=False)

    per_slice = None
    if cfg.save_slice_embeddings:
        per_slice = slice_embeds.detach().cpu().numpy().astype(np.float32, copy=False)

    return CellResult(
        cell_id=cell_id,
        embedding=pooled,
        n_valid_slices=int(slice_embeds.shape[0]),
        valid_indices=valid_indices,
        slice_embeddings=per_slice,
        original_shape=(int(z), int(y), int(x)),
        centroid_zyx=None,
    )


def _write_csv(
    out_csv: Path,
    rows: list[dict],
    cfg: DinoV2Config,
    embed_dim: int,
) -> None:
    fieldnames = [
        "cell_id",
        "embedding_dim",
        "n_valid_slices",
        "original_z",
        "original_y",
        "original_x",
        "model",
        "target_size",
        "apply_mask",
        "use_imagenet_stats",
        "norm_p_low",
        "norm_p_high",
        "empty_slice_nonzero_frac",
        "extraction_mode",
        "norm_scope",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    ]
    tmp = str(out_csv) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            cz = r.get("centroid_zyx")
            writer.writerow(
                {
                    "cell_id": r["cell_id"],
                    "embedding_dim": embed_dim,
                    "n_valid_slices": r["n_valid_slices"],
                    "original_z": r["original_shape"][0],
                    "original_y": r["original_shape"][1],
                    "original_x": r["original_shape"][2],
                    "model": cfg.model_name,
                    "target_size": cfg.target_size,
                    "apply_mask": int(cfg.apply_mask),
                    "use_imagenet_stats": int(cfg.use_imagenet_stats),
                    "norm_p_low": cfg.norm_p_low,
                    "norm_p_high": cfg.norm_p_high,
                    "empty_slice_nonzero_frac": cfg.empty_slice_nonzero_frac,
                    "extraction_mode": cfg.extraction_mode,
                    "norm_scope": cfg.norm_scope,
                    "centroid_z": "" if cz is None else f"{cz[0]:.6g}",
                    "centroid_y": "" if cz is None else f"{cz[1]:.6g}",
                    "centroid_x": "" if cz is None else f"{cz[2]:.6g}",
                }
            )
    Path(tmp).replace(out_csv)


def _write_summary_log(
    log_path: Path,
    cfg: DinoV2Config,
    n_total: int,
    n_processed: int,
    n_failed: int,
    avg_valid_slices: float,
    elapsed_s: float,
    *,
    cell_boxing_dirname: str,
    cell_qc_dirname: str,
) -> None:
    log_path.write_text(
        "\n".join(
            [
                "DINOv2 extraction summary",
                f"  model:                {cfg.model_name}",
                f"  device:               {cfg.device}",
                f"  extraction_mode:      {cfg.extraction_mode}",
                f"  norm_scope:           {cfg.norm_scope}",
                f"  target_size:          {cfg.target_size}",
                f"  apply_mask:           {cfg.apply_mask}",
                f"  use_imagenet_stats:   {cfg.use_imagenet_stats}",
                f"  norm_pct:             [{cfg.norm_p_low}, {cfg.norm_p_high}]",
                f"  empty_slice_thresh:   {cfg.empty_slice_nonzero_frac}",
                f"  batch_size:           {cfg.batch_size}",
                f"  cell_boxing_dir:      {cell_boxing_dirname}",
                f"  cell_qc_dir:          {cell_qc_dirname}",
                "",
                f"  cells discovered:     {n_total}",
                f"  cells processed:      {n_processed}",
                f"  cells failed/empty:   {n_failed}",
                f"  avg valid slices:     {avg_valid_slices:.2f}",
                f"  elapsed (s):          {elapsed_s:.1f}",
                "",
            ]
        )
    )


def extract_sample(
    output_dir: Path,
    cfg: DinoV2Config,
    *,
    encoder: DinoV2Encoder | None = None,
    force: bool = False,
    recompute_volume_norm: bool = False,
    volume_bounds_csv: Path | None = None,
    max_cells: int | None = None,
    only_cell_ids: set[int] | None = None,
    cell_boxing_dirname: str = "cell_boxing",
    cell_qc_dirname: str = "cell_qc",
    combined_filename: str = "filtered_642_combined.tif",
    bounds_csv_filename: str | None = None,
) -> int:
    """Process cell crops under ``<output_dir>/<cell_boxing_dirname>/``."""
    box_dir = output_dir / cell_boxing_dirname
    qc_dir = output_dir / cell_qc_dirname
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_npy = qc_dir / EMBEDDINGS_NPY
    out_csv = qc_dir / EMBEDDINGS_CSV
    out_log = qc_dir / EXTRACT_LOG
    out_per_slice = qc_dir / PER_SLICE_NPZ

    if not box_dir.is_dir():
        print(f"ERROR: cell crop directory not found: {box_dir}")
        return 1

    if out_npy.exists() and out_csv.exists() and not force:
        print(f"SKIP (exists): {out_npy}  (use --force to overwrite)")
        return 0

    cell_files = list_cell_crops(box_dir)
    if only_cell_ids is not None:
        cell_files = [p for p in cell_files if parse_cell_id(p.name) in only_cell_ids]
    if max_cells is not None and max_cells >= 0:
        cell_files = cell_files[:max_cells]
    if not cell_files:
        print(f"ERROR: no cell_*.tif files in {box_dir} (after filters)")
        return 1

    try:
        volume_lo_hi = _volume_lo_hi_for_sample(
            output_dir,
            cfg,
            recompute_volume_norm=recompute_volume_norm,
            volume_bounds_csv=volume_bounds_csv,
            combined_filename=combined_filename,
            bounds_csv_filename=bounds_csv_filename,
        )
    except FileNotFoundError:
        return 1

    if encoder is None:
        encoder = DinoV2Encoder(cfg)

    embeddings: list[np.ndarray] = []
    csv_rows: list[dict] = []
    per_slice_dump: dict[str, np.ndarray] = {}
    failed = 0
    valid_slice_counts: list[int] = []

    t0 = time.time()
    for idx, crop_path in enumerate(cell_files):
        result = extract_one_cell(crop_path, encoder, cfg, volume_lo_hi=volume_lo_hi)
        if result is None:
            failed += 1
            continue
        embeddings.append(result.embedding)
        csv_rows.append(
            {
                "cell_id": result.cell_id,
                "n_valid_slices": result.n_valid_slices,
                "original_shape": result.original_shape,
                "centroid_zyx": result.centroid_zyx,
            }
        )
        valid_slice_counts.append(result.n_valid_slices)
        if cfg.save_slice_embeddings and result.slice_embeddings is not None:
            per_slice_dump[f"cell_{result.cell_id:04d}"] = result.slice_embeddings
        if (idx + 1) % 50 == 0 or (idx + 1) == len(cell_files):
            print(f"  [{idx + 1}/{len(cell_files)}] processed (failed={failed})")

    if not embeddings:
        print("No cells produced an embedding.")
        return 1

    arr = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    embed_dim = int(arr.shape[1])
    np.save(str(out_npy), arr)
    _write_csv(out_csv, csv_rows, cfg, embed_dim)

    if cfg.save_slice_embeddings and per_slice_dump:
        np.savez_compressed(str(out_per_slice), **per_slice_dump)

    avg_valid = float(np.mean(valid_slice_counts)) if valid_slice_counts else 0.0
    elapsed = time.time() - t0
    _write_summary_log(
        out_log,
        cfg,
        len(cell_files),
        len(embeddings),
        failed,
        avg_valid,
        elapsed,
        cell_boxing_dirname=cell_boxing_dirname,
        cell_qc_dirname=cell_qc_dirname,
    )

    print(
        f"\nWrote {out_npy}  shape={arr.shape}\n"
        f"      {out_csv}  ({len(csv_rows)} rows)\n"
        f"      {out_log}\n"
        f"  processed={len(embeddings)}  failed={failed}  "
        f"avg_valid_slices={avg_valid:.2f}  elapsed={elapsed:.1f}s"
    )
    return 0
