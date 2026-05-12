"""Shared helpers to discover and load per-cell DINOv2 embeddings from disk.

Pooling means **vertical concatenation** of per-sample ``(n_i, D)`` arrays into
one ``(N, D)`` matrix with ``N = sum_i n_i`` — each row is still **one cell**,
not an aggregate across cells.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMB_NPY_NAME = "dinov2_embeddings.npy"
EMB_CSV_NAME = "dinov2_embeddings.csv"
QC_FILTERED_CSV_NAME = "qc_features_filtered.csv"


def discover_embeddings_under_qc_dir(
    base: Path,
    cell_qc_dirname: str = "cell_qc",
) -> list[tuple[str, Path, Path]]:
    """Find ``(sample, npy_path, csv_path)`` triplets under ``base``.

    ``sample`` is the immediate parent folder of ``cell_qc_dirname`` (the
    sample name as used elsewhere in the pipeline).
    """
    results: list[tuple[str, Path, Path]] = []
    pattern = f"{cell_qc_dirname}/{EMB_NPY_NAME}"
    for npy_path in sorted(base.rglob(pattern)):
        csv_path = npy_path.with_name(EMB_CSV_NAME)
        if not csv_path.exists():
            logger.warning("Missing CSV next to %s; skipping", npy_path)
            continue
        sample_name = npy_path.parent.parent.name
        results.append((sample_name, npy_path, csv_path))
    return results


def discover_embeddings_multi_dataset(
    output_root: Path,
    dataset_specs: Sequence[tuple[str, str]],
) -> list[tuple[str, Path, Path, str]]:
    """Find embeddings under several dataset roots with different QC dir names.

    Args:
        output_root: e.g. ``<cmap>/output``.
        dataset_specs: sequence of ``(dataset_relpath, cell_qc_dirname)`` such as
            ``("4_18_25", "cell_qc_filtered")``, ``("4_24_25_CGN_6_10_2", "cell_qc")``.

    Returns:
        List of ``(sample, npy_path, csv_path, dataset)`` where ``dataset`` is
        ``dataset_relpath`` and ``sample`` is the sample folder name (unique
        with ``dataset`` for joins).
    """
    out: list[tuple[str, Path, Path, str]] = []
    for dataset_relpath, qc_dir in dataset_specs:
        ds_root = output_root / dataset_relpath
        if not ds_root.is_dir():
            logger.warning("Dataset root missing, skipping: %s", ds_root)
            continue
        pattern = f"{qc_dir}/{EMB_NPY_NAME}"
        for npy_path in sorted(ds_root.rglob(pattern)):
            csv_path = npy_path.with_name(EMB_CSV_NAME)
            if not csv_path.exists():
                logger.warning("Missing CSV next to %s; skipping", npy_path)
                continue
            sample_name = npy_path.parent.parent.name
            out.append((sample_name, npy_path, csv_path, dataset_relpath))
    return out


def load_pool_from_triplets(
    triplets: list[tuple[str, Path, Path]] | list[tuple[str, Path, Path, str]],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate per-sample matrices into ``(N, D)`` + metadata DataFrame."""
    arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    for item in triplets:
        if len(item) == 4:
            sample_name, npy_path, csv_path, dataset = item
        else:
            sample_name, npy_path, csv_path = item  # type: ignore[misc]
            dataset = None
        emb = np.load(str(npy_path))
        df = pd.read_csv(csv_path)
        if emb.shape[0] != len(df):
            logger.warning(
                "Row count mismatch for %s: %s rows in CSV vs %s in NPY -- skipping",
                sample_name,
                len(df),
                emb.shape[0],
            )
            continue
        df = df.copy()
        df.insert(0, "sample", sample_name)
        if dataset is not None:
            df.insert(1, "dataset", dataset)
        bg_csv = npy_path.parent / "bg_stats.csv"
        if bg_csv.is_file():
            bg_df = pd.read_csv(bg_csv)
            if len(bg_df) == 1:
                for col in bg_df.columns:
                    df[col] = bg_df.iloc[0][col]
        arrays.append(emb)
        frames.append(df)
        print(f"  {dataset + '/' if dataset else ''}{sample_name}: {emb.shape}")
    if not arrays:
        raise RuntimeError("No DINOv2 embeddings found to pool")
    pooled = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    pooled_df = pd.concat(frames, ignore_index=True)
    print(f"\nPooled: {pooled.shape} from {len(frames)} samples")
    return pooled, pooled_df


def apply_qc_filter(
    pooled: np.ndarray,
    pooled_df: pd.DataFrame,
    output_root: Path,
    filter_col: str,
    *,
    qc_csv_parent_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Keep rows whose ``qc_features_filtered.csv`` row has ``filter_col == 1``.

    Discovers ``qc_features_filtered.csv`` under ``output_root`` and keeps only
    files whose immediate parent directory name is in ``qc_csv_parent_names``.
    If ``qc_csv_parent_names`` is ``None`` and ``pooled_df`` has a ``dataset``
    column, defaults to ``("cell_qc", "cell_qc_filtered")`` so mixed 4_18 /
    4_24 style trees resolve. If there is no ``dataset`` column, defaults to
    ``("cell_qc",)``.
    """
    if qc_csv_parent_names is None:
        if "dataset" in pooled_df.columns:
            qc_csv_parent_names = ("cell_qc", "cell_qc_filtered")
        else:
            qc_csv_parent_names = ("cell_qc",)

    qc_pairs: list[Path] = []
    for qc_csv in sorted(output_root.rglob(QC_FILTERED_CSV_NAME)):
        if qc_csv.parent.name in qc_csv_parent_names:
            qc_pairs.append(qc_csv)
    if not qc_pairs:
        raise RuntimeError(
            f"--filter-col was set but no {QC_FILTERED_CSV_NAME} files were found under {output_root}"
        )
    keep_keys: set[tuple[str, int]] = set()
    for qc_csv in qc_pairs:
        sample_dir = qc_csv.parent.parent
        sample_name = sample_dir.name
        dataset_dir = sample_dir.parent.name
        df = pd.read_csv(qc_csv)
        if filter_col not in df.columns:
            logger.warning("  %s: missing column '%s' -- skipping", qc_csv, filter_col)
            continue
        passed = df[df[filter_col] == 1]
        for cid in passed["cell_id"].astype(int):
            if "dataset" in pooled_df.columns:
                keep_keys.add((dataset_dir, sample_name, int(cid)))
            else:
                keep_keys.add((sample_name, int(cid)))

    if "dataset" in pooled_df.columns:
        sample_arr = pooled_df["dataset"].astype(str).to_numpy()
        sample2 = pooled_df["sample"].astype(str).to_numpy()
        cellid_arr = pooled_df["cell_id"].astype(int).to_numpy()
        mask = np.array(
            [(d, s, c) in keep_keys for d, s, c in zip(sample_arr, sample2, cellid_arr)],
            dtype=bool,
        )
    else:
        sample_arr = pooled_df["sample"].astype(str).to_numpy()
        cellid_arr = pooled_df["cell_id"].astype(int).to_numpy()
        mask = np.array(
            [(s, c) in keep_keys for s, c in zip(sample_arr, cellid_arr)],
            dtype=bool,
        )
    n_keep = int(mask.sum())
    if n_keep == 0:
        raise RuntimeError(f"No cells survived the {filter_col} filter")
    print(f"QC filter '{filter_col}': kept {n_keep}/{len(pooled_df)} cells")
    return pooled[mask], pooled_df.loc[mask].reset_index(drop=True)
