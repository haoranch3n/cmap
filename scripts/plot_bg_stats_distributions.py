#!/usr/bin/env python3
"""Plot distributions of background stats across volumes using only ``bg_stats.csv``.

Each sample directory under ``--dataset-root`` should contain::

    cell_qc_union_488_560/bg_stats.csv

(one row per volume, columns ``bg_mean_*`` / ``bg_std_*`` from ``qc/filter_by_intensity.py``).

**Default:** one PNG **per** ``--dataset-root`` (e.g. all 4_18 samples in one figure, all 4_24 in
another). Use ``--pool`` to merge multiple roots into a single figure.

This is **no TIFF I/O** — only CSV.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_CHANNEL_ORDER = ["642", "488", "560"]
_COLORS = {"642": "#8C2318", "488": "#2B6A9E", "560": "#2D8F4E"}


def _sample_dirs(dataset_root: Path) -> list[Path]:
    """Immediate child dirs (sample folders), skipping artifact dirs like ``_lsf_logs``."""
    return sorted(
        p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    )


def _discover_bg_csvs(dataset_root: Path, qc_leaf: Path) -> list[Path]:
    out: list[Path] = []
    for rd in _sample_dirs(dataset_root):
        p = rd / qc_leaf / "bg_stats.csv"
        if p.is_file():
            out.append(p)
    return sorted(out)


def _first_data_row(csv_path: Path, prefix: str) -> dict[str, str] | None:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row0 = next(reader, None)
    if row0 is None:
        print(f"SKIP (empty CSV): {csv_path}", file=sys.stderr)
        return None
    miss = [
        ch
        for ch in _CHANNEL_ORDER
        if row0.get(f"{prefix}_{ch}") in (None, "")
    ]
    if miss:
        print(f"SKIP ({prefix}_* missing): {csv_path}", file=sys.stderr)
        return None
    return row0


def _build_paired(csv_paths: list[Path], prefix: str) -> list[tuple[Path, dict[str, str]]]:
    paired: list[tuple[Path, dict[str, str]]] = []
    for p in csv_paths:
        row = _first_data_row(p, prefix)
        if row is not None:
            paired.append((p, row))
    return paired


def _figure_for_paired(
    paired: list[tuple[Path, dict[str, str]]],
    *,
    metric: str,
    prefix: str,
    qc_leaf: Path,
    title_suffix: str,
    subtitle_suffix: str,
) -> plt.Figure:
    pulls: dict[str, list[float]] = {ch: [] for ch in _CHANNEL_ORDER}
    for _csv_path, row in paired:
        for ch in _CHANNEL_ORDER:
            pulls[ch].append(float(row[f"{prefix}_{ch}"]))

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for nm in _CHANNEL_ORDER:
        v = np.asarray(pulls[nm], dtype=np.float64)
        if v.size == 0:
            continue
        n_img = int(v.size)
        w = np.full(n_img, 1.0 / n_img, dtype=np.float64)
        ax.hist(
            v,
            bins=min(36, max(8, int(np.ceil(np.sqrt(n_img))))),
            weights=w,
            density=False,
            histtype="stepfilled",
            alpha=0.45,
            label=f"{nm} nm (n={n_img})",
            color=_COLORS[nm],
        )

    xlab = "intensity mean" if metric == "mean" else "intensity std"
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel("frequency (# images in bin / total # images)", fontsize=10)
    ax.set_title(f"Per-channel BG stats ({title_suffix})", fontsize=11)
    ax.legend(loc="best", framealpha=0.93)
    subtitle = subtitle_suffix + f" • qc={qc_leaf.as_posix()}"
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9)
    return fig


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    default_roots = (
        repo / "output" / "4_18_25",
        repo / "output" / "4_24_25_CGN_6_10_2",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-root",
        nargs="+",
        type=Path,
        default=list(default_roots),
        metavar="DIR",
    )
    ap.add_argument(
        "--qc-subdir",
        type=Path,
        default=Path("cell_qc_union_488_560"),
        help="QC folder beneath each sample (contains bg_stats.csv)",
    )
    ap.add_argument("--min-samples-per-channel", type=int, default=3)
    ap.add_argument(
        "--metric",
        choices=("mean", "std"),
        default="mean",
        help="Plot bg_mean_* (default) or bg_std_* per channel.",
    )
    ap.add_argument(
        "--pool",
        action="store_true",
        help="Merge all dataset roots into one figure instead of one figure per root.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG — only allowed with a single --dataset-root unless --pool (see --out-dir).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNGs when multiple --dataset-root and not --pool (default: cmap/output/).",
    )
    args = ap.parse_args()

    roots = [p.resolve() for p in args.dataset_root]
    prefix = "bg_mean" if args.metric == "mean" else "bg_std"
    repo_out = repo / "output"

    err = _run_all(
        repo_out=repo_out,
        roots=roots,
        qc_leaf=args.qc_subdir,
        prefix=prefix,
        metric=args.metric,
        pool=args.pool,
        min_samples=args.min_samples_per_channel,
        out=args.out,
        out_dir=args.out_dir,
    )
    return err


def _run_all(
    *,
    repo_out: Path,
    roots: list[Path],
    qc_leaf: Path,
    prefix: str,
    metric: str,
    pool: bool,
    min_samples: int,
    out: Path | None,
    out_dir: Path | None,
) -> int:
    if pool:
        all_paths: list[Path] = []
        for root in roots:
            all_paths.extend(_discover_bg_csvs(root, qc_leaf))
        paired = _build_paired(all_paths, prefix)
        if len(paired) < min_samples:
            print(f"POOL: only {len(paired)} usable bg_stats CSV(s).", file=sys.stderr)
            return 1

        pooled_stub = "__".join(r.name for r in roots)
        tgt = (
            out
            if out is not None
            else repo_out / f"bg_stats_{metric}_distributions_pooled__{pooled_stub}.png"
        )
        subtitle = "__".join(r.name for r in roots) + f" • {len(paired)} volumes (pooled)"
        fig = _figure_for_paired(
            paired,
            metric=metric,
            prefix=prefix,
            qc_leaf=qc_leaf,
            title_suffix=f"{metric}: pooled datasets — CSV only",
            subtitle_suffix=subtitle,
        )
        tgt.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(tgt, dpi=150)
        plt.close(fig)
        print(f"wrote {tgt}  (pooled n_csv={len(paired)})")
        return 0

    # Separate figure per dataset — every sample folder is enumerated; CSV gaps are reported.
    if out is not None and len(roots) > 1:
        print("ERROR: use --out-dir (or omit --out) when passing multiple --dataset-root.", file=sys.stderr)
        return 2

    base_dir = repo_out if out_dir is None else Path(out_dir)
    exit_code = 0
    for root in roots:
        sample_paths = _sample_dirs(root)
        n_samples = len(sample_paths)
        csv_paths = _discover_bg_csvs(root, qc_leaf)
        missing_samples = sorted(
            s
            for s in sample_paths
            if not (s / qc_leaf / "bg_stats.csv").is_file()
        )

        paired = _build_paired(csv_paths, prefix)
        if len(paired) < min_samples:
            print(
                f"ERROR [{root.name}]: only {len(paired)} usable bg_stats CSV "
                f"(need ≥{min_samples}); sample dirs={n_samples} missing CSV={len(missing_samples)}.",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        subtitle = (
            f"{root.name} • CSV used={len(paired)} / sample folders={n_samples}"
            + (f" • missing CSV={len(missing_samples)}" if missing_samples else "")
        )
        if missing_samples:
            for sdir in missing_samples[:12]:
                print(
                    f"MISSING BG CSV [{root.name}] {sdir.name}/"
                    f"{qc_leaf.as_posix()}/bg_stats.csv",
                    file=sys.stderr,
                )
            if len(missing_samples) > 12:
                print(f"MISSING BG CSV [{root.name}] … ({len(missing_samples) - 12} more)", file=sys.stderr)

        if out is not None:
            tgt = Path(out).resolve()
        else:
            base_dir.mkdir(parents=True, exist_ok=True)
            tgt = base_dir / f"bg_stats_{metric}_distributions_{root.name}.png"

        fig = _figure_for_paired(
            paired,
            metric=metric,
            prefix=prefix,
            qc_leaf=qc_leaf,
            title_suffix=f"{metric}: {root.name} — CSV only",
            subtitle_suffix=subtitle,
        )
        tgt.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(tgt, dpi=150)
        plt.close(fig)
        print(f"wrote {tgt}")
        print(f"  [{root.name}] n_csv={len(paired)} sample_dirs={n_samples} missing_csv={len(missing_samples)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
