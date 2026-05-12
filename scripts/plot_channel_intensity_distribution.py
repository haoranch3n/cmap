#!/usr/bin/env python3
"""Pool subsampled voxel intensities from many combined TIFFs and plot distributions.

Loads ``(Z, C, Y, X)`` combined volumes slice-by-slice (no full-volume ``memmap``),
subsampling in Z and XY to reduce I/O. Reservoir-caps pooled pixels per channel.

By default aggregates **both** ``output/4_18_25`` and ``output/4_24_25_CGN_6_10_2``,
drawing **642 / 488 / 560 nm** as **distinct colors on one axes** using a robust
within-channel affine scale (so channels share one readable x-axis).

Use ``--separate-subplots`` for raw intensity with one panel per channel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

_CHANNEL_ORDER = ["642", "488", "560"]
_COLORS = {"642": "#8C2318", "488": "#2B6A9E", "560": "#2D8F4E"}


def _robust_affine01(x: np.ndarray, pct_lo: float = 1.0, pct_hi: float = 99.0) -> np.ndarray:
    """Linear map [p_lo,p_hi]→[0,1] for pooled overlay (visualization only)."""
    lo, hi = np.percentile(x, [pct_lo, pct_hi]).astype(np.float64)
    den = float(max(hi - lo, 1e-12))
    return np.clip((x.astype(np.float64) - lo) / den, -0.05, 1.05)


def _discover_paths(root: Path, filename: str) -> list[Path]:
    roots = sorted(p for p in root.iterdir() if p.is_dir())
    paths: list[Path] = []
    for rd in roots:
        cand = rd / filename
        if cand.is_file():
            paths.append(cand)
    return sorted(paths)


def _squash_chunks(rng: np.random.Generator, chunks: list[np.ndarray], cap: int) -> list[np.ndarray]:
    """Concatenate and subsample-without-replacement until total <= ``cap``."""
    if not chunks:
        return []
    cat = np.concatenate(chunks)
    if cat.size <= cap:
        return [cat]
    idx = rng.choice(cat.size, size=cap, replace=False)
    return [cat[idx]]


def _collect_samples(
    path: Path,
    *,
    z_stride: int,
    xy_stride: int,
    cap_per_ch: int,
    rng: np.random.Generator,
    chunk_lists: list[list[np.ndarray]],
    totals: list[int],
) -> None:
    with tf.TiffFile(path) as tif:
        s = tif.series[0]
        if len(s.shape) != 4:
            raise ValueError(f"{path}: expected ZCYX, got shape {s.shape}")
        nz, nc, ny, nx = (int(i) for i in s.shape)
        if nc < 3:
            raise ValueError(f"{path}: need C>=3, got C={nc}")
        for zi in range(0, nz, z_stride):
            for ci in range(3):
                if ci >= nc:
                    break
                k = zi * nc + ci
                plane = np.asarray(s.asarray(key=k), dtype=np.float64)
                sub = plane[::xy_stride, ::xy_stride].ravel()
                xv = sub[np.isfinite(sub)]
                if xv.size == 0:
                    continue
                chunk_lists[ci].append(xv)
                totals[ci] += xv.size
                # Periodically squash so pooled lists stay small.
                if totals[ci] > cap_per_ch * 3:
                    chunk_lists[ci] = _squash_chunks(rng, chunk_lists[ci], cap_per_ch)
                    totals[ci] = sum(c.size for c in chunk_lists[ci])


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
        help="One or more dataset roots (sample directories are immediate children)",
    )
    ap.add_argument(
        "--tif-name",
        default="union_488_560_combined.tif",
        help="Combined volume filename inside each sample directory",
    )
    ap.add_argument("--z-stride", type=int, default=6, help="Take every nth Z plane")
    ap.add_argument("--xy-stride", type=int, default=16, help="Subsampling along Y,X within each plane")
    ap.add_argument(
        "--max-per-channel",
        type=int,
        default=350_000,
        help="Reservoir capacity per channel across all pooled volumes",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Process only this many TIFFs after sorting (subset for probes)",
    )
    ap.add_argument(
        "--separate-subplots",
        action="store_true",
        help="Use three axes with raw intensities instead of one overlay (robust‑scaled)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <first-root>/channel_intensity_distributions_<tif-stem>.png)",
    )
    args = ap.parse_args()

    roots = [Path(p).resolve() for p in args.dataset_root]
    paths: list[tuple[Path, str]] = []
    for root in roots:
        ds_name = root.name
        for pth in _discover_paths(root, args.tif_name):
            paths.append((pth, ds_name))

    # Deterministic ordering: dataset name then sample path string
    paths.sort(key=lambda t: (t[1], str(t[0])))
    if args.max_volumes is not None:
        paths = paths[: max(0, args.max_volumes)]
    if not paths:
        for root in roots:
            print(f"  (nothing under {root}/*/{args.tif_name})", file=sys.stderr)
        print("No TIFFs found.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    cap = args.max_per_channel
    chunk_lists: list[list[np.ndarray]] = [[], [], []]
    totals = [0, 0, 0]

    for p, ds_label in paths:
        print(f"sampling … [{ds_label}] {p.parent.name}")
        _collect_samples(
            p,
            z_stride=max(1, args.z_stride),
            xy_stride=max(1, args.xy_stride),
            cap_per_ch=cap,
            rng=rng,
            chunk_lists=chunk_lists,
            totals=totals,
        )

    series: list[np.ndarray] = []
    for ci in range(3):
        squashed = _squash_chunks(rng, chunk_lists[ci], cap)
        series.append(np.concatenate(squashed) if squashed else np.empty(0, dtype=np.float64))

    out = args.out
    if out is None:
        safe = Path(args.tif_name).stem.replace(" ", "_")
        ds_tag = "_".join(r.name for r in roots)
        repo_out = repo / "output"
        base_dir = repo_out if repo_out.is_dir() else roots[0]
        out = base_dir / f"channel_intensity_distributions_{safe}_{ds_tag}.png"

    if args.separate_subplots:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        axes_list = list(axes)  # type: ignore[arg-type]
    else:
        fig, ax_overlay = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
        axes_list = []

    subtitle_ds = "+".join(r.name for r in roots)

    if args.separate_subplots:
        for axi, bi, nm in zip(axes_list, range(3), _CHANNEL_ORDER):
            x = series[bi]
            if x.size == 0:
                axi.set_visible(False)
                continue
            axi.hist(x, bins=120, density=True, color=_COLORS[nm], alpha=0.82)
            axi.set_title(f"{nm} nm (n={x.size:,})", fontsize=11)
            axi.set_xlabel("intensity (raw)")
            axi.set_ylabel("density")

        subtitle = (
            f"{subtitle_ds} • {len(paths)} volumes • "
            f"z_stride={args.z_stride} xy_stride={args.xy_stride} cap/channel={cap}"
        )
        fig.suptitle(subtitle, fontsize=10)
    else:
        ax_overlay.set_title(
            "Pooled voxel intensity — 642 / 488 / 560 nm (robust‑scaled per channel)",
            fontsize=11,
        )
        for bi, nm in enumerate(_CHANNEL_ORDER):
            x = series[bi]
            if x.size == 0:
                continue
            xs = _robust_affine01(x)
            ax_overlay.hist(xs, bins=100, density=True, color=_COLORS[nm], alpha=0.42, label=f"{nm} nm (n={x.size:,})")
        ax_overlay.set_xlim(-0.05, 1.05)
        ax_overlay.set_xlabel("within-channel percentile scale (≈ P1→0, P99→1)")
        ax_overlay.set_ylabel("density")
        ax_overlay.legend(loc="upper right", framealpha=0.92)

        subtitle = (
            f"Datasets: {subtitle_ds}\n"
            f"{len(paths)} volumes • z_stride={args.z_stride} xy_stride={args.xy_stride} "
            f"cap/channel={cap}"
        )
        fig.text(0.5, 0.03, subtitle, ha="center", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
