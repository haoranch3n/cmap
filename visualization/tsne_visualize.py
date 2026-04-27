#!/usr/bin/env python3
"""
Run t-SNE and UMAP on QC features and produce scatter-plot visualizations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

FEATURE_COLS = [
    "volume",
    "mean_642", "median_642", "std_642", "total_642", "cv_642", "pct95_642",
    "mean_488", "median_488", "std_488", "total_488", "cv_488", "pct95_488",
    "mean_560", "median_560", "std_560", "total_560", "cv_560", "pct95_560",
    "corr_642_488", "corr_642_560", "corr_488_560",
    "min_channel_mean", "mean_pairwise_corr",
]
FEATURE_COLS_NO_VOL = [c for c in FEATURE_COLS if c != "volume"]
COLOR_PANELS = [("mean_488", "Mean 488 Intensity", "viridis"), ("mean_560", "Mean 560 Intensity", "inferno")]


def _resolve_dirs(args) -> tuple[Path, Path]:
    if args.data_rel:
        data_dir = Path(DATA_DIR) / args.data_rel
        output_dir = (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
    elif args.data_dir and args.output_dir:
        data_dir = args.data_dir.resolve()
        output_dir = args.output_dir.resolve()
    else:
        raise SystemExit("Provide either --data-rel or both --data-dir and --output-dir")
    return data_dir, output_dir


def _rgb_colors(df: pd.DataFrame, col_g: str = "mean_488", col_m: str = "mean_560", clip_pct: float = 99) -> np.ndarray:
    g = np.clip(df[col_g].values, 0, np.percentile(df[col_g], clip_pct))
    m = np.clip(df[col_m].values, 0, np.percentile(df[col_m], clip_pct))
    g = g / g.max() if g.max() > 0 else g
    m = m / m.max() if m.max() > 0 else m
    return np.column_stack([m, g, m, np.full(len(g), 0.85)])


def _plot_embedding_grid(df: pd.DataFrame, coords: dict[str, tuple[str, str]], method_name: str, out_path: Path) -> None:
    panels = [(c, t, cm) for c, t, cm in COLOR_PANELS if c in df.columns]
    has_rgb = "mean_488" in df.columns and "mean_560" in df.columns
    ncols = len(panels) + (1 if has_rgb else 0)
    row_labels = list(coords.keys())
    nrows = len(row_labels)
    n_cells = len(df)
    point_size = max(15, min(80, 4000 // max(n_cells, 1)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]
    rgba = _rgb_colors(df) if has_rgb else None

    for ri, row_label in enumerate(row_labels):
        x_col, y_col = coords[row_label]
        col_idx = 0
        for col, title, cmap in panels:
            ax = axes[ri][col_idx]
            vals = df[col].values
            sc = ax.scatter(
                df[x_col], df[y_col], c=vals, cmap=cmap, s=point_size, alpha=0.85,
                edgecolors="k", linewidths=0.3, vmin=0, vmax=1000,
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8, extend="max")
            cbar.set_label(col)
            ax.set_title(f"{title}\n({row_label})", fontsize=13)
            ax.set_xlabel(f"{method_name} 1")
            ax.set_ylabel(f"{method_name} 2")
            col_idx += 1

        if has_rgb:
            ax = axes[ri][col_idx]
            ax.scatter(df[x_col], df[y_col], c=rgba, s=point_size, alpha=0.85, edgecolors="k", linewidths=0.3)
            ax.set_title(f"Combined 488+560\n({row_label})", fontsize=13)
            ax.set_xlabel(f"{method_name} 1")
            ax.set_ylabel(f"{method_name} 2")
            ax.annotate(
                "Green = 488\nMagenta = 560\nWhite = both high",
                xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

    fig.suptitle(f"Cell QC {method_name}  (n={n_cells})", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _compute_embedding(X_scaled, df, perplexity, method, suffix):
    n = len(df)
    x_col = f"{method}_1_{suffix}"
    y_col = f"{method}_2_{suffix}"
    if method == "tsne":
        eff_perp = min(perplexity, max(5.0, n / 4.0))
        model = TSNE(
            n_components=2, perplexity=eff_perp, random_state=42, init="pca", learning_rate="auto",
        )
    else:
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP is required. Install umap-learn.") from exc
        n_neighbors = min(15, max(2, n - 1))
        model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    emb = model.fit_transform(X_scaled)
    df[x_col] = emb[:, 0]
    df[y_col] = emb[:, 1]
    return x_col, y_col


def run(output_dir: Path, force: bool = False, perplexity: float = 30.0) -> int:
    qc_dir = output_dir / "cell_qc"
    csv_path = qc_dir / "qc_features.csv"
    if not csv_path.exists():
        print(f"ERROR: features CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    avail_vol = [c for c in FEATURE_COLS if c in df.columns]
    avail_novol = [c for c in FEATURE_COLS_NO_VOL if c in df.columns]
    if not avail_vol:
        print("ERROR: no recognized feature columns in CSV")
        return 1

    X_vol = df[avail_vol].values.astype(np.float64)
    X_novol = df[avail_novol].values.astype(np.float64)
    nan_mask = np.isnan(X_vol).any(axis=1) | np.isnan(X_novol).any(axis=1)
    if nan_mask.any():
        df = df[~nan_mask].reset_index(drop=True)
        X_vol = X_vol[~nan_mask]
        X_novol = X_novol[~nan_mask]

    X_vol_scaled = StandardScaler().fit_transform(X_vol)
    X_novol_scaled = StandardScaler().fit_transform(X_novol)
    for method in ("tsne", "umap"):
        label = "t-SNE" if method == "tsne" else "UMAP"
        xv, yv = _compute_embedding(X_vol_scaled, df, perplexity, method, "vol")
        xn, yn = _compute_embedding(X_novol_scaled, df, perplexity, method, "novol")
        coords = {"With Volume": (xv, yv), "Without Volume": (xn, yn)}
        png_path = qc_dir / f"{method}_qc.png"
        _plot_embedding_grid(df, coords, label, png_path)

    tsne_csv = qc_dir / "tsne_embedding.csv"
    df.to_csv(str(tsne_csv), index=False)
    umap_csv = qc_dir / "umap_embedding.csv"
    df.to_csv(str(umap_csv), index=False)
    print(f"Saved: {tsne_csv}")
    print(f"Saved: {umap_csv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    args = ap.parse_args()
    _, output_dir = _resolve_dirs(args)
    return run(output_dir, force=args.force, perplexity=args.perplexity)


if __name__ == "__main__":
    sys.exit(main())

