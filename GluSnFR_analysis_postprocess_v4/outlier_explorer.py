"""
Outlier Explorer – Interactive histogram-based outlier flagging for iGluSnFR.

Opens a Qt dialog with one histogram per metric. Each histogram has a
RangeSlider that defines the *acceptable* range.  ROIs whose metric value
falls outside the range on **any** enabled metric are flagged.

Usage (from viewer.py)::

    explorer = OutlierExplorer(roi_metrics_df, roi_ids, roi_centroids,
                               napari_viewer_instance)
    explorer.show()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
    QPushButton, QSizePolicy, QWidget, QGridLayout,
)
from qtpy.QtCore import Qt, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RangeSlider

# Metric display labels  (source_col -> friendly name)
METRIC_LABELS = {
    "baseline_median": "Baseline Median",
    "baseline_sd":     "Baseline SD",
    "max_signal":      "Max Signal",
    "total_firings":   "Firing Count",
    "area":            "Area",
    "solidity":        "Solidity",
    "circularity":     "Circularity",
}

METRIC_ORDER = list(METRIC_LABELS.keys())


class OutlierExplorer(QDialog):
    """Interactive histogram dialog for outlier range selection.

    Parameters
    ----------
    roi_df : pd.DataFrame
        Must contain columns for each metric in ``METRIC_ORDER`` and a
        ``label`` column identifying the ROI.
    apply_callback : callable(set)
        Called with a *set* of flagged ROI label IDs when the user
        clicks **Apply to Viewer**.
    parent : QWidget or None
    """

    # Signal emitted with set of flagged ROI labels
    flags_changed = Signal(object)

    def __init__(self, roi_df: pd.DataFrame, apply_callback=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Outlier Explorer")
        self.setMinimumSize(820, 520)
        self.resize(960, 640)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self._roi_df = roi_df.copy()
        self._apply_callback = apply_callback

        # Per-metric state: checkbox enabled, slider range, axes, bars, slider
        self._metric_state: dict[str, dict] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # --- Checkbox row ---
        cb_layout = QHBoxLayout()
        cb_layout.setSpacing(10)
        self._checkboxes: dict[str, QCheckBox] = {}
        for metric in METRIC_ORDER:
            cb = QCheckBox(METRIC_LABELS[metric])
            cb.setChecked(metric != "max_signal")  # off by default for max_signal
            cb.stateChanged.connect(self._on_checkbox_changed)
            cb_layout.addWidget(cb)
            self._checkboxes[metric] = cb
        cb_layout.addStretch()
        outer.addLayout(cb_layout)

        # --- Matplotlib figure ---
        self._fig = Figure(figsize=(9, 5), dpi=100, facecolor="#2b2b2b")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self._canvas, stretch=1)

        # --- Bottom row: status + apply ---
        bottom = QHBoxLayout()
        self._status_label = QLabel("Flagged: 0 / 0 ROIs")
        self._status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        bottom.addWidget(self._status_label)
        bottom.addStretch()

        apply_btn = QPushButton("Apply to Viewer")
        apply_btn.setMinimumWidth(130)
        apply_btn.clicked.connect(self._on_apply)
        bottom.addWidget(apply_btn)
        outer.addLayout(bottom)

        self._draw_histograms()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_histograms(self):
        """Create histogram subplots with RangeSliders for enabled metrics."""
        self._fig.clear()
        self._metric_state.clear()

        enabled = [m for m in METRIC_ORDER if self._checkboxes[m].isChecked()]
        n = len(enabled)
        if n == 0:
            self._canvas.draw_idle()
            self._update_status()
            return

        # Grid: up to 4 columns, rows as needed. Each metric needs 2 rows
        # (histogram + slider).
        ncols = min(n, 4)
        nrows = -(-n // ncols)  # ceil division

        # gridspec: each metric block = 4 rows hist + 1 row slider
        gs = self._fig.add_gridspec(
            nrows * 5, ncols, hspace=0.6, wspace=0.35,
            left=0.06, right=0.97, top=0.96, bottom=0.04,
        )

        for idx, metric in enumerate(enabled):
            col = idx % ncols
            row_base = (idx // ncols) * 5

            ax_hist = self._fig.add_subplot(gs[row_base:row_base + 3, col])
            ax_slider = self._fig.add_subplot(gs[row_base + 3:row_base + 5, col])

            vals = pd.to_numeric(self._roi_df.get(metric, pd.Series(dtype=float)),
                                 errors="coerce").dropna().values
            if len(vals) == 0:
                ax_hist.set_title(METRIC_LABELS[metric] + " (no data)",
                                  fontsize=9, color="white")
                ax_hist.set_facecolor("#2b2b2b")
                ax_slider.set_visible(False)
                self._metric_state[metric] = {
                    "enabled": True, "vals": vals,
                    "range": (0.0, 0.0), "ax_hist": ax_hist,
                    "ax_slider": ax_slider, "bars": None, "slider": None,
                }
                continue

            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            # Guard against zero range
            if vmax - vmin < 1e-12:
                vmax = vmin + 1.0

            # Add small padding to range so slider can cover all data
            pad = (vmax - vmin) * 0.02
            smin, smax = vmin - pad, vmax + pad

            nbins = min(50, max(10, len(vals) // 5))
            counts, edges, bars = ax_hist.hist(
                vals, bins=nbins, color="#4a90d9", edgecolor="#333",
                linewidth=0.5)

            ax_hist.set_title(METRIC_LABELS[metric], fontsize=9,
                              color="white", pad=3)
            ax_hist.tick_params(labelsize=7, colors="white")
            ax_hist.set_facecolor("#2b2b2b")
            for spine in ax_hist.spines.values():
                spine.set_color("#555")

            # Vertical lines for range boundaries
            lo_line = ax_hist.axvline(smin, color="#ff4444", ls="--", lw=1.2)
            hi_line = ax_hist.axvline(smax, color="#ff4444", ls="--", lw=1.2)

            # Slider axes styling
            ax_slider.set_facecolor("#2b2b2b")
            for spine in ax_slider.spines.values():
                spine.set_color("#555")
            ax_slider.tick_params(labelsize=7, colors="white")

            slider = RangeSlider(
                ax_slider, "", smin, smax,
                valinit=(smin, smax),
                valstep=(smax - smin) / 500,
                color="#4a90d9",
            )
            # Style the slider label (the value text)
            slider.valtext.set_fontsize(7)
            slider.valtext.set_color("white")
            slider.label.set_color("white")

            self._metric_state[metric] = {
                "enabled": True,
                "vals": vals,
                "range": (smin, smax),
                "data_range": (vmin, vmax),
                "ax_hist": ax_hist,
                "ax_slider": ax_slider,
                "bars": bars,
                "slider": slider,
                "lo_line": lo_line,
                "hi_line": hi_line,
                "edges": edges,
            }

            # Connect slider callback (closure over metric name)
            slider.on_changed(self._make_slider_callback(metric))

        self._canvas.draw_idle()
        self._update_status()

    def _make_slider_callback(self, metric: str):
        """Return a closure that handles slider value changes."""
        def _on_slider(val):
            lo, hi = val
            state = self._metric_state.get(metric)
            if state is None:
                return
            state["range"] = (lo, hi)

            # Move vertical lines
            state["lo_line"].set_xdata([lo, lo])
            state["hi_line"].set_xdata([hi, hi])

            # Recolor bars: outside range = red, inside = blue
            bars = state["bars"]
            edges = state["edges"]
            if bars is not None and edges is not None:
                for bar, left_edge, right_edge in zip(
                        bars, edges[:-1], edges[1:]):
                    mid = (left_edge + right_edge) / 2
                    if mid < lo or mid > hi:
                        bar.set_facecolor("#e74c3c")
                    else:
                        bar.set_facecolor("#4a90d9")

            self._canvas.draw_idle()
            self._update_status()
        return _on_slider

    # ------------------------------------------------------------------
    # Flagging logic
    # ------------------------------------------------------------------
    def _get_flagged_labels(self) -> set:
        """Return set of ROI labels that fall outside any enabled range."""
        flagged = set()
        df = self._roi_df

        for metric, state in self._metric_state.items():
            if not state["enabled"]:
                continue
            lo, hi = state["range"]
            if metric not in df.columns:
                continue
            vals = pd.to_numeric(df[metric], errors="coerce")
            mask = (vals < lo) | (vals > hi)
            flagged.update(df.loc[mask, "label"].tolist())

        return flagged

    def _update_status(self):
        flagged = self._get_flagged_labels()
        total = len(self._roi_df)
        self._status_label.setText(
            f"Flagged: {len(flagged)} / {total} ROIs")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_checkbox_changed(self, _state):
        """Rebuild histograms when metric selection changes."""
        self._draw_histograms()

    def _on_apply(self):
        """Send flagged set to the viewer."""
        flagged = self._get_flagged_labels()
        if self._apply_callback is not None:
            self._apply_callback(flagged)
        self.flags_changed.emit(flagged)
