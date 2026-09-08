"""Napari dock widget for CMAP cell exemption review."""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .data_io import (
    annotation_csv_path,
    discover_samples,
    load_cell_boxes,
    load_combined_tiff,
    path_key,
    save_annotations_atomic,
)
from .geometry import cell_box_to_point, cell_box_to_slice_rectangle, corners_zyx_to_missing_box
from .models import (
    BAD_LABEL,
    EXISTING_CELL,
    GOOD_LABEL,
    MISSING_CELL,
    MISSING_LABEL,
    UNANNOTATED_LABEL,
    AnnotationRow,
    CellBoxRecord,
    SampleRecord,
)

try:
    from napari.viewer import Viewer
except Exception:  # pragma: no cover - typing fallback outside napari
    Viewer = object  # type: ignore[misc, assignment]


LAYER_642 = "CMAP 642 Original"
LAYER_488 = "CMAP 488 Original"
LAYER_560 = "CMAP 560 Original"
LAYER_LABELS = "CMAP Cell ID Labels"
LAYER_BOUNDARY = "CMAP Mask Boundary"
LAYER_CENTROIDS = "CMAP Cell Centers"
LAYER_SELECTED_MASK = "CMAP Selected Cell Mask"
LAYER_SELECTED_BOX = "CMAP Selected Cell Box"
LAYER_MISSING = "CMAP Missing Cell Boxes"

PLUGIN_LAYER_NAMES = {
    LAYER_642,
    LAYER_488,
    LAYER_560,
    LAYER_LABELS,
    LAYER_BOUNDARY,
    LAYER_CENTROIDS,
    LAYER_SELECTED_MASK,
    LAYER_SELECTED_BOX,
    LAYER_MISSING,
}

CHANNELS = [
    (0, LAYER_642, "blue", True),
    (1, LAYER_488, "green", True),
    (2, LAYER_560, "red", True),
]

STYLE = """
QGroupBox {
    font-weight: bold;
    padding-top: 10px;
    margin-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    padding: 3px 6px;
}
QPushButton {
    padding: 5px 10px;
}
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _shape_rgba(n: int, rgba: tuple[float, float, float, float]) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.tile(np.asarray(rgba, dtype=np.float32), (n, 1))


_TEXT_INPUT_TYPES = (QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QAbstractSpinBox)


class _HotkeyEventFilter(QObject):
    """Application-level key filter that owns the plugin's single-key hotkeys.

    napari binds some of our letters (B, M, D, P, ...) to built-in layer
    actions through its app-model keybinding registry, which would otherwise
    shadow viewer-level bindings. Intercepting key presses here — before
    napari's canvas sees them — guarantees our shortcuts win, while keys are
    left untouched whenever a text-entry widget has focus so the user can type
    normally.
    """

    def __init__(self, widget: "CellExemptionWidget") -> None:
        super().__init__(widget)
        self._widget = widget

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QEvent.KeyPress:
            try:
                if self._widget._handle_hotkey(event):
                    return True
            except Exception:
                return False
        return False


class CellExemptionWidget(QWidget):
    """Main plugin widget for existing-cell and missing-cell annotations."""

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        self.output_root: Path | None = None
        self.annotation_dir: Path | None = None
        self.annotation_csv: Path | None = None
        self.samples: list[SampleRecord] = []
        self.sample_index = 0
        self.current_sample: SampleRecord | None = None
        self.cell_boxes: list[CellBoxRecord] = []
        self.cell_box_by_id: dict[int, CellBoxRecord] = {}
        self.current_cell_id: int | None = None
        self.label_data: np.ndarray | None = None
        self.rows_by_key: dict[tuple[str, str, str], AnnotationRow] = {}
        self.dirty = False
        self._loading_ui = False
        self._suppress_missing_events = False
        self._missing_mode = False
        self._session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._about_to_quit_hooked = False

        self._labels_layer = None
        self._base_layer = None
        self._selected_mask_layer = None
        self._selected_box_layer = None
        self._missing_layer = None

        self._window_sized = False
        self._disabled_drag_to_zoom = None

        self.setStyleSheet(STYLE)
        self._build_ui()
        self._wire_shortcuts()
        self._hook_application_quit()

        # Viewer-level click handler so clicking any cell selects it even when
        # the active layer is not the segmentation labels layer.
        try:
            self.viewer.mouse_drag_callbacks.append(self._on_canvas_click)
        except Exception:
            pass

        # While drawing missing-cell boxes, napari auto-zooms the camera to the
        # new shape. Keep the view fixed by restoring the camera after the draw.
        try:
            self.viewer.mouse_drag_callbacks.append(self._lock_camera_during_missing_draw)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Wrap all controls in a scroll area so the dock panel never grows
        # taller than the monitor on small screens.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget()
        main = QVBoxLayout(content)

        paths_box = QGroupBox("Folders")
        paths_form = QFormLayout()

        self._root_edit = QLineEdit()
        self._root_btn = QPushButton("Open Output Root...")
        self._root_btn.clicked.connect(self._pick_output_root)
        root_row = QHBoxLayout()
        root_row.addWidget(self._root_edit)
        root_row.addWidget(self._root_btn)
        paths_form.addRow("CMAP output", root_row)

        self._ann_edit = QLineEdit()
        self._ann_btn = QPushButton("Annotation Output...")
        self._ann_btn.clicked.connect(self._pick_annotation_dir)
        ann_row = QHBoxLayout()
        ann_row.addWidget(self._ann_edit)
        ann_row.addWidget(self._ann_btn)
        paths_form.addRow("CSV output", ann_row)

        paths_box.setLayout(paths_form)

        nav_box = QGroupBox("Samples")
        nav_layout = QVBoxLayout()
        self._sample_combo = QComboBox()
        self._sample_combo.currentIndexChanged.connect(self._on_sample_combo_changed)
        nav_buttons = QHBoxLayout()
        self._prev_btn = QPushButton("[P] Previous")
        self._next_btn = QPushButton("[N] Next")
        self._prev_btn.clicked.connect(self._previous_sample)
        self._next_btn.clicked.connect(self._next_sample)
        nav_buttons.addWidget(self._prev_btn)
        nav_buttons.addWidget(self._next_btn)
        self._sample_info = QLabel("No sample loaded.")
        self._sample_info.setWordWrap(True)
        nav_layout.addWidget(self._sample_combo)
        nav_layout.addLayout(nav_buttons)
        nav_layout.addWidget(self._sample_info)
        nav_box.setLayout(nav_layout)

        cell_box = QGroupBox("Existing 3D Cell Crop")
        cell_layout = QVBoxLayout()
        cell_form = QFormLayout()
        self._cell_combo = QComboBox()
        self._cell_combo.currentIndexChanged.connect(self._on_cell_combo_changed)
        self._reviewer_edit = QLineEdit(os.environ.get("USER", ""))
        cell_form.addRow("Cell ID", self._cell_combo)
        cell_form.addRow("Reviewer", self._reviewer_edit)
        cell_layout.addLayout(cell_form)
        self._cell_status = QLabel("No cell selected.")
        self._cell_status.setWordWrap(True)
        cell_layout.addWidget(self._cell_status)
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText("Optional note for the selected existing cell")
        self._notes_edit.setFixedHeight(60)
        cell_layout.addWidget(self._notes_edit)
        cell_buttons = QHBoxLayout()
        self._good_btn = QPushButton("[G] Good")
        self._bad_btn = QPushButton("[B] Bad")
        self._unannotated_btn = QPushButton("[U] Clear")
        self._good_btn.clicked.connect(lambda: self._set_current_cell_label(GOOD_LABEL))
        self._bad_btn.clicked.connect(lambda: self._set_current_cell_label(BAD_LABEL))
        self._unannotated_btn.clicked.connect(
            lambda: self._set_current_cell_label(UNANNOTATED_LABEL)
        )
        cell_buttons.addWidget(self._good_btn)
        cell_buttons.addWidget(self._bad_btn)
        cell_buttons.addWidget(self._unannotated_btn)
        cell_layout.addLayout(cell_buttons)
        cell_box.setLayout(cell_layout)

        missing_box = QGroupBox("Missing Cell 2D Boxes")
        missing_layout = QVBoxLayout()
        self._missing_mode_btn = QPushButton("[M] Missing Box Mode: OFF")
        self._missing_mode_btn.setCheckable(True)
        self._missing_mode_btn.toggled.connect(self._set_missing_mode)
        self._delete_missing_btn = QPushButton("[D] Delete Selected Missing Box")
        self._delete_missing_btn.clicked.connect(self._delete_selected_missing)
        self._missing_info = QLabel("Draw rectangles on the active Z slice.")
        self._missing_info.setWordWrap(True)
        missing_layout.addWidget(self._missing_mode_btn)
        missing_layout.addWidget(self._delete_missing_btn)
        missing_layout.addWidget(self._missing_info)
        missing_box.setLayout(missing_layout)

        save_row = QHBoxLayout()
        self._save_btn = QPushButton("[Ctrl+S] Save CSV")
        self._save_btn.clicked.connect(self._save_current_annotations)
        save_row.addWidget(self._save_btn)

        self._status = QPlainTextEdit()
        self._status.setReadOnly(True)
        self._status.setMaximumBlockCount(500)
        self._status.setFixedHeight(130)

        main.addWidget(paths_box)
        main.addWidget(nav_box)
        main.addWidget(cell_box)
        main.addWidget(missing_box)
        main.addLayout(save_row)
        main.addWidget(QLabel("Status"))
        main.addWidget(self._status)
        main.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _log(self, msg: str) -> None:
        self._status.appendPlainText(msg)
        # Also mirror to a logfile so failures are recoverable after the
        # session and easy to share for debugging.
        try:
            logdir = Path.home() / ".cmap_cell_exemption"
            logdir.mkdir(parents=True, exist_ok=True)
            with open(logdir / "session.log", "a") as fh:
                fh.write(msg.rstrip("\n") + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Folder and sample loading
    # ------------------------------------------------------------------

    def _pick_output_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "CMAP output root")
        if not d:
            return
        self._root_edit.setText(d)
        self._load_output_root(Path(d))

    def _pick_annotation_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Annotation CSV output folder")
        if not d:
            return
        self._set_annotation_dir(Path(d))

    def _set_annotation_dir(self, annotation_dir: Path) -> None:
        self.annotation_dir = annotation_dir
        self.annotation_csv = annotation_csv_path(
            annotation_dir, self._session_stamp, self._reviewer_edit.text()
        )
        self._ann_edit.setText(str(annotation_dir))
        # Each session starts fresh: prior session CSVs are NOT loaded back in,
        # so annotations from a previous run do not reappear on reopen. The new
        # session writes its own timestamped CSV, leaving older files intact.
        self._log(f"Annotation output: {self.annotation_csv}")
        self._refresh_current_from_annotations()

    def _load_output_root(self, root: Path) -> None:
        if not self._try_autosave():
            return
        self.output_root = root
        self.samples = discover_samples(root)
        self._loading_ui = True
        try:
            self._sample_combo.clear()
            for sample in self.samples:
                self._sample_combo.addItem(f"{sample.batch}/{sample.sample}", sample)
        finally:
            self._loading_ui = False
        if not self.samples:
            self._log(f"No reconstructed samples found under {root}")
            self._sample_info.setText("No samples found.")
            return
        if self.annotation_dir is None:
            suggested = root / "cell_exemption_annotations"
            self._ann_edit.setText(str(suggested))
            self._log(f"Suggested annotation output: {suggested}")
        self.sample_index = 0
        self._sample_combo.setCurrentIndex(0)
        self._load_sample(0)

    def _on_sample_combo_changed(self, index: int) -> None:
        if self._loading_ui or index < 0 or index >= len(self.samples):
            return
        if index == self.sample_index:
            return
        if not self._try_autosave():
            self._loading_ui = True
            try:
                self._sample_combo.setCurrentIndex(self.sample_index)
            finally:
                self._loading_ui = False
            return
        self._load_sample(index)

    def _load_sample(self, index: int) -> None:
        if index < 0 or index >= len(self.samples):
            return
        sample = self.samples[index]
        self.sample_index = index
        self.current_sample = sample
        self.current_cell_id = None

        # Missing-box drawing mode always starts OFF on a new image.
        if self._missing_mode_btn.isChecked():
            self._missing_mode_btn.setChecked(False)
        else:
            self._set_missing_mode(False)

        banner = self._show_loading_banner(sample)
        try:
            self.cell_boxes = load_cell_boxes(sample.coordinate_csv)
            self.cell_box_by_id = {box.cell_id: box for box in self.cell_boxes}
            data = load_combined_tiff(sample.image_path)
            self._display_sample(data, sample)
            self._populate_cell_combo()
            self._restore_missing_shapes()
            self._update_sample_info()
            self._log(
                f"Loaded {sample.batch}/{sample.sample}: "
                f"{len(self.cell_boxes)} cells, {data.shape}"
            )
        except Exception as exc:
            self._log(f"Load failed for {sample.sample}: {exc}\n{traceback.format_exc()}")
            QMessageBox.warning(self, "Load failed", str(exc))
        finally:
            self._close_loading_banner(banner)

    def _show_loading_banner(self, sample: SampleRecord):
        """Show a non-blocking popup warning that loading is slow.

        Loading happens synchronously on the GUI thread, so we paint the popup
        with ``processEvents`` before the blocking read and close it afterwards.
        """
        try:
            dlg = QMessageBox(self)
            dlg.setIcon(QMessageBox.Information)
            dlg.setWindowTitle("Loading image")
            dlg.setText(
                f"Loading {sample.batch}/{sample.sample}...\n\n"
                "These are multi-GB volumes; reading takes a few seconds. "
                "The viewer may stay unresponsive until loading finishes."
            )
            dlg.setStandardButtons(QMessageBox.NoButton)
            dlg.setModal(False)
            dlg.show()
            QApplication.processEvents()
            return dlg
        except Exception:
            return None

    def _close_loading_banner(self, banner) -> None:
        if banner is None:
            return
        try:
            banner.close()
            banner.deleteLater()
        except Exception:
            pass

    def _display_sample(self, data: np.ndarray, sample: SampleRecord) -> None:
        if data.ndim != 4 or data.shape[1] < 5:
            raise ValueError(
                f"Expected reconstructed TIFF shape (Z, 5, Y, X), got {data.shape}"
            )
        self._remove_plugin_layers()
        # ``data`` may be a lazy dask array (preferred, for fast navigation) or
        # an eager numpy array (fallback). ``rint``/``astype`` work on both;
        # dask keeps the label channel lazy so only displayed/clicked slices
        # decode. Avoid the ``copy=`` kwarg which dask's ``astype`` rejects.
        self.label_data = np.rint(data[:, 3, :, :]).astype(np.int32)

        # Each decoration is added in its own guarded section so that a failure
        # in one (e.g. a napari/vispy quirk) does not abort the whole load and
        # leave the big image without a fitted camera. ``reset_view`` always
        # runs in the finally block.
        try:
            for ch, name, colormap, visible in CHANNELS:
                kwargs = dict(
                    name=name,
                    colormap=colormap,
                    blending="additive",
                    visible=visible,
                )
                clim = self._contrast_limits(data, ch)
                if clim is not None:
                    kwargs["contrast_limits"] = clim
                layer = self.viewer.add_image(data[:, ch, :, :], **kwargs)
                if self._base_layer is None:
                    # Reused only for world->data coordinate conversion on click,
                    # since the cell-ID labels layer is no longer displayed.
                    self._base_layer = layer

            def _add_boundary() -> None:
                kwargs = dict(
                    name=LAYER_BOUNDARY,
                    colormap="gray",
                    blending="additive",
                    opacity=0.5,
                    visible=True,
                )
                clim = self._contrast_limits(data, 4)
                if clim is not None:
                    kwargs["contrast_limits"] = clim
                self.viewer.add_image(data[:, 4, :, :], **kwargs)

            self._section("boundary", _add_boundary)
            self._section("selected-box", self._add_selected_box_layer)
            self._section("missing", self._add_missing_layer)
        finally:
            self.viewer.title = f"CMAP Cell Exemption - {sample.batch}/{sample.sample}"
            self._fit_window_and_view()

    def _contrast_limits(self, data, ch: int):
        """Estimate 1-99.99 percentile display contrast limits for one channel.

        With lazy (dask) loading napari samples only the z=0 plane to guess
        contrast limits; for these stacks that plane is empty, so it falls back
        to ``[0, 1]`` and the float32 image saturates to solid white. We instead
        sample a few mid-stack slices (cheap: a handful of page decodes) and use
        the 1st/99.99th percentiles so the image renders with sensible brightness.
        """
        try:
            z_size = int(data.shape[0])
            if z_size <= 0:
                return None
            zs = sorted({int(z_size * f) for f in (0.25, 0.5, 0.75)})
            zs = [min(max(z, 0), z_size - 1) for z in zs] or [z_size // 2]
            planes = [np.asarray(data[z, ch, :, :]) for z in zs]
            sample = np.concatenate([p.ravel() for p in planes])
            sample = sample[np.isfinite(sample)]
            if sample.size == 0:
                return None
            lo = float(np.percentile(sample, 1.0))
            hi = float(np.percentile(sample, 99.99))
            if not (hi > lo):
                lo, hi = float(sample.min()), float(sample.max())
            if not (hi > lo):
                return None
            return (lo, hi)
        except Exception as exc:
            self._log(f"[contrast ch{ch}] {exc}")
            return None

    def _fit_window_and_view(self) -> None:
        """Give the window a stable size and fit the whole image into it.

        ``reset_view`` only fits correctly once the canvas has its final size,
        so we size the main window first (once) and then reset the camera both
        immediately and again after the Qt event loop has laid everything out.
        """
        if not self._window_sized:
            try:
                qwin = self.viewer.window._qt_window
                qwin.resize(1500, 950)
                self._window_sized = True
            except Exception:
                pass

        def _do_reset() -> None:
            try:
                self.viewer.dims.ndisplay = 2
                self.viewer.reset_view()
            except Exception:
                pass

        _do_reset()
        try:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, _do_reset)
            QTimer.singleShot(200, _do_reset)
        except Exception:
            pass

    def _section(self, label: str, fn) -> None:
        """Run a display section, logging (not raising) on failure."""
        try:
            fn()
        except Exception as exc:
            self._log(f"[display:{label}] skipped: {exc}\n{traceback.format_exc()}")

    def _add_labels_layer(self) -> None:
        self._labels_layer = self.viewer.add_labels(
            self.label_data,
            name=LAYER_LABELS,
            opacity=0.35,
            visible=True,
        )

    def _add_centroids_layer(self) -> None:
        points = np.asarray(
            [cell_box_to_point(box) for box in self.cell_boxes], dtype=float
        )
        if len(points) == 0:
            points = np.zeros((0, 3), dtype=float)
        features = pd.DataFrame({"cell_id": [box.cell_id for box in self.cell_boxes]})
        # napari >=0.5 renamed Points ``edge_color`` -> ``border_color``.
        points_kwargs = dict(
            name=LAYER_CENTROIDS,
            size=6,
            ndim=3,
            features=features,
            face_color="yellow",
            visible=True,
        )
        try:
            self.viewer.add_points(points, border_color="black", **points_kwargs)
        except TypeError:
            self.viewer.add_points(points, edge_color="black", **points_kwargs)

    def _add_selected_box_layer(self) -> None:
        # Create empty with just name+ndim to avoid empty-color-array quirks;
        # colors are applied when a cell is selected.
        self._selected_box_layer = self.viewer.add_shapes(
            name=LAYER_SELECTED_BOX,
            ndim=3,
        )

    def _add_missing_layer(self) -> None:
        self._missing_layer = self.viewer.add_shapes(
            name=LAYER_MISSING,
            ndim=3,
        )
        for attr, value in (
            ("current_edge_color", "cyan"),
            ("current_face_color", [0.0, 0.8, 1.0, 0.25]),
            ("current_edge_width", 2),
        ):
            try:
                setattr(self._missing_layer, attr, value)
            except Exception:
                pass
        self._connect_missing_layer_events()

    def _remove_plugin_layers(self) -> None:
        for name in list(PLUGIN_LAYER_NAMES):
            if name in self.viewer.layers:
                try:
                    self.viewer.layers.remove(name)
                except Exception:
                    pass
        self._labels_layer = None
        self._base_layer = None
        self._selected_mask_layer = None
        self._selected_box_layer = None
        self._missing_layer = None

    def _populate_cell_combo(self) -> None:
        self._loading_ui = True
        try:
            self._cell_combo.clear()
            for box in self.cell_boxes:
                label = self._existing_label_for_cell(box.cell_id)
                suffix = "" if label == UNANNOTATED_LABEL else f" ({label})"
                self._cell_combo.addItem(f"{box.cell_id}{suffix}", box.cell_id)
        finally:
            self._loading_ui = False
        if self.cell_boxes:
            self._cell_combo.setCurrentIndex(0)
            self._select_cell(self.cell_boxes[0].cell_id)
        else:
            self._cell_status.setText("No cells in coordinate CSV.")

    # ------------------------------------------------------------------
    # Existing-cell selection and labels
    # ------------------------------------------------------------------

    def _on_cell_combo_changed(self, index: int) -> None:
        if self._loading_ui or index < 0:
            return
        cell_id = self._cell_combo.itemData(index)
        if cell_id is not None:
            self._select_cell(int(cell_id))

    def _lock_camera_during_missing_draw(self, viewer, event):
        """Keep the camera fixed while a missing-cell box is being drawn.

        In ``add_rectangle`` mode napari recenters/zooms the camera onto the
        freshly drawn shape. Reviewers want the field of view to stay put, so we
        snapshot the camera at drag start and restore it after the shape is
        created (deferred so it overrides napari's own post-draw adjustment).
        """
        if not self._missing_mode:
            return
        center = tuple(viewer.camera.center)
        zoom = float(viewer.camera.zoom)
        yield  # let napari handle the press/drag (drawing the rectangle)

        def _restore() -> None:
            try:
                viewer.camera.center = center
                viewer.camera.zoom = zoom
            except Exception:
                pass

        _restore()
        try:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, _restore)
            QTimer.singleShot(50, _restore)
        except Exception:
            pass

    def _on_canvas_click(self, viewer, event) -> None:
        # Skip while drawing missing-cell boxes so clicks create shapes.
        if self._missing_mode:
            return
        if self.label_data is None or self._base_layer is None:
            return
        if getattr(event, "type", "mouse_press") != "mouse_press":
            return
        try:
            coords = np.round(
                self._base_layer.world_to_data(event.position)
            ).astype(int)
        except Exception:
            return
        coords = np.atleast_1d(coords)
        if coords.shape[0] < 3:
            return
        z, y, x = int(coords[-3]), int(coords[-2]), int(coords[-1])
        if (
            z < 0
            or y < 0
            or x < 0
            or z >= self.label_data.shape[0]
            or y >= self.label_data.shape[1]
            or x >= self.label_data.shape[2]
        ):
            return
        cell_id = int(self.label_data[z, y, x])
        if cell_id > 0:
            self._select_cell(cell_id)

    def _select_cell(self, cell_id: int) -> None:
        box = self.cell_box_by_id.get(cell_id)
        if box is None:
            self._log(f"Cell ID {cell_id} is not in the coordinate CSV.")
            return
        self.current_cell_id = cell_id
        self._sync_cell_combo(cell_id)

        if self._selected_box_layer is not None:
            # Draw the highlight rectangle on every Z slice spanning the cell's
            # crop range so it stays visible as the user scrolls through Z,
            # instead of appearing on a single slice only.
            z_lo = max(0, int(box.z0))
            z_hi = int(box.z1)
            if self.label_data is not None:
                z_hi = min(z_hi, self.label_data.shape[0])
            z_slices = list(range(z_lo, max(z_hi, z_lo + 1)))
            rects = [cell_box_to_slice_rectangle(box, z) for z in z_slices]
            n = len(rects)
            # Scale the border thickness to the image so it stays visible when
            # the full field of view is fit to the window.
            edge_w = 3.0
            if self.label_data is not None:
                edge_w = max(2.0, round(min(self.label_data.shape[1:]) / 250.0))
            try:
                self._selected_box_layer.edge_width = edge_w
                self._selected_box_layer.data = rects
                self._selected_box_layer.shape_type = ["rectangle"] * n
                # Transparent interior + solid yellow border so the cell's
                # original intensity is never overridden.
                self._selected_box_layer.face_color = _shape_rgba(n, (1.0, 1.0, 0.0, 0.0))
                self._selected_box_layer.edge_color = _shape_rgba(n, (1.0, 1.0, 0.0, 1.0))
                self._selected_box_layer.edge_width = [edge_w] * n
            except Exception as exc:
                self._log(f"[select-box] {exc}")

        row = self._existing_row_for_cell(cell_id)
        if row is not None:
            self._notes_edit.setPlainText(row.notes)
        else:
            self._notes_edit.clear()
        self._update_cell_status()

    def _sync_cell_combo(self, cell_id: int) -> None:
        if self._loading_ui:
            return
        for i in range(self._cell_combo.count()):
            if int(self._cell_combo.itemData(i)) == cell_id:
                if self._cell_combo.currentIndex() != i:
                    self._loading_ui = True
                    try:
                        self._cell_combo.setCurrentIndex(i)
                    finally:
                        self._loading_ui = False
                return

    def _current_image_key(self) -> str | None:
        if self.current_sample is None:
            return None
        return path_key(self.current_sample.image_path)

    def _existing_row_for_cell(self, cell_id: int) -> AnnotationRow | None:
        image_key = self._current_image_key()
        if image_key is None:
            return None
        row = self.rows_by_key.get((image_key, EXISTING_CELL, str(cell_id)))
        if row is not None:
            return row
        # Match rows saved under an alternate mount alias (jude vs dept/dnb).
        for candidate in self.rows_by_key.values():
            if (
                path_key(candidate.image_path) == image_key
                and candidate.annotation_type == EXISTING_CELL
                and candidate.cell_id == str(cell_id)
            ):
                return candidate
        return None

    def _existing_label_for_cell(self, cell_id: int) -> str:
        row = self._existing_row_for_cell(cell_id)
        return row.label if row is not None else UNANNOTATED_LABEL

    def _set_current_cell_label(self, label: str) -> None:
        if self.current_sample is None or self.current_cell_id is None:
            self._log("Select a cell before setting a label.")
            return
        box = self.cell_box_by_id.get(self.current_cell_id)
        if box is None:
            return
        image_key = path_key(self.current_sample.image_path)
        key = (image_key, EXISTING_CELL, str(box.cell_id))
        if label == UNANNOTATED_LABEL:
            self.rows_by_key.pop(key, None)
        else:
            old = self.rows_by_key.get(key)
            now = _now()
            self.rows_by_key[key] = AnnotationRow.for_existing_cell(
                self.current_sample,
                box,
                label,
                reviewer=self._reviewer_edit.text().strip(),
                notes=self._notes_edit.toPlainText().strip(),
                created_at=old.created_at if old is not None else now,
                updated_at=now,
                image_path_key=image_key,
            )
        self.dirty = True
        self._refresh_cell_combo_labels()
        self._update_cell_status()
        self._log(f"Cell {box.cell_id}: {label}")

    def _refresh_cell_combo_labels(self) -> None:
        current = self.current_cell_id
        self._loading_ui = True
        try:
            for i, box in enumerate(self.cell_boxes):
                label = self._existing_label_for_cell(box.cell_id)
                suffix = "" if label == UNANNOTATED_LABEL else f" ({label})"
                self._cell_combo.setItemText(i, f"{box.cell_id}{suffix}")
        finally:
            self._loading_ui = False
        if current is not None:
            self._sync_cell_combo(current)

    def _update_cell_status(self) -> None:
        if self.current_cell_id is None:
            self._cell_status.setText("No cell selected.")
            return
        box = self.cell_box_by_id[self.current_cell_id]
        label = self._existing_label_for_cell(box.cell_id)
        self._cell_status.setText(
            f"Cell {box.cell_id} | label: {label}\n"
            f"bbox z[{box.z0},{box.z1}) y[{box.y0},{box.y1}) x[{box.x0},{box.x1})"
        )

    # ------------------------------------------------------------------
    # Missing-cell shapes
    # ------------------------------------------------------------------

    def _connect_missing_layer_events(self) -> None:
        if self._missing_layer is None:
            return
        try:
            self._missing_layer.events.data.connect(self._on_missing_shapes_changed)
            self._missing_layer.events.features.connect(self._on_missing_shapes_changed)
        except Exception:
            pass

    def _on_missing_shapes_changed(self, event=None) -> None:
        if self._suppress_missing_events:
            return
        self.dirty = True
        self._update_missing_info()

    def _toggle_drag_to_zoom(self, disabled: bool) -> None:
        """Suspend napari's built-in ``drag_to_zoom`` while drawing boxes.

        Drawing a missing-cell rectangle is a canvas drag, which also triggers
        napari's drag-to-zoom and snaps the camera onto the new box. We pull
        that callback out of the viewer while missing mode is on and put it back
        afterwards so the field of view stays exactly where the reviewer left it.
        """
        try:
            cbs = self.viewer.mouse_drag_callbacks
            if disabled:
                for cb in list(cbs):
                    if getattr(cb, "__name__", "") == "drag_to_zoom":
                        self._disabled_drag_to_zoom = cb
                        cbs.remove(cb)
            else:
                cb = getattr(self, "_disabled_drag_to_zoom", None)
                if cb is not None and cb not in cbs:
                    cbs.append(cb)
                self._disabled_drag_to_zoom = None
        except Exception:
            pass

    def _set_missing_mode(self, on: bool) -> None:
        self._missing_mode = bool(on)
        self._missing_mode_btn.setText(
            "[M] Missing Box Mode: ON" if on else "[M] Missing Box Mode: OFF"
        )
        self._toggle_drag_to_zoom(disabled=on)
        if self._missing_layer is not None:
            try:
                self._missing_layer.mode = "add_rectangle" if on else "select"
                self.viewer.layers.selection.active = self._missing_layer
            except Exception:
                pass
        self._log("Missing box mode " + ("on" if on else "off") + ".")

    def _delete_selected_missing(self) -> None:
        layer = self._missing_layer
        if layer is None:
            return
        selected = sorted(layer.selected_data, reverse=True)
        if not selected:
            self._log("No missing-cell box selected.")
            return
        data = [d for i, d in enumerate(layer.data) if i not in selected]
        features = layer.features
        if features is None or len(features) == 0:
            new_features = pd.DataFrame()
        else:
            new_features = features.drop(index=selected).reset_index(drop=True)
        self._suppress_missing_events = True
        try:
            layer.data = data
            layer.features = new_features
            layer.selected_data = set()
        finally:
            self._suppress_missing_events = False
        self.dirty = True
        self._update_missing_info()

    def _restore_missing_shapes(self) -> None:
        if self.current_sample is None or self._missing_layer is None:
            return
        image_key = path_key(self.current_sample.image_path)
        rows = [
            row
            for row in self.rows_by_key.values()
            if row.image_path == image_key and row.annotation_type == MISSING_CELL
        ]
        data = []
        cell_ids = []
        z_indices = []
        for idx, row in enumerate(sorted(rows, key=lambda r: r.cell_id)):
            try:
                z = float(row.z_index or row.z0)
                y0 = float(row.y0)
                y1 = float(row.y1)
                x0 = float(row.x0)
                x1 = float(row.x1)
            except ValueError:
                continue
            data.append(
                np.asarray(
                    [[z, y0, x0], [z, y0, x1], [z, y1, x1], [z, y1, x0]],
                    dtype=np.float64,
                )
            )
            # Normalize legacy/blank ids (e.g. "nan" from older saves) to a
            # stable generated id so they round-trip cleanly.
            cid = str(row.cell_id).strip()
            if cid.lower() in ("", "nan", "none"):
                cid = f"missing_{idx + 1:04d}"
            cell_ids.append(cid)
            z_indices.append(int(round(z)))
        self._suppress_missing_events = True
        try:
            self._missing_layer.data = data
            self._missing_layer.features = pd.DataFrame(
                {"cell_id": cell_ids, "z_index": z_indices}
            )
            # napari crashes if ``shape_type``/color arrays are set to empty
            # lists on an empty Shapes layer (it does ``zip(*[])``), so only
            # apply per-shape attributes when there is at least one shape.
            if data:
                self._missing_layer.shape_type = ["rectangle"] * len(data)
                self._missing_layer.face_color = _shape_rgba(
                    len(data), (0.0, 0.8, 1.0, 0.18)
                )
                self._missing_layer.edge_color = _shape_rgba(
                    len(data), (0.0, 1.0, 1.0, 1.0)
                )
        finally:
            self._suppress_missing_events = False
        # Move the Z slider to a restored box so it is visible again instead of
        # being hidden on an off-screen slice.
        if z_indices:
            try:
                self.viewer.dims.set_current_step(0, int(z_indices[0]))
            except Exception:
                pass
        self._update_missing_info()

    def _missing_rows_from_shapes(self) -> list[AnnotationRow]:
        if self.current_sample is None or self._missing_layer is None:
            return []
        layer = self._missing_layer
        features = layer.features if layer.features is not None else pd.DataFrame()
        rows: list[AnnotationRow] = []
        image_key = path_key(self.current_sample.image_path)
        now = _now()
        for i, corners in enumerate(layer.data):
            try:
                values = corners_zyx_to_missing_box(np.asarray(corners))
            except ValueError as exc:
                self._log(f"Skipping malformed missing box {i + 1}: {exc}")
                continue
            raw_id = ""
            if i < len(features) and "cell_id" in features.columns:
                val = features.iloc[i].get("cell_id", "")
                # Newly drawn shapes get a NaN feature value; ``str(nan)``
                # yields the truthy string "nan", which would otherwise be
                # used as the id (and collide across boxes).
                if pd.notna(val):
                    raw_id = str(val).strip()
                    if raw_id.lower() in ("", "nan", "none"):
                        raw_id = ""
            cell_id = raw_id or f"missing_{i + 1:04d}"
            old = self.rows_by_key.get((image_key, MISSING_CELL, cell_id))
            z_index = int(values["z_index"])
            rows.append(
                AnnotationRow(
                    image_name=self.current_sample.image_name,
                    cell_id=cell_id,
                    annotation_type=MISSING_CELL,
                    label=MISSING_LABEL,
                    batch=self.current_sample.batch,
                    sample=self.current_sample.sample,
                    image_path=image_key,
                    z0=str(z_index),
                    z1=str(z_index + 1),
                    y0=f"{float(values['y0']):.3f}",
                    y1=f"{float(values['y1']):.3f}",
                    x0=f"{float(values['x0']):.3f}",
                    x1=f"{float(values['x1']):.3f}",
                    z_index=str(z_index),
                    center_x=f"{float(values['center_x']):.3f}",
                    center_y=f"{float(values['center_y']):.3f}",
                    width=f"{float(values['width']):.3f}",
                    height=f"{float(values['height']):.3f}",
                    reviewer=self._reviewer_edit.text().strip(),
                    notes="",
                    created_at=old.created_at if old is not None else now,
                    updated_at=now,
                )
            )
        return rows

    def _update_missing_rows_for_current_sample(self) -> None:
        if self.current_sample is None:
            return
        image_key = path_key(self.current_sample.image_path)
        for key in list(self.rows_by_key):
            if key[0] == image_key and key[1] == MISSING_CELL:
                del self.rows_by_key[key]
        for row in self._missing_rows_from_shapes():
            self.rows_by_key[row.key] = row

    def _update_missing_info(self) -> None:
        n = len(self._missing_layer.data) if self._missing_layer is not None else 0
        self._missing_info.setText(
            f"{n} missing-cell box(es). Draw rectangles on the active Z slice."
        )

    # ------------------------------------------------------------------
    # Save, navigation, shortcuts
    # ------------------------------------------------------------------

    def _save_current_annotations(self) -> bool:
        if self.annotation_dir is None:
            raw = self._ann_edit.text().strip()
            if raw:
                self._set_annotation_dir(Path(raw))
            else:
                QMessageBox.warning(self, "No output folder", "Choose an annotation output folder.")
                return False
        # Keep the reviewer name in the output filename current (it may have
        # been entered after the output folder was first chosen).
        self.annotation_csv = annotation_csv_path(
            self.annotation_dir, self._session_stamp, self._reviewer_edit.text()
        )
        if self.annotation_csv is None:
            return False
        self._update_missing_rows_for_current_sample()
        rows = sorted(
            self.rows_by_key.values(),
            key=lambda r: (r.batch, r.sample, r.annotation_type, r.cell_id),
        )
        try:
            save_annotations_atomic(self.annotation_csv, rows)
        except Exception as exc:
            self._log(f"Save failed: {exc}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.dirty = False
        self._log(f"Saved {len(rows)} rows to {self.annotation_csv}")
        return True

    def _try_autosave(self) -> bool:
        # Always capture the current sample's freehand missing-cell boxes so
        # they are not lost when navigating; existing-cell labels are already
        # in ``rows_by_key``. Saving is cheap/atomic, so we do not rely on the
        # ``dirty`` flag (interactive shape edits may not emit a change event).
        self._update_missing_rows_for_current_sample()
        has_output = self.annotation_dir is not None or bool(
            self._ann_edit.text().strip()
        )
        if not has_output:
            if not self.rows_by_key:
                return True
            QMessageBox.warning(
                self,
                "Unsaved annotations",
                "Choose an annotation output folder before navigating away.",
            )
            return False
        return self._save_current_annotations()

    def _previous_sample(self) -> None:
        if not self.samples:
            return
        if not self._try_autosave():
            return
        self._load_sample(max(0, self.sample_index - 1))
        self._sync_sample_combo()

    def _next_sample(self) -> None:
        if not self.samples:
            return
        if not self._try_autosave():
            return
        self._load_sample(min(len(self.samples) - 1, self.sample_index + 1))
        self._sync_sample_combo()

    def _sync_sample_combo(self) -> None:
        self._loading_ui = True
        try:
            self._sample_combo.setCurrentIndex(self.sample_index)
        finally:
            self._loading_ui = False

    def _update_sample_info(self) -> None:
        if self.current_sample is None:
            self._sample_info.setText("No sample loaded.")
            return
        image_key = path_key(self.current_sample.image_path)
        labeled = sum(
            1
            for row in self.rows_by_key.values()
            if path_key(row.image_path) == image_key
            and row.annotation_type == EXISTING_CELL
            and row.label in (GOOD_LABEL, BAD_LABEL)
        )
        total = len(self.cell_boxes)
        self._sample_info.setText(
            f"Sample {self.sample_index + 1} / {len(self.samples)}\n"
            f"{self.current_sample.batch}/{self.current_sample.sample}\n"
            f"{total} existing cells in this sample ({labeled} labeled)."
        )

    def _refresh_current_from_annotations(self) -> None:
        self._refresh_cell_combo_labels()
        if self.current_cell_id is not None:
            self._update_cell_status()
        self._restore_missing_shapes()
        self._update_sample_info()

    def _wire_shortcuts(self) -> None:
        self._key_actions = {
            Qt.Key_G: lambda: self._set_current_cell_label(GOOD_LABEL),
            Qt.Key_B: lambda: self._set_current_cell_label(BAD_LABEL),
            Qt.Key_U: lambda: self._set_current_cell_label(UNANNOTATED_LABEL),
            Qt.Key_N: self._next_sample,
            Qt.Key_P: self._previous_sample,
            Qt.Key_M: self._missing_mode_btn.toggle,
            Qt.Key_D: self._delete_selected_missing,
            Qt.Key_Delete: self._delete_selected_missing,
            Qt.Key_Backspace: self._delete_selected_missing,
        }
        app = QApplication.instance()
        if app is not None:
            self._hotkey_filter = _HotkeyEventFilter(self)
            app.installEventFilter(self._hotkey_filter)

    def _handle_hotkey(self, event) -> bool:
        # Never steal keys while the user is typing in a text field.
        app = QApplication.instance()
        focus = app.focusWidget() if app is not None else None
        if isinstance(focus, _TEXT_INPUT_TYPES):
            return False
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        if ctrl and event.key() == Qt.Key_S:
            self._save_current_annotations()
            return True
        # Plain single-key shortcuts only (allow Shift, ignore Ctrl/Alt/Meta).
        if ctrl or (mods & Qt.AltModifier) or (mods & Qt.MetaModifier):
            return False
        fn = self._key_actions.get(event.key())
        if fn is None:
            return False
        fn()
        return True

    def _hook_application_quit(self) -> None:
        if self._about_to_quit_hooked:
            return
        app = QApplication.instance()
        if app is None:
            return
        app.aboutToQuit.connect(self._on_application_quit)
        self._about_to_quit_hooked = True

    def _commit_and_save_if_possible(self) -> None:
        self._update_missing_rows_for_current_sample()
        has_output = self.annotation_dir is not None or bool(
            self._ann_edit.text().strip()
        )
        if has_output and self.rows_by_key:
            self._save_current_annotations()

    def _on_application_quit(self) -> None:
        self._commit_and_save_if_possible()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._commit_and_save_if_possible()
        app = QApplication.instance()
        flt = getattr(self, "_hotkey_filter", None)
        if app is not None and flt is not None:
            try:
                app.removeEventFilter(flt)
            except Exception:
                pass
        super().closeEvent(event)
