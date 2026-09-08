"""Launch Napari with the CMAP Cell Exemption widget docked.

Usage:
    python -m cmap_cell_exemption_plugin [OUTPUT_ROOT] [--annotation-dir DIR]
    python -m cmap_cell_exemption_plugin --check

If OUTPUT_ROOT is provided, the plugin discovers reconstructed samples under
that root immediately. ``--annotation-dir`` preselects the CSV output root; this
session's file is written to ``<dir>/<reviewer>/<YYYYMMDD>/`` beneath it. Prior
annotations in that folder are left alone, not loaded back into the widget.
``--check`` runs environment diagnostics instead of launching the GUI.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _check() -> int:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split("\n", 1)[0])

    for module_name in ("napari", "npe2"):
        if importlib.util.find_spec(module_name) is None:
            print(f"ERROR: {module_name} is not importable in this environment.")
            return 1

    try:
        from npe2 import PluginManifest

        manifest = PluginManifest.from_distribution("cmap-cell-exemption-plugin")
    except Exception as exc:
        print("ERROR: could not load plugin manifest:", exc)
        print("Install with: pip install -e /path/to/cell-exemption-napari")
        return 1

    print("Manifest OK:", manifest.name, "-", manifest.display_name)
    if manifest.contributions.widgets:
        for widget in manifest.contributions.widgets:
            print("  Widget:", widget.display_name)

    try:
        from cmap_cell_exemption_plugin.widget import CellExemptionWidget  # noqa: F401
    except Exception as exc:
        print("ERROR: widget import failed:", exc)
        return 1

    print("Widget class import OK.")
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse CLI args without interfering with napari's own sys.argv usage."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("output_root", nargs="?", default=None)
    parser.add_argument(
        "--annotation-dir",
        dest="annotation_dir",
        default=None,
        metavar="DIR",
        help="Root folder for annotation CSVs (filed under <reviewer>/<date>/).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run environment diagnostics instead of launching the GUI.",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()
    if args.check:
        raise SystemExit(_check())

    import napari

    from .widget import CellExemptionWidget

    viewer = napari.Viewer(title="CMAP Cell Exemption")
    widget = CellExemptionWidget(viewer)
    viewer.window.add_dock_widget(widget, name="CMAP Cell Exemption", area="right")

    if args.annotation_dir:
        widget._set_annotation_dir(Path(args.annotation_dir).expanduser())

    if args.output_root:
        root = Path(args.output_root).expanduser()
        if root.is_dir():
            widget._load_output_root(root)
        else:
            print(f"[CMAP Cell Exemption] Not a directory: {root}", file=sys.stderr)

    napari.run()


if __name__ == "__main__":
    main()
