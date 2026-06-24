#!/usr/bin/env sh
"exec" "python" "$0" "$@"
"""Convenience launcher for the CMAP Cell Exemption plugin.

The primary workflow is still to install this package as a Napari plugin and
open it from the Plugins menu. This script is useful on shared systems where
users want to run directly from the source tree.
"""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

PLUGIN_SRC = Path(__file__).resolve().parent / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

# Newest GLIBCXX symbol required by recent system Mesa drivers (e.g. via
# libLLVM-15). Conda base environments often ship an older libstdc++ that lacks
# this, which makes the OpenGL driver (swrast) fail to load with confusing
# "MESA-LOADER: failed to open swrast" / "QOpenGLWidget: Failed to create
# context" errors.
_REQUIRED_GLIBCXX = b"GLIBCXX_3.4.30"
_SYSTEM_LIBSTDCXX_CANDIDATES = (
    "/lib/x86_64-linux-gnu/libstdc++.so.6",
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    "/usr/lib64/libstdc++.so.6",
)
_REEXEC_SENTINEL = "CMAP_EXEMPTION_REEXEC"


def _lib_has_symbol(path: str, symbol: bytes) -> bool:
    try:
        with open(path, "rb") as fh:
            return symbol in fh.read()
    except OSError:
        return False


def _maybe_fix_libstdcxx() -> None:
    """Re-exec with the system libstdc++ preloaded if the conda one is too old.

    Only triggers when the interpreter's bundled libstdc++ is missing the
    GLIBCXX version the system Mesa driver needs and a newer system libstdc++
    is available. Guarded by a sentinel env var to avoid re-exec loops.
    """

    if os.environ.get(_REEXEC_SENTINEL) == "1":
        return
    if os.environ.get("CMAP_EXEMPTION_NO_GL_FIX") == "1":
        return

    conda_lib = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
    if not os.path.exists(conda_lib):
        return
    if _lib_has_symbol(conda_lib, _REQUIRED_GLIBCXX):
        return  # conda libstdc++ is already new enough

    for system_lib in _SYSTEM_LIBSTDCXX_CANDIDATES:
        if not _lib_has_symbol(system_lib, _REQUIRED_GLIBCXX):
            continue
        preload = os.environ.get("LD_PRELOAD", "")
        if system_lib in preload.split(":"):
            return
        new_preload = f"{system_lib}:{preload}" if preload else system_lib
        env = dict(os.environ)
        env["LD_PRELOAD"] = new_preload
        env[_REEXEC_SENTINEL] = "1"
        print(f"[CMAP Cell Exemption] Preloading system libstdc++ for OpenGL: {system_lib}")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return


def main() -> int:
    if "--diagnose" in sys.argv:
        print("Python executable:", sys.executable)
        print("Plugin source:", PLUGIN_SRC)
        print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", "(unset)"))
        print("Re-exec sentinel:", os.environ.get(_REEXEC_SENTINEL, "(unset)"))
        return 0

    _maybe_fix_libstdcxx()

    try:
        from npe2.manifest.schema import discovery_blocked
    except Exception:
        discovery_context = nullcontext
    else:
        discovery_context = discovery_blocked

    try:
        with discovery_context():
            import napari
            from cmap_cell_exemption_plugin.widget import CellExemptionWidget

            viewer = napari.Viewer(title="CMAP Cell Exemption")
            widget = CellExemptionWidget(viewer)
            viewer.window.add_dock_widget(
                widget,
                name="CMAP Cell Exemption",
                area="right",
            )
            napari.run()
    except ImportError as exc:
        print("ERROR: napari or a plugin dependency is not available in this Python environment.")
        print("Load or activate the Napari environment, then re-run this script.")
        print(f"Details: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
