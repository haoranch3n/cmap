"""CMAP Cell Exemption Napari plugin."""

from __future__ import annotations

try:
    from .widget import CellExemptionWidget
except Exception:  # pragma: no cover - lets npe2 report import errors cleanly
    CellExemptionWidget = None  # type: ignore[assignment]

__all__ = ["CellExemptionWidget"]
__version__ = "0.1.0"
