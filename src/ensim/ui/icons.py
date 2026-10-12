"""Monochrome line icons from ``assets/icons``, tinted per use."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt6.QtGui import QIcon, QImage, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from ensim.utils.numba_support import user_cache_dir

ICON_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"
LOGO = ICON_DIR.parent / "logo.svg"


@cache
def _svg(name: str) -> str:
    path = LOGO if name == "logo" else ICON_DIR / f"{name}.svg"
    return path.read_text(encoding="utf-8")


def svg_bytes(name: str, color: str) -> QByteArray:
    return QByteArray(_svg(name).replace("currentColor", color).encode("utf-8"))


def pixmap(name: str, color: str, size: int, ratio: float = 2.0) -> QPixmap:
    """Render an icon at ``size`` logical pixels for a ``ratio`` device-pixel display."""
    pixels = round(size * ratio)
    image = QImage(pixels, pixels, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(svg_bytes(name, color)).render(painter, QRectF(0, 0, pixels, pixels))
    painter.end()
    result = QPixmap.fromImage(image)
    result.setDevicePixelRatio(ratio)
    return result


def icon(name: str, color: str, active: str | None = None, size: int = 20) -> QIcon:
    """QIcon with a normal tint and an optional tint for checked/active states."""
    result = QIcon()
    result.addPixmap(pixmap(name, color, size), QIcon.Mode.Normal, QIcon.State.Off)
    if active:
        result.addPixmap(pixmap(name, active, size), QIcon.Mode.Normal, QIcon.State.On)
        result.addPixmap(pixmap(name, active, size), QIcon.Mode.Active, QIcon.State.Off)
    return result


def cached_svg_path(name: str, color: str) -> Path:
    """A tinted copy on disk, for stylesheet ``url()`` references."""
    folder = user_cache_dir() / "icons"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-{color.lstrip('#')}.svg"
    if not path.exists():
        path.write_text(_svg(name).replace("currentColor", color), encoding="utf-8")
    return path


ICON_SIZE = QSize(18, 18)
