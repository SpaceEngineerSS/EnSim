"""Render the application icons from assets/logo.svg.

Writes assets/icon.png (512 px) and assets/icon.ico (16-256 px) used by the
window and by the PyInstaller bundles. Needs PyQt6 and Pillow.

    python scripts/make_icons.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath  # noqa: E402
from PyQt6.QtSvg import QSvgRenderer  # noqa: E402

from ensim.ui.theme import CONSOLE  # noqa: E402

ASSETS = ROOT / "src" / "ensim" / "assets"


def render(size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(0, 0, size, size), size * 0.18, size * 0.18)
    painter.fillPath(tile, QColor(CONSOLE.surface))
    svg = (ASSETS / "logo.svg").read_text(encoding="utf-8").replace("currentColor", CONSOLE.signal)
    inset = size * 0.16
    QSvgRenderer(QByteArray(svg.encode())).render(
        painter, QRectF(inset, inset, size - 2 * inset, size - 2 * inset)
    )
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data())))


def main() -> None:
    QGuiApplication.instance() or QGuiApplication([])
    render(512).save(ASSETS / "icon.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    render(256).save(ASSETS / "icon.ico", sizes=[(s, s) for s in sizes])
    print("icon.png, icon.ico written")


if __name__ == "__main__":
    main()
