"""Start-up card shown while the main window loads.

A plain frameless widget rather than QSplashScreen: on Windows,
QSplashScreen.show() blocks for about a second waiting for the window to be
exposed, which is longer than everything else the card is there to cover.
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QWidget

from ensim import __version__

from . import theme
from .icons import svg_bytes

WIDTH, HEIGHT = 520, 300


class EnSimSplashScreen(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._step = 0
        self._steps = 4
        self._pixmap = self._paint("")
        self.setFixedSize(WIDTH, HEIGHT)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            centre = screen.availableGeometry().center()
            self.move(centre.x() - WIDTH // 2, centre.y() - HEIGHT // 2)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()

    def finish(self, _window) -> None:
        self.close()

    def _paint(self, message: str) -> QPixmap:
        p = theme.current()
        screen = QGuiApplication.primaryScreen()
        ratio = screen.devicePixelRatio() if screen else 1.0
        pixmap = QPixmap(round(WIDTH * ratio), round(HEIGHT * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QColor(p.bg))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(p.line))
        painter.drawRect(0, 0, WIDTH - 1, HEIGHT - 1)

        QSvgRenderer(svg_bytes("logo", p.signal)).render(painter, QRectF(40, 52, 64, 64))

        title = QFont(theme.SANS)
        title.setPixelSize(38)
        title.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title)
        painter.setPen(QColor(p.text))
        painter.drawText(
            QRectF(124, 52, 300, 46),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "EnSim",
        )

        sub = QFont(theme.SANS)
        sub.setPixelSize(13)
        painter.setFont(sub)
        painter.setPen(QColor(p.muted))
        painter.drawText(
            QRectF(126, 98, 360, 22),
            Qt.AlignmentFlag.AlignLeft,
            "Rocket engine and flight simulation",
        )

        mono = QFont(theme.MONO)
        mono.setPixelSize(11)
        painter.setFont(mono)
        painter.drawText(
            QRectF(40, HEIGHT - 66, 200, 18), Qt.AlignmentFlag.AlignLeft, f"version {__version__}"
        )
        painter.drawText(
            QRectF(40, HEIGHT - 44, WIDTH - 80, 18), Qt.AlignmentFlag.AlignLeft, message
        )

        painter.fillRect(QRectF(0, HEIGHT - 2, WIDTH, 2), QColor(p.line))
        painter.fillRect(
            QRectF(0, HEIGHT - 2, WIDTH * self._step / self._steps, 2), QColor(p.signal)
        )
        painter.end()
        return pixmap

    def showMessage(self, message: str) -> None:  # noqa: N802 - matches the Qt name
        self._step = min(self._step + 1, self._steps)
        self._pixmap = self._paint(message)
        self.repaint()
        QGuiApplication.processEvents()
