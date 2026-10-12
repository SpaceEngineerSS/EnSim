"""Short, purposeful motion.

Three uses only: a page settling in when the section changes (140 ms), a
number counting to its new value when a result arrives (180 ms), and the
indeterminate progress line while a task runs. Everything is skipped when
the user asks for reduced motion or Qt runs without a real display.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, QVariantAnimation
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect, QWidget

_enabled = True


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def enabled() -> bool:
    app = QApplication.instance()
    headless = app is not None and app.platformName() in {"offscreen", "minimal"}
    return _enabled and not headless


def fade_in(widget: QWidget, duration: int = 140) -> None:
    """Fade a freshly shown page from 0 to 1 opacity, then drop the effect."""
    if not enabled():
        return
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    animation = QPropertyAnimation(effect, b"opacity", widget)
    animation.setDuration(duration)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    # The effect forces offscreen rendering; remove it as soon as it is not needed.
    animation.finished.connect(lambda: widget.setGraphicsEffect(None))
    animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


def tween(
    start: float,
    end: float,
    apply: Callable[[float], None],
    owner: QWidget,
    duration: int = 180,
) -> QVariantAnimation | None:
    """Animate a float from ``start`` to ``end``, calling ``apply`` each frame."""
    if not enabled() or start == end:
        apply(end)
        return None
    animation = QVariantAnimation(owner)
    animation.setStartValue(float(start))
    animation.setEndValue(float(end))
    animation.setDuration(duration)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.valueChanged.connect(lambda value: apply(float(value)))
    animation.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)
    return animation


class ActivityLine(QWidget):
    """Two-pixel line: a sliding segment while busy, a filled bar when progress is known."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(2)
        self._color = QColor("#e8743b")
        self._track = QColor("#2a2d31")
        self._fraction = -1.0
        self._phase = 0.0
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_colors(self, signal: str, track: str) -> None:
        self._color = QColor(signal)
        self._track = QColor(track)
        self.update()

    def start(self) -> None:
        self._busy = True
        self._fraction = -1.0
        if enabled():
            self._timer.start()
        self.update()

    def set_fraction(self, fraction: float) -> None:
        self._fraction = fraction
        self.update()

    def stop(self) -> None:
        self._busy = False
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.012) % 1.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        width = self.width()
        painter.fillRect(0, 0, width, 2, self._track if self._busy else Qt.GlobalColor.transparent)
        if not self._busy:
            return
        if self._fraction >= 0.0:
            painter.fillRect(0, 0, round(width * self._fraction), 2, self._color)
        elif self._timer.isActive():
            segment = width * 0.22
            x = -segment + (width + segment) * self._phase
            painter.fillRect(round(x), 0, round(segment), 2, self._color)
        else:
            painter.fillRect(0, 0, width, 2, self._color)
