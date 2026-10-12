"""Matplotlib canvas that loads on first use and follows the theme.

The owner supplies a ``render(figure)`` function that draws everything from
its current data. The canvas calls it when data changes (``refresh``) and
when the theme changes, so figures never hold stale colours.

Mouse: wheel zooms about the cursor, drag pans, double-click resets, the
cursor position is shown in the corner. Right-click saves the figure or the
plotted data.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMenu, QVBoxLayout, QWidget

from .. import theme

_fonts_registered = False


def _register_fonts() -> None:
    global _fonts_registered
    if _fonts_registered:
        return
    from matplotlib import font_manager

    for path in sorted((theme.ASSETS / "fonts").glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    _fonts_registered = True


def decimate(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation: keeps peaks while bounding the point count."""
    n = len(x)
    if n <= max_points:
        return x, y
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    keep = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        chunk = y[start:stop]
        low = start + int(np.argmin(chunk))
        high = start + int(np.argmax(chunk))
        keep.extend(sorted({low, high}))
    keep.append(n - 1)
    index = np.unique(keep)
    return x[index], y[index]


Renderer = Callable[["object"], None]
DataSource = Callable[[], dict[str, np.ndarray]]


class PlotCanvas(QWidget):
    """A themed figure. Nothing from matplotlib is imported until it is shown."""

    def __init__(
        self,
        render: Renderer,
        data: DataSource | None = None,
        placeholder: str = "Run the calculation to see this plot.",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._render = render
        self._data = data
        self._placeholder = placeholder
        self._has_data = False
        self._figure = None
        self._canvas = None
        self._readout = None
        self._drag = None
        self._stale = False
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    # lifecycle

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._canvas is None:
            self._build()
            self._draw()
        elif self._stale:
            self._draw()

    def _build(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        _register_fonts()
        self._figure = Figure(layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._canvas.customContextMenuRequested.connect(self._menu)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._layout.addWidget(self._canvas)

    def refresh(self, has_data: bool = True) -> None:
        """Redraw from the owner's data; cheap when the canvas is hidden."""
        self._has_data = has_data
        if self._canvas is not None and self.isVisible():
            self._draw()
        elif self._canvas is not None:
            self._stale = True

    def restyle(self) -> None:
        if self._canvas is not None:
            self._draw()

    def _draw(self) -> None:
        import matplotlib as mpl

        self._stale = False
        figure = self._figure
        figure.clear()
        with mpl.rc_context(theme.matplotlib_rc()):
            figure.set_facecolor(theme.current().bg)
            if self._has_data:
                self._render(figure)
            else:
                ax = figure.add_subplot(111)
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    self._placeholder,
                    ha="center",
                    va="center",
                    color=theme.current().muted,
                    fontsize=10,
                    transform=ax.transAxes,
                )
            self._readout = figure.text(
                0.995,
                0.995,
                "",
                ha="right",
                va="top",
                fontsize=8,
                color=theme.current().muted,
                family=theme.MONO,
            )
        self._canvas.draw_idle()

    def figure(self):
        return self._figure

    # interaction

    def _on_scroll(self, event) -> None:
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        factor = 0.85 if event.button == "up" else 1 / 0.85
        for get, set_, centre in (
            (ax.get_xlim, ax.set_xlim, event.xdata),
            (ax.get_ylim, ax.set_ylim, event.ydata),
        ):
            low, high = get()
            set_(centre - (centre - low) * factor, centre + (high - centre) * factor)
        self._canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes is None:
            return
        if event.dblclick:
            event.inaxes.relim()
            event.inaxes.autoscale()
            self._canvas.draw_idle()
            return
        if event.button == 1:
            self._drag = (
                event.inaxes,
                event.x,
                event.y,
                event.inaxes.get_xlim(),
                event.inaxes.get_ylim(),
            )

    def _on_release(self, _event) -> None:
        self._drag = None

    def _on_motion(self, event) -> None:
        if self._drag is not None and event.x is not None:
            ax, x0, y0, xlim, ylim = self._drag
            inverse = ax.transData.inverted()
            start = inverse.transform((x0, y0))
            now = inverse.transform((event.x, event.y))
            dx, dy = start - now
            ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self._canvas.draw_idle()
            return
        if self._readout is None or not self._has_data:
            return
        text = "" if event.inaxes is None else f"{event.xdata:.4g}, {event.ydata:.4g}"
        if self._readout.get_text() != text:
            self._readout.set_text(text)
            self._canvas.draw_idle()

    # export

    def _menu(self, position) -> None:
        menu = QMenu(self)
        menu.addAction("Save image…", self._save_image)
        if self._data is not None and self._has_data:
            menu.addAction("Export data as CSV…", self._save_data)
        menu.exec(self._canvas.mapToGlobal(position))

    def _save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "plot.png", "PNG (*.png);;SVG (*.svg)"
        )
        if path:
            self._figure.savefig(path, dpi=200)

    def _save_data(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export data", "data.csv", "CSV (*.csv)")
        if not path:
            return
        columns = self._data()
        names = list(columns)
        length = max(len(v) for v in columns.values())
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(names)
            for i in range(length):
                writer.writerow([columns[n][i] if i < len(columns[n]) else "" for n in names])
