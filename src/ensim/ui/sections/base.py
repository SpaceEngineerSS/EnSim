"""Common layout for a section: inputs on the left, views on the right."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..settings import Settings
from ..widgets.controls import hint, primary_button
from ..workspace import Workspace


class Section(QWidget):
    """A page of the main window, built the first time it is shown.

    Subclasses implement :meth:`build`. :meth:`run` is the section's primary
    action (F5); sections without one leave :attr:`run_label` empty.
    """

    title = ""
    icon_name = ""
    run_label = ""

    def __init__(self, workspace: Workspace, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.workspace = workspace
        self.settings = settings
        self._built = False
        self.run_button: QPushButton | None = None
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

    @property
    def built(self) -> bool:
        return self._built

    def ensure_built(self) -> None:
        if not self._built:
            self._built = True
            self.build()
            self.load_from_project()

    def build(self) -> None:
        raise NotImplementedError

    def load_from_project(self) -> None:
        """Copy project values into the inputs (after build and after a project loads)."""

    def run(self) -> None:
        """Primary action of the section."""

    def restyle(self) -> None:
        """Called after the theme changes."""

    def set_running(self, running: bool) -> None:
        if self.run_button is not None:
            self.run_button.setEnabled(not running)
            self.run_button.setText("Running…" if running else self.run_label)

    def split(
        self,
        inputs: QWidget,
        views: QWidget,
        footer: QWidget | None = None,
        inputs_width: int = 300,
    ) -> QSplitter:
        """Inputs panel (scrolling, with a fixed run button) beside the views."""
        panel = QFrame()
        panel.setObjectName("InputsPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(16, 6, 16, 16)
        holder_layout.addWidget(inputs)
        holder_layout.addStretch()
        scroll.setWidget(holder)
        panel_layout.addWidget(scroll, 1)

        if self.run_label:
            bar = QFrame()
            bar_layout = QVBoxLayout(bar)
            bar_layout.setContentsMargins(16, 10, 16, 14)
            bar_layout.setSpacing(6)
            if footer is not None:
                bar_layout.addWidget(footer)
            self.run_button = primary_button(self.run_label, "run")
            self.run_button.setToolTip(f"{self.run_label}  (F5)")
            self.run_button.clicked.connect(self.run)
            bar_layout.addWidget(self.run_button)
            panel_layout.addWidget(bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(panel)
        splitter.addWidget(views)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([inputs_width, 1000])
        panel.setMinimumWidth(250)
        self._root.addWidget(splitter)
        self._splitter = splitter
        return splitter


def tabs(*pages: tuple[str, QWidget]) -> QTabWidget:
    widget = QTabWidget()
    widget.setDocumentMode(True)
    for title, page in pages:
        widget.addTab(page, title)
    return widget


def padded(widget: QWidget, margins: tuple[int, int, int, int] = (18, 12, 18, 12)) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(*margins)
    layout.addWidget(widget)
    return holder


def notice(text: str) -> QWidget:
    """Centered hint for views that need a prerequisite (usually an engine run)."""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.addStretch()
    label = hint(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    layout.addStretch()
    return holder


def metric_row(*metrics: QWidget) -> QWidget:
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(18, 14, 18, 10)
    layout.setSpacing(28)
    for metric in metrics:
        layout.addWidget(metric)
    layout.addStretch()
    return holder


def caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    return label


class Lazy(QWidget):
    """Placeholder that builds its real content the first time it is shown."""

    def __init__(self, factory):
        super().__init__()
        self._factory = factory
        self.widget: QWidget | None = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.widget is None:
            self.widget = self._factory()
            self._layout.addWidget(self.widget)


def lazy(factory) -> Lazy:
    return Lazy(factory)
