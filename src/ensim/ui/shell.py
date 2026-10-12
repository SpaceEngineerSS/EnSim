"""Window furniture: navigation rail, header and status strip."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .icons import icon, pixmap
from .motion import ActivityLine


class NavRail(QFrame):
    """Vertical list of sections."""

    selected = pyqtSignal(int)

    def __init__(self, entries: list[tuple[str, str]], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("NavRail")
        self.setFixedWidth(76)
        self._entries = entries
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(2)

        self._logo = QLabel()
        self._logo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._logo)
        layout.addSpacing(14)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QToolButton] = []
        for index, (label, _icon) in enumerate(entries):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(QSize(20, 20))
            button.setFixedWidth(76)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(f"{label}  (Ctrl+{index + 1})")
            self._group.addButton(button, index)
            self._buttons.append(button)
            layout.addWidget(button)
        layout.addStretch()
        self._group.idClicked.connect(self.selected.emit)
        self.restyle()

    def set_current(self, index: int) -> None:
        self._buttons[index].setChecked(True)

    def restyle(self) -> None:
        p = theme.current()
        self._logo.setPixmap(pixmap("logo", p.signal, 30))
        for button, (_label, name) in zip(self._buttons, self._entries, strict=True):
            button.setIcon(icon(name, p.muted, p.text))


class Header(QFrame):
    """Project name on the left, engine summary in the middle, run state on the right."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Header")
        self.setFixedHeight(44)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(18)
        self._project = QLabel("Untitled")
        self._project.setObjectName("Title")
        self._summary = QLabel("")
        self._summary.setObjectName("Muted")
        summary_font = QFont(theme.MONO)
        summary_font.setPixelSize(12)
        self._summary.setFont(summary_font)
        self._state = QLabel("")
        self._state.setObjectName("Muted")
        layout.addWidget(self._project)
        layout.addWidget(self._summary)
        layout.addStretch()
        layout.addWidget(self._state)

    def set_project(self, name: str, modified: bool) -> None:
        self._project.setText(f"{name}{'  ·  edited' if modified else ''}")

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)

    def set_state(self, text: str, kind: str = "Muted") -> None:
        self._state.setObjectName(kind)
        self._state.setText(text)
        self._state.style().unpolish(self._state)
        self._state.style().polish(self._state)


class StatusStrip(QFrame):
    """Message on the left, activity line along the top edge, Cancel while busy."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StatusStrip")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.activity = ActivityLine()
        outer.addWidget(self.activity)
        row = QHBoxLayout()
        row.setContentsMargins(14, 3, 10, 3)
        self._message = QLabel("Ready")
        self._detail = QLabel("")
        self._cancel = QPushButton("Cancel  Esc")
        self._cancel.setObjectName("Quiet")
        self._cancel.setVisible(False)
        self._cancel.clicked.connect(self.cancel_requested.emit)
        row.addWidget(self._message, 1)
        row.addWidget(self._detail)
        row.addWidget(self._cancel)
        outer.addLayout(row)
        self.restyle()

    def restyle(self) -> None:
        p = theme.current()
        self.activity.set_colors(p.signal, p.line)

    def show_message(self, text: str) -> None:
        self._message.setText(text)

    def set_detail(self, text: str) -> None:
        self._detail.setText(text)

    def set_busy(self, busy: bool) -> None:
        self._cancel.setVisible(busy)
        if busy:
            self.activity.start()
        else:
            self.activity.stop()
            self._detail.setText("")

    def set_progress(self, fraction: float, text: str) -> None:
        if fraction >= 0.0:
            self.activity.set_fraction(fraction)
        if text:
            self._detail.setText(text)
