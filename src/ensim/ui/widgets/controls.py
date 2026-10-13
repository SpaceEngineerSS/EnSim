"""Small building blocks shared by every section."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ensim.utils.units import Quantity, UnitSystem

from .. import motion, theme
from ..icons import icon
from ..units import UNITS, fmt


def section_label(text: str) -> QLabel:
    """Small uppercase group heading."""
    label = QLabel(text)
    label.setObjectName("SectionLabel")
    font = label.font()
    font.setCapitalization(QFont.Capitalization.AllUppercase)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
    font.setPixelSize(11)
    font.setWeight(QFont.Weight.DemiBold)
    label.setFont(font)
    return label


def hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


def primary_button(text: str, icon_name: str | None = None) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("Primary")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if icon_name:
        p = theme.current()
        button.setIcon(icon(icon_name, p.signal_text))
    return button


def quiet_button(text: str, icon_name: str | None = None, tooltip: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("Quiet")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if icon_name:
        p = theme.current()
        button.setIcon(icon(icon_name, p.muted, p.text))
    if tooltip:
        button.setToolTip(tooltip)
    return button


class NumberField(QDoubleSpinBox):
    """Numeric input that stores SI and shows the user's unit system.

    ``quantity=None`` makes a plain dimensionless field. Editing finishes on
    Enter or focus-out; :attr:`changed_si` fires only then, so typing a long
    number does not trigger work for every keystroke.
    """

    changed_si = pyqtSignal(float)

    def __init__(
        self,
        value: float,
        minimum: float,
        maximum: float,
        quantity: Quantity | None = None,
        *,
        decimals: int = 3,
        step: float | None = None,
        units: dict[UnitSystem, str] | None = None,
        suffix: str = "",
        tooltip: str = "",
    ):
        super().__init__()
        self._quantity = quantity
        self._overrides = units
        self._si_min, self._si_max = minimum, maximum
        self._decimals = decimals
        self._step = step
        self._suffix = suffix
        self._si_value = value
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(88)
        if tooltip:
            self.setToolTip(tooltip)
        self.valueChanged.connect(self._on_value)
        if quantity is not None:
            UNITS.changed.connect(lambda _system: self._apply_units())
        self._apply_units()

    @property
    def unit(self) -> str:
        return UNITS.unit(self._quantity, self._overrides) if self._quantity else self._suffix

    def _to_display(self, value_si: float) -> float:
        return UNITS.show(value_si, self._quantity, self.unit) if self._quantity else value_si

    def _apply_units(self) -> None:
        self.blockSignals(True)
        low, high = sorted((self._to_display(self._si_min), self._to_display(self._si_max)))
        self.setRange(low, high)
        self.setDecimals(self._decimals)
        if self._step is not None:
            scale = (
                self._to_display(self._step) - self._to_display(0.0)
                if self._quantity
                else self._step
            )
            self.setSingleStep(abs(scale) or self._step)
        unit = self.unit
        self.setSuffix(f"  {unit}" if unit else "")
        self.setValue(self._to_display(self._si_value))
        self.blockSignals(False)

    def _on_value(self, shown: float) -> None:
        self._si_value = UNITS.read(shown, self._quantity, self.unit) if self._quantity else shown
        self.changed_si.emit(self._si_value)

    def value_si(self) -> float:
        return self._si_value

    def set_value_si(self, value: float) -> None:
        self._si_value = value
        self.blockSignals(True)
        self.setValue(self._to_display(value))
        self.blockSignals(False)


class Choice(QComboBox):
    """Combo box over (label, data) pairs."""

    def __init__(self, items: Iterable[tuple], tooltip: str = ""):
        super().__init__()
        for item in items:
            label, data = item[0], item[1]
            self.addItem(label, data)
            if len(item) > 2:
                self.setItemData(self.count() - 1, item[2], Qt.ItemDataRole.ToolTipRole)
        if tooltip:
            self.setToolTip(tooltip)
        # Long entries must not widen the inputs panel; the popup still shows them in full
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def value(self):
        return self.currentData()

    def set_value(self, data) -> bool:
        index = self.findData(data)
        if index >= 0:
            self.setCurrentIndex(index)
        return index >= 0


class Form(QWidget):
    """Label / field grid with optional group headings."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(1, 1)
        self._row = 0

    def heading(self, text: str) -> None:
        self._grid.addWidget(section_label(text), self._row, 0, 1, 2)
        self._row += 1

    def add(self, label: str, field: QWidget, tooltip: str = "") -> QWidget:
        text = QLabel(label)
        text.setObjectName("Muted")
        if tooltip:
            text.setToolTip(tooltip)
            field.setToolTip(field.toolTip() or tooltip)
        self._grid.addWidget(text, self._row, 0)
        self._grid.addWidget(field, self._row, 1)
        self._row += 1
        return field

    def span(self, widget: QWidget) -> QWidget:
        self._grid.addWidget(widget, self._row, 0, 1, 2)
        self._row += 1
        return widget


class Metric(QFrame):
    """A labelled number. New values count up from the previous one."""

    def __init__(
        self,
        label: str,
        quantity: Quantity | None = None,
        unit: str = "",
        digits: int = 4,
        integer: bool = False,
    ):
        super().__init__()
        self._quantity = quantity
        self._unit = unit
        self._digits = digits
        self._integer = integer
        self._si: float | None = None
        self._shown: float | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        caption = QLabel(label)
        # Sizes and families come from the theme stylesheet, which overrides setFont
        caption.setObjectName("MetricCaption")
        row = QHBoxLayout()
        row.setSpacing(6)
        self._value = QLabel("–")
        self._value.setObjectName("MetricValue")
        self._unit_label = QLabel(self._unit_text())
        self._unit_label.setObjectName("Muted")
        row.addWidget(self._value)
        row.addWidget(self._unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch()
        layout.addWidget(caption)
        layout.addLayout(row)
        if quantity is not None:
            UNITS.changed.connect(lambda _system: self._refresh_units())

    def _unit_text(self) -> str:
        return UNITS.unit(self._quantity) if self._quantity else self._unit

    def _display(self, value_si: float) -> float:
        return UNITS.show(value_si, self._quantity) if self._quantity else value_si

    def set_value(self, value_si: float | None) -> None:
        self._si = value_si
        if value_si is None:
            self._shown = None
            self._value.setText("–")
            return
        target = self._display(value_si)
        start = self._shown if self._shown is not None else target
        self._shown = target
        text = (lambda v: f"{round(v):d}") if self._integer else (lambda v: fmt(v, self._digits))
        motion.tween(start, target, lambda v: self._value.setText(text(v)), self)

    def _refresh_units(self) -> None:
        self._unit_label.setText(self._unit_text())
        self._shown = None
        self.set_value(self._si)


class ResultTable(QTableWidget):
    """Read-only table for numeric results."""

    def __init__(self, headers: list[str], parent: QWidget | None = None):
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.verticalHeader().setDefaultSectionSize(24)

    def set_rows(self, rows: Iterable[Iterable[object]]) -> None:
        rows = [list(row) for row in rows]
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                text = fmt(value) if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                if isinstance(value, (int, float)):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.setItem(r, c, item)


def hbox(*widgets: QWidget, spacing: int = 8, stretch_last: bool = False) -> QWidget:
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    if stretch_last:
        layout.addStretch()
    return holder


def connect_all(fields: Iterable[QWidget], slot: Callable[[], None]) -> None:
    """Call ``slot`` when any field finishes editing."""
    for field in fields:
        if isinstance(field, NumberField):
            field.changed_si.connect(lambda _v: slot())
        elif isinstance(field, QComboBox):
            field.currentIndexChanged.connect(lambda _i: slot())
        elif hasattr(field, "toggled"):
            field.toggled.connect(lambda _b: slot())
