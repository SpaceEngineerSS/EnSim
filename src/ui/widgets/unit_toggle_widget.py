"""
Unit System Toggle Widget.

Provides UI for switching between SI and Imperial units with:
- Quick toggle button
- Visual indicator of current system
- Signal emission for app-wide unit updates
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFrame, QButtonGroup
)
from PyQt6.QtGui import QFont

from ...utils.units import UnitSystem, UnitConverter, UnitCategory


class UnitToggleWidget(QWidget):
    """
    Compact unit system toggle widget.
    
    Provides SI/Imperial toggle with visual feedback.
    
    Signals:
        unit_system_changed: Emitted when unit system changes
    """
    
    unit_system_changed = pyqtSignal(object)  # Emits UnitSystem
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_system = UnitSystem.SI
        self._converter = UnitConverter(UnitSystem.SI)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Label
        label = QLabel("Units:")
        label.setObjectName("unitLabel")
        layout.addWidget(label)
        
        # Toggle buttons
        self.si_btn = QPushButton("SI")
        self.si_btn.setObjectName("unitToggleSI")
        self.si_btn.setCheckable(True)
        self.si_btn.setChecked(True)
        self.si_btn.setFixedWidth(50)
        self.si_btn.clicked.connect(lambda: self._set_system(UnitSystem.SI))
        layout.addWidget(self.si_btn)
        
        self.imperial_btn = QPushButton("IMP")
        self.imperial_btn.setObjectName("unitToggleIMP")
        self.imperial_btn.setCheckable(True)
        self.imperial_btn.setFixedWidth(50)
        self.imperial_btn.clicked.connect(lambda: self._set_system(UnitSystem.IMPERIAL))
        layout.addWidget(self.imperial_btn)
        
        # Button group for exclusive selection
        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.si_btn)
        self.btn_group.addButton(self.imperial_btn)
        self.btn_group.setExclusive(True)
    
    def _set_system(self, system: UnitSystem):
        """Set the unit system."""
        if system == self._current_system:
            return
        
        self._current_system = system
        self._converter.set_system(system)
        
        # Update button states
        self.si_btn.setChecked(system == UnitSystem.SI)
        self.imperial_btn.setChecked(system == UnitSystem.IMPERIAL)
        
        # Emit signal
        self.unit_system_changed.emit(system)
    
    def get_system(self) -> UnitSystem:
        """Get current unit system."""
        return self._current_system
    
    def get_converter(self) -> UnitConverter:
        """Get the unit converter configured for current system."""
        return self._converter
    
    def format_value(self, value_si: float, category: UnitCategory, precision: int = 4) -> str:
        """Format a SI value for display in current unit system."""
        return self._converter.display(value_si, category, precision)
    
    def to_si(self, value: float, category: UnitCategory) -> float:
        """Convert user input to SI units."""
        return self._converter.input_to_si(value, category)


class UnitDisplayLabel(QLabel):
    """
    Smart label that auto-converts values based on unit system.
    
    Updates display when unit system changes.
    """
    
    def __init__(self, category: UnitCategory, precision: int = 4, parent=None):
        super().__init__(parent)
        self.category = category
        self.precision = precision
        self._value_si = 0.0
        self._converter = UnitConverter(UnitSystem.SI)
        self.setObjectName("unitDisplayLabel")
    
    def set_converter(self, converter: UnitConverter):
        """Set the unit converter to use."""
        self._converter = converter
        self._update_display()
    
    def set_value_si(self, value: float):
        """Set value in SI units."""
        self._value_si = value
        self._update_display()
    
    def _update_display(self):
        """Update displayed value."""
        display = self._converter.display(self._value_si, self.category, self.precision)
        self.setText(display)
    
    def get_value_si(self) -> float:
        """Get value in SI units."""
        return self._value_si


class UnitAwareSpinBox(QWidget):
    """
    Spin box that displays and accepts values in current unit system
    but stores/returns SI values internally.
    """
    
    value_changed = pyqtSignal(float)  # Emits SI value
    
    def __init__(self, category: UnitCategory, parent=None):
        super().__init__(parent)
        self.category = category
        self._converter = UnitConverter(UnitSystem.SI)
        self._setup_ui()
    
    def _setup_ui(self):
        from PyQt6.QtWidgets import QDoubleSpinBox
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(4)
        self.spinbox.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.spinbox)
        
        self.unit_label = QLabel("")
        self.unit_label.setObjectName("spinboxUnit")
        self.unit_label.setMinimumWidth(40)
        layout.addWidget(self.unit_label)
        
        self._update_units()
    
    def set_converter(self, converter: UnitConverter):
        """Set the unit converter."""
        old_si = self.get_value_si()
        self._converter = converter
        self._update_units()
        self.set_value_si(old_si)
    
    def _update_units(self):
        """Update unit display and ranges."""
        from ...utils.units import get_unit_symbol, SI_DEFAULTS, IMPERIAL_DEFAULTS
        
        defaults = SI_DEFAULTS if self._converter.system == UnitSystem.SI else IMPERIAL_DEFAULTS
        unit = defaults.get(self.category, "")
        symbol = get_unit_symbol(unit) if unit else ""
        self.unit_label.setText(symbol)
    
    def _on_value_changed(self, display_value: float):
        """Handle spinbox value change."""
        si_value = self._converter.input_to_si(display_value, self.category)
        self.value_changed.emit(si_value)
    
    def set_value_si(self, si_value: float):
        """Set value in SI units."""
        display_value = self._converter.si_to_display(si_value, self.category)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(display_value)
        self.spinbox.blockSignals(False)
    
    def get_value_si(self) -> float:
        """Get value in SI units."""
        return self._converter.input_to_si(self.spinbox.value(), self.category)
    
    def set_range(self, min_si: float, max_si: float):
        """Set range in SI units."""
        min_display = self._converter.si_to_display(min_si, self.category)
        max_display = self._converter.si_to_display(max_si, self.category)
        self.spinbox.setRange(min_display, max_display)
    
    def set_decimals(self, decimals: int):
        """Set decimal precision."""
        self.spinbox.setDecimals(decimals)


class UnitSystemBar(QFrame):
    """
    Toolbar/status bar widget showing current unit system with quick toggle.
    """
    
    unit_system_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("unitSystemBar")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        # Current system indicator
        self.system_label = QLabel("Unit System: SI (Metric)")
        self.system_label.setObjectName("systemIndicator")
        font = QFont()
        font.setBold(True)
        self.system_label.setFont(font)
        layout.addWidget(self.system_label)
        
        layout.addStretch()
        
        # Unit examples
        self.examples_label = QLabel("Force: N | Pressure: Pa | Temp: K")
        self.examples_label.setObjectName("unitExamples")
        layout.addWidget(self.examples_label)
        
        layout.addStretch()
        
        # Toggle
        self.toggle = UnitToggleWidget()
        self.toggle.unit_system_changed.connect(self._on_system_changed)
        layout.addWidget(self.toggle)
    
    def _on_system_changed(self, system: UnitSystem):
        """Handle system change."""
        if system == UnitSystem.SI:
            self.system_label.setText("Unit System: SI (Metric)")
            self.examples_label.setText("Force: N | Pressure: Pa | Temp: K")
        else:
            self.system_label.setText("Unit System: Imperial (US)")
            self.examples_label.setText("Force: lbf | Pressure: psi | Temp: °R")
        
        self.unit_system_changed.emit(system)
    
    def get_converter(self) -> UnitConverter:
        """Get the unit converter."""
        return self.toggle.get_converter()

