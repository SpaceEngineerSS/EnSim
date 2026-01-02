"""
Input Panel Widget - Mission Control Style.

Professional aerospace dashboard input panel with card-based grouping
and scroll area for flexible content.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class InputCard(QGroupBox):
    """Styled card container for input groups."""

    def __init__(self, title: str, icon: str = "", parent=None):
        super().__init__(parent)
        if icon:
            self.setTitle(f"{icon}  {title}")
        else:
            self.setTitle(title)

        self.layout = QFormLayout(self)
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(16, 20, 16, 16)
        self.layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

    def add_row(self, label: str, widget: QWidget):
        """Add a labeled row to the card."""
        lbl = QLabel(label)
        lbl.setMinimumWidth(100)
        self.layout.addRow(lbl, widget)


class InputPanel(QWidget):
    """
    Mission Control style input panel.

    Features:
    - Card-based input groups
    - Scroll area for flexible height
    - Large, readable input fields
    - Clear visual hierarchy

    Signals:
        run_clicked: Emitted when Run button is pressed
    """

    run_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Build the mission control input panel."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Container for cards
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(16)
        container_layout.setContentsMargins(16, 16, 16, 16)

        # === PROPELLANTS CARD ===
        prop_card = InputCard("Propellants", "🚀")

        self.fuel_combo = QComboBox()
        self.fuel_combo.addItems([
            "H2 (Hydrogen)",
            "CH4 (Methane)",
            "RP-1 (Kerosene)",
            "N2H4 (Hydrazine)",
            "MMH (Monomethyl Hydrazine)",
            "UDMH (Dimethyl Hydrazine)"
        ])
        self.fuel_combo.setCurrentIndex(0)
        self.fuel_combo.setToolTip("Select the fuel propellant")
        prop_card.add_row("Fuel:", self.fuel_combo)

        self.oxidizer_combo = QComboBox()
        self.oxidizer_combo.addItems([
            "O2 (LOX)",
            "N2O (Nitrous)",
            "N2O4 (NTO)",
            "H2O2 (Peroxide 90%)"
        ])
        self.oxidizer_combo.setCurrentIndex(0)
        self.oxidizer_combo.setToolTip("Select the oxidizer")
        prop_card.add_row("Oxidizer:", self.oxidizer_combo)

        self.of_ratio_spin = QDoubleSpinBox()
        self.of_ratio_spin.setRange(0.5, 20.0)
        self.of_ratio_spin.setValue(6.0)
        self.of_ratio_spin.setSingleStep(0.5)
        self.of_ratio_spin.setDecimals(2)
        self.of_ratio_spin.setSuffix("  :1")
        self.of_ratio_spin.setToolTip("Oxidizer-to-Fuel mass ratio\n• H2/O2: 5-8\n• CH4/O2: 3-4\n• RP-1/O2: 2.5-3")
        prop_card.add_row("O/F Ratio:", self.of_ratio_spin)

        container_layout.addWidget(prop_card)

        # === CHAMBER CARD ===
        chamber_card = InputCard("Chamber Conditions", "🔥")

        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(1.0, 500.0)
        self.pressure_spin.setValue(68.0)
        self.pressure_spin.setSingleStep(5.0)
        self.pressure_spin.setDecimals(1)
        self.pressure_spin.setSuffix("  bar")
        self.pressure_spin.setToolTip("Combustion chamber pressure (Pc)\nTypical: 50-250 bar")
        chamber_card.add_row("Pressure (Pc):", self.pressure_spin)

        self.throat_area_spin = QDoubleSpinBox()
        self.throat_area_spin.setRange(0.1, 10000.0)
        self.throat_area_spin.setValue(100.0)
        self.throat_area_spin.setSingleStep(10.0)
        self.throat_area_spin.setDecimals(1)
        self.throat_area_spin.setSuffix("  cm²")
        self.throat_area_spin.setToolTip("Nozzle throat cross-sectional area (At)")
        chamber_card.add_row("Throat Area (At):", self.throat_area_spin)

        container_layout.addWidget(chamber_card)

        # === NOZZLE CARD ===
        nozzle_card = InputCard("Nozzle Design", "⭐")

        self.expansion_spin = QDoubleSpinBox()
        self.expansion_spin.setRange(1.1, 200.0)
        self.expansion_spin.setValue(50.0)
        self.expansion_spin.setSingleStep(5.0)
        self.expansion_spin.setDecimals(1)
        self.expansion_spin.setPrefix("ε = ")
        self.expansion_spin.setToolTip("Expansion ratio (ε = Ae/At)\nVacuum: 50-100, Sea level: 10-20")
        nozzle_card.add_row("Expansion Ratio:", self.expansion_spin)

        self.ambient_combo = QComboBox()
        self.ambient_combo.addItems([
            "Vacuum (0 bar)",
            "Sea Level (1.01 bar)",
            "10 km Altitude (0.26 bar)"
        ])
        self.ambient_combo.setCurrentIndex(0)
        self.ambient_combo.setToolTip("Ambient pressure at nozzle exit")
        nozzle_card.add_row("Ambient:", self.ambient_combo)

        container_layout.addWidget(nozzle_card)

        # === EFFICIENCY CARD ===
        eff_card = InputCard("Efficiency Factors", "⚙️")

        self.eta_cstar_spin = QDoubleSpinBox()
        self.eta_cstar_spin.setRange(0.80, 1.00)
        self.eta_cstar_spin.setValue(0.97)
        self.eta_cstar_spin.setSingleStep(0.01)
        self.eta_cstar_spin.setDecimals(3)
        self.eta_cstar_spin.setToolTip("Combustion efficiency (η_c*)\nTypical: 0.94-0.99")
        eff_card.add_row("η_c* (Comb):", self.eta_cstar_spin)

        self.eta_cf_spin = QDoubleSpinBox()
        self.eta_cf_spin.setRange(0.80, 1.00)
        self.eta_cf_spin.setValue(0.98)
        self.eta_cf_spin.setSingleStep(0.01)
        self.eta_cf_spin.setDecimals(3)
        self.eta_cf_spin.setToolTip("Nozzle efficiency (η_Cf)\nTypical: 0.96-0.99")
        eff_card.add_row("η_Cf (Nozzle):", self.eta_cf_spin)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(5.0, 30.0)
        self.alpha_spin.setValue(15.0)
        self.alpha_spin.setSingleStep(1.0)
        self.alpha_spin.setDecimals(1)
        self.alpha_spin.setSuffix("  °")
        self.alpha_spin.setToolTip("Nozzle half-angle (α)\nDivergence loss: λ = (1+cos(α))/2")
        eff_card.add_row("Nozzle Angle:", self.alpha_spin)

        container_layout.addWidget(eff_card)

        # Add stretch to push everything up
        container_layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll, stretch=1)

        # === RUN BUTTON (Fixed at bottom) ===
        button_frame = QFrame()
        button_frame.setObjectName("runButtonFrame")
        button_layout = QVBoxLayout(button_frame)
        button_layout.setContentsMargins(16, 16, 16, 16)

        self.run_button = QPushButton("▶  RUN SIMULATION")
        self.run_button.setObjectName("runButton")
        self.run_button.setMinimumHeight(56)
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.clicked.connect(self.run_clicked.emit)
        button_layout.addWidget(self.run_button)

        main_layout.addWidget(button_frame)

    def get_ambient_pressure_bar(self) -> float:
        """Get ambient pressure from combo selection."""
        text = self.ambient_combo.currentText()
        if "Vacuum" in text:
            return 0.0
        elif "Sea Level" in text:
            return 1.01325
        elif "10 km" in text:
            return 0.26
        return 0.0

    def set_enabled(self, enabled: bool):
        """Enable or disable all inputs."""
        self.fuel_combo.setEnabled(enabled)
        self.oxidizer_combo.setEnabled(enabled)
        self.of_ratio_spin.setEnabled(enabled)
        self.pressure_spin.setEnabled(enabled)
        self.throat_area_spin.setEnabled(enabled)
        self.expansion_spin.setEnabled(enabled)
        self.ambient_combo.setEnabled(enabled)
        self.eta_cstar_spin.setEnabled(enabled)
        self.eta_cf_spin.setEnabled(enabled)
        self.alpha_spin.setEnabled(enabled)
        self.run_button.setEnabled(enabled)

        if enabled:
            self.run_button.setText("▶  RUN SIMULATION")
        else:
            self.run_button.setText("⏳  CALCULATING...")

    def validate_inputs(self) -> tuple:
        """
        Validate all inputs and return (is_valid, warnings).

        Returns:
            Tuple of (bool is_valid, list of warning strings)
        """
        warnings = []
        is_valid = True

        # Check O/F ratio
        of_ratio = self.of_ratio_spin.value()
        if of_ratio > 15.0:
            warnings.append(f"⚠️ O/F ratio ({of_ratio:.1f}) is very high")
        elif of_ratio < 1.0:
            warnings.append(f"⚠️ O/F ratio ({of_ratio:.1f}) is very low")

        # Check chamber pressure
        pc = self.pressure_spin.value()
        if pc < 10.0:
            warnings.append(f"⚠️ Low chamber pressure ({pc:.1f} bar)")
        elif pc > 300.0:
            warnings.append(f"⚠️ Very high pressure ({pc:.1f} bar)")

        # Check expansion ratio
        eps = self.expansion_spin.value()
        amb = self.get_ambient_pressure_bar()
        if eps > 100.0 and amb > 0.5:
            warnings.append(f"⚠️ High ε ({eps:.0f}) at sea level")

        return is_valid, warnings
