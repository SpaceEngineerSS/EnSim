"""
Thermal Analysis Widget - Heat flux and wall temperature visualization.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ThermalAnalysisWidget(QWidget):
    """
    Widget for thermal analysis visualization.

    Shows heat flux and wall temperature profiles with material limits.
    """

    analysis_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thermal_result = None
        self._material_limit = 700.0
        self._setup_ui()

    def _setup_ui(self):
        """Build the thermal analysis UI."""
        layout = QHBoxLayout(self)

        # Left panel - inputs
        from PyQt6.QtWidgets import QScrollArea, QFrame
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMaximumWidth(340)
        scroll_area.setMinimumWidth(300)

        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(12)

        # Material selection
        mat_group = QGroupBox("Wall Material")
        mat_layout = QFormLayout(mat_group)
        mat_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        mat_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.material_combo = QComboBox()
        from src.core.materials import get_material, list_materials
        for mat_name in list_materials():
            mat = get_material(mat_name)
            self.material_combo.addItem(f"{mat_name}") # Simpler name
        self.material_combo.currentIndexChanged.connect(self._on_material_changed)
        mat_layout.addRow("Material:", self.material_combo)

        # Conductivity display
        self.k_label = QLabel("--")
        self.k_label.setObjectName("calcValue")
        mat_layout.addRow("Conductivity:", self.k_label)

        self.t_melt_label = QLabel("--")
        self.t_melt_label.setObjectName("calcValue")
        mat_layout.addRow("Melt Point:", self.t_melt_label)

        self.t_limit_label = QLabel("--")
        self.t_limit_label.setObjectName("calcValue")
        mat_layout.addRow("Service Limit:", self.t_limit_label)

        input_layout.addWidget(mat_group)

        # Cooling parameters
        cool_group = QGroupBox("Cooling Parameters")
        cool_layout = QFormLayout(cool_group)
        cool_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        cool_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.wall_thick_spin = QDoubleSpinBox()
        self.wall_thick_spin.setRange(0.5, 20.0)
        self.wall_thick_spin.setValue(3.0)
        self.wall_thick_spin.setSuffix(" mm")
        cool_layout.addRow("Wall Thickness:", self.wall_thick_spin)

        self.coolant_temp_spin = QDoubleSpinBox()
        self.coolant_temp_spin.setRange(20, 1000)
        self.coolant_temp_spin.setValue(300)
        self.coolant_temp_spin.setSuffix(" K")
        cool_layout.addRow("Coolant Temp:", self.coolant_temp_spin)

        self.coolant_htc_spin = QDoubleSpinBox()
        self.coolant_htc_spin.setRange(100, 200000)
        self.coolant_htc_spin.setValue(10000)
        self.coolant_htc_spin.setSuffix(" W/(m²K)")
        cool_layout.addRow("Coolant HTC:", self.coolant_htc_spin)

        input_layout.addWidget(cool_group)

        # Analysis button
        self.analyze_btn = QPushButton("🚀 Run Thermal Analysis") # Updated icon
        self.analyze_btn.setObjectName("analyzeBtn")
        self.analyze_btn.setMinimumHeight(36)
        self.analyze_btn.clicked.connect(self.analysis_requested.emit)
        input_layout.addWidget(self.analyze_btn)

        # Status
        self.status_label = QLabel("System Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("notesText")
        input_layout.addWidget(self.status_label)

        input_layout.addStretch()
        scroll_area.setWidget(input_panel)
        layout.addWidget(scroll_area)

        # Right panel - plots
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        self.figure = Figure(figsize=(8, 8), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(plot_panel, stretch=1)

        self._create_empty_plots()
        self._on_material_changed()

    def _on_material_changed(self):
        """Update material properties display."""
        from src.core.materials import get_material, list_materials

        idx = self.material_combo.currentIndex()
        mat_names = list_materials()
        if 0 <= idx < len(mat_names):
            mat = get_material(mat_names[idx])
            self.k_label.setText(f"{mat.thermal_conductivity:.1f} W/(m·K)")
            self.t_melt_label.setText(f"{mat.melting_point:.0f} K")
            self.t_limit_label.setText(f"{mat.max_service_temp:.0f} K")
            self._material_limit = mat.max_service_temp

    def get_material_name(self) -> str:
        """Get selected material name."""
        from src.core.materials import list_materials
        idx = self.material_combo.currentIndex()
        return list_materials()[idx]

    def get_cooling_params(self) -> dict:
        """Get cooling parameters from UI."""
        return {
            'wall_thickness': self.wall_thick_spin.value() / 1000.0,  # mm to m
            'coolant_temp': self.coolant_temp_spin.value(),
            'coolant_htc': self.coolant_htc_spin.value()
        }

    def _create_empty_plots(self):
        """Create empty subplot structure."""
        self.figure.clear()

        self.ax_flux = self.figure.add_subplot(2, 1, 1)
        self.ax_temp = self.figure.add_subplot(2, 1, 2)

        for ax in [self.ax_flux, self.ax_temp]:
            ax.set_facecolor('#252525')
            ax.tick_params(colors='#888888')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.spines['right'].set_color('#444444')
            ax.grid(True, color='#333333', alpha=0.7)
            ax.text(0.5, 0.5, 'Run thermal analysis',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#555555', fontsize=12)

        self.ax_flux.set_title('Heat Flux Profile', color='white', fontweight='bold')
        self.ax_temp.set_title('Wall Temperature', color='white', fontweight='bold')

        self.figure.tight_layout()
        self.canvas.draw()

    def update_plots(self, thermal_result):
        """Update plots with thermal analysis results."""
        self._thermal_result = thermal_result

        self.ax_flux.clear()
        self.ax_temp.clear()

        x = thermal_result.x_position * 1000  # m to mm

        # Heat flux plot
        q_MW = thermal_result.heat_flux / 1e6  # W/m² to MW/m²
        self.ax_flux.plot(x, q_MW, color='#ff6b6b', linewidth=2, label='q (MW/m²)')
        self.ax_flux.fill_between(x, q_MW, alpha=0.3, color='#ff6b6b')
        self.ax_flux.axvline(x=0, color='#ffff00', linestyle='--', linewidth=1, label='Throat')
        self.ax_flux.set_xlabel('Axial Position (mm)', color='#cccccc')
        self.ax_flux.set_ylabel('Heat Flux (MW/m²)', color='#cccccc')
        self.ax_flux.set_title('Heat Flux Profile', color='white', fontweight='bold')
        self.ax_flux.legend(facecolor='#2d2d2d', edgecolor='#444444', labelcolor='#cccccc')

        # Temperature plot
        self.ax_temp.plot(x, thermal_result.wall_temp_gas, color='#ff4444',
                         linewidth=2, label='Gas-side Wall')
        self.ax_temp.plot(x, thermal_result.wall_temp_coolant, color='#4488ff',
                         linewidth=2, label='Coolant-side Wall')

        # Material limit line
        self.ax_temp.axhline(y=self._material_limit, color='#ff0000',
                            linestyle='--', linewidth=2, label=f'Limit ({self._material_limit:.0f}K)')

        # Fill danger zone
        max_temp = max(thermal_result.wall_temp_gas.max(), self._material_limit * 1.1)
        self.ax_temp.fill_between(x, self._material_limit, max_temp,
                                  alpha=0.2, color='#ff0000', label='MELT ZONE')

        self.ax_temp.axvline(x=0, color='#ffff00', linestyle='--', linewidth=1)
        self.ax_temp.set_xlabel('Axial Position (mm)', color='#cccccc')
        self.ax_temp.set_ylabel('Temperature (K)', color='#cccccc')
        self.ax_temp.set_title('Wall Temperature', color='white', fontweight='bold')
        self.ax_temp.legend(facecolor='#2d2d2d', edgecolor='#444444', labelcolor='#cccccc')

        for ax in [self.ax_flux, self.ax_temp]:
            ax.set_facecolor('#252525')
            ax.tick_params(colors='#888888')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.spines['right'].set_color('#444444')
            ax.grid(True, color='#333333', alpha=0.7)

        self.figure.tight_layout()
        self.canvas.draw()

        # Update status
        if thermal_result.is_safe:
            self.status_label.setText(
                f"✅ SAFE\nMax T: {thermal_result.max_wall_temp:.0f}K\n"
                f"Max q: {thermal_result.max_heat_flux/1e6:.1f} MW/m²"
            )
            self.status_label.setStyleSheet("color: #00ff88;")
        else:
            self.status_label.setText(
                f"🔥 MELT WARNING!\nMax T: {thermal_result.max_wall_temp:.0f}K > {self._material_limit:.0f}K\n"
                f"Critical at x = {thermal_result.critical_x*1000:.1f} mm"
            )
            self.status_label.setStyleSheet("color: #ff4444;")
