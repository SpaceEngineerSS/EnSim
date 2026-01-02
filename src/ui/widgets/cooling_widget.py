"""
Regenerative Cooling Analysis Widget.

Provides UI for thermal analysis of rocket engine cooling:
- Cooling channel design
- Heat flux visualization
- Wall temperature prediction
- Coolant properties
"""

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QProgressBar, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter
)
from PyQt6.QtGui import QFont

import numpy as np

from ...core.cooling import (
    CoolantType, CoolingType,
    CoolingChannel, CoolingSystemDesign,
    design_cooling_channels, analyze_cooling_system,
    COOLANT_DATABASE
)


class CoolingDesignWorker(QThread):
    """Background worker for cooling design calculations."""
    
    finished = pyqtSignal(object, list)  # design, results
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
    
    def run(self):
        try:
            self.progress.emit(20)
            
            # Design cooling channels
            design = design_cooling_channels(
                thrust=self.params['thrust'],
                chamber_pressure=self.params['chamber_pressure'],
                chamber_temp=self.params['chamber_temp'],
                coolant=self.params['coolant'],
                wall_material=self.params['wall_material']
            )
            
            self.progress.emit(50)
            
            # Create simple nozzle profile for analysis
            throat_d = 2 * np.sqrt(design.channels.flow_area / np.pi / design.channels.num_channels)
            exit_d = throat_d * np.sqrt(self.params.get('area_ratio', 40))
            
            nozzle_profile = [
                (0.0, throat_d * 2),      # Chamber
                (0.1, throat_d),          # Throat
                (design.channels.length, exit_d)  # Exit
            ]
            
            chamber_conditions = {
                'T_chamber': self.params['chamber_temp'],
                'P_chamber': self.params['chamber_pressure'],
                'gamma': 1.2,
                'c_star': 1800
            }
            
            self.progress.emit(70)
            
            # Run thermal analysis
            results = analyze_cooling_system(
                design=design,
                nozzle_profile=nozzle_profile,
                chamber_conditions=chamber_conditions,
                num_stations=30
            )
            
            self.progress.emit(100)
            self.finished.emit(design, results)
            
        except Exception as e:
            self.error.emit(str(e))


class CoolingAnalysisWidget(QWidget):
    """
    Complete cooling analysis widget.
    
    Features:
    - Input engine parameters
    - Select coolant type and wall material
    - Auto-design cooling channels
    - View thermal analysis results
    - Heat flux and temperature plots
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._worker = None
        self._design = None
        self._results = None
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Splitter for left inputs / right results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # === Left: Input Panel ===
        from PyQt6.QtWidgets import QScrollArea, QFrame
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(12)
        
        # Engine parameters
        engine_group = QGroupBox("Engine Parameters")
        engine_layout = QGridLayout(engine_group)
        engine_layout.setSpacing(8)
        
        lbl_thrust = QLabel("Thrust:")
        lbl_thrust.setMinimumWidth(110)
        engine_layout.addWidget(lbl_thrust, 0, 0, Qt.AlignmentFlag.AlignRight)
        
        self.thrust_input = QDoubleSpinBox()
        self.thrust_input.setRange(10, 50000)
        self.thrust_input.setValue(1000)
        self.thrust_input.setSuffix(" kN")
        self.thrust_input.setDecimals(0)
        engine_layout.addWidget(self.thrust_input, 0, 1)
        
        lbl_pc = QLabel("Chamber Pressure:")
        engine_layout.addWidget(lbl_pc, 1, 0, Qt.AlignmentFlag.AlignRight)
        
        self.pc_input = QDoubleSpinBox()
        self.pc_input.setRange(1, 50)
        self.pc_input.setValue(7)
        self.pc_input.setSuffix(" MPa")
        self.pc_input.setDecimals(1)
        engine_layout.addWidget(self.pc_input, 1, 1)
        
        lbl_tc = QLabel("Chamber Temp:")
        engine_layout.addWidget(lbl_tc, 2, 0, Qt.AlignmentFlag.AlignRight)
        
        self.tc_input = QDoubleSpinBox()
        self.tc_input.setRange(2000, 5000)
        self.tc_input.setValue(3500)
        self.tc_input.setSuffix(" K")
        self.tc_input.setDecimals(0)
        engine_layout.addWidget(self.tc_input, 2, 1)
        
        lbl_eps = QLabel("Expansion Ratio:")
        engine_layout.addWidget(lbl_eps, 3, 0, Qt.AlignmentFlag.AlignRight)
        
        self.epsilon_input = QDoubleSpinBox()
        self.epsilon_input.setRange(5, 200)
        self.epsilon_input.setValue(40)
        self.epsilon_input.setDecimals(0)
        engine_layout.addWidget(self.epsilon_input, 3, 1)
        
        left_layout.addWidget(engine_group)
        
        # Cooling configuration
        cooling_group = QGroupBox("Cooling Configuration")
        cooling_layout = QGridLayout(cooling_group)
        cooling_layout.setSpacing(8)
        
        lbl_coolant = QLabel("Coolant:")
        cooling_layout.addWidget(lbl_coolant, 0, 0, Qt.AlignmentFlag.AlignRight)
        
        self.cool_combo = QComboBox() # Keep original name for compatibility if used elsewhere, wait, file used coolant_combo
        self.coolant_combo = QComboBox()
        self.coolant_combo.addItems([
            "RP-1 (Kerosene)",
            "LH2 (Liquid Hydrogen)",
            "LCH4 (Liquid Methane)",
            "LOX (Liquid Oxygen)",
            "Water (Testing)"
        ])
        self.coolant_combo.currentIndexChanged.connect(self._update_coolant_info)
        cooling_layout.addWidget(self.coolant_combo, 0, 1)
        
        lbl_mat = QLabel("Wall Material:")
        cooling_layout.addWidget(lbl_mat, 1, 0, Qt.AlignmentFlag.AlignRight)
        
        self.material_combo = QComboBox()
        self.material_combo.addItems([
            "Inconel 718",
            "OFHC Copper",
            "GRCop-84",
            "Haynes 230",
            "Monel 400"
        ])
        cooling_layout.addWidget(self.material_combo, 1, 1)
        
        left_layout.addWidget(cooling_group)
        
        # Coolant properties display
        props_group = QGroupBox("Coolant Properties")
        props_layout = QGridLayout(props_group)
        props_layout.setSpacing(10)
        
        self.coolant_density = QLabel("--")
        self.coolant_density.setObjectName("calcValue")
        self.coolant_cp = QLabel("--")
        self.coolant_cp.setObjectName("calcValue")
        self.coolant_k = QLabel("--")
        self.coolant_k.setObjectName("calcValue")
        self.coolant_bp = QLabel("--")
        self.coolant_bp.setObjectName("calcValue")
        
        props_layout.addWidget(QLabel("Density:"), 0, 0, Qt.AlignmentFlag.AlignRight)
        props_layout.addWidget(self.coolant_density, 0, 1)
        props_layout.addWidget(QLabel("Specific Heat:"), 1, 0, Qt.AlignmentFlag.AlignRight)
        props_layout.addWidget(self.coolant_cp, 1, 1)
        props_layout.addWidget(QLabel("Conductivity:"), 2, 0, Qt.AlignmentFlag.AlignRight)
        props_layout.addWidget(self.coolant_k, 2, 1)
        props_layout.addWidget(QLabel("Boiling Point:"), 3, 0, Qt.AlignmentFlag.AlignRight)
        props_layout.addWidget(self.coolant_bp, 3, 1)
        
        left_layout.addWidget(props_group)
        self._update_coolant_info()
        
        # Run button
        self.run_btn = QPushButton("Design & Analyze Cooling")
        self.run_btn.setObjectName("analyzeBtn")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run_analysis)
        left_layout.addWidget(self.run_btn)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)
        
        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)
        
        # === Right: Results Panel ===
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(12)
        
        # Design results
        design_group = QGroupBox("Channel Design")
        design_layout = QGridLayout(design_group)
        design_layout.setSpacing(8)
        
        labels = [
            ("Num. Channels:", "num_channels"),
            ("Channel Width:", "channel_width"),
            ("Channel Height:", "channel_height"),
            ("Wall Thickness:", "wall_thickness"),
            ("Hydraulic Dia.:", "hydraulic_dia"),
            ("Coolant Flow:", "coolant_flow"),
        ]
        
        for i, (text, attr) in enumerate(labels):
            lbl = QLabel(text)
            design_layout.addWidget(lbl, i, 0, Qt.AlignmentFlag.AlignRight)
            label = QLabel("--")
            label.setObjectName("calcValue")
            setattr(self, f"design_{attr}", label)
            design_layout.addWidget(label, i, 1)
        
        right_layout.addWidget(design_group)
        
        # Thermal results summary
        thermal_group = QGroupBox("Thermal Summary")
        thermal_layout = QGridLayout(thermal_group)
        thermal_layout.setSpacing(8)
        
        thermal_labels = [
            ("Peak Wall Temp:", "peak_tw", ""),
            ("Peak Heat Flux:", "peak_q", ""),
            ("Coolant ΔT:", "coolant_dt", ""),
            ("Min. Margin (Melt):", "margin_melt", ""),
            ("Min. Margin (Boil):", "margin_boil", ""),
        ]
        
        for i, (text, attr, _) in enumerate(thermal_labels):
            lbl = QLabel(text)
            thermal_layout.addWidget(lbl, i, 0, Qt.AlignmentFlag.AlignRight)
            label = QLabel("--")
            label.setObjectName("summaryValue")
            setattr(self, f"thermal_{attr}", label)
            thermal_layout.addWidget(label, i, 1)
        
        right_layout.addWidget(thermal_group)
        
        # Station-by-station results table
        table_group = QGroupBox("Station Analysis")
        table_layout = QVBoxLayout(table_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Position", "T_wall", "q (MW/m²)", "T_coolant", "Margin"
        ])
        self.results_table.setMinimumHeight(200)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.results_table)
        
        right_layout.addWidget(table_group)
        
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)
        splitter.setSizes([340, 660])
        
        layout.addWidget(splitter)
    
    def _get_coolant_type(self) -> CoolantType:
        """Get selected coolant type."""
        index = self.coolant_combo.currentIndex()
        mapping = {
            0: CoolantType.RP1,
            1: CoolantType.LH2,
            2: CoolantType.LCH4,
            3: CoolantType.LOX,
            4: CoolantType.WATER
        }
        return mapping.get(index, CoolantType.RP1)
    
    def _update_coolant_info(self):
        """Update coolant properties display."""
        coolant = self._get_coolant_type()
        props = COOLANT_DATABASE.get(coolant)
        
        if props:
            self.coolant_density.setText(f"{props.density:.0f} kg/m³")
            self.coolant_cp.setText(f"{props.specific_heat:.0f} J/(kg·K)")
            self.coolant_k.setText(f"{props.thermal_conductivity:.3f} W/(m·K)")
            self.coolant_bp.setText(f"{props.boiling_point:.1f} K")
    
    def _run_analysis(self):
        """Run cooling design and analysis."""
        params = {
            'thrust': self.thrust_input.value() * 1000,  # kN to N
            'chamber_pressure': self.pc_input.value() * 1e6,  # MPa to Pa
            'chamber_temp': self.tc_input.value(),
            'coolant': self._get_coolant_type(),
            'wall_material': self.material_combo.currentText(),
            'area_ratio': self.epsilon_input.value()
        }
        
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        
        self._worker = CoolingDesignWorker(params)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.error.connect(self._on_error)
        self._worker.start()
    
    def _on_finished(self, design: CoolingSystemDesign, results: list):
        """Handle analysis completion."""
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        
        self._design = design
        self._results = results
        
        # Update design display
        ch = design.channels
        self.design_num_channels.setText(str(ch.num_channels))
        self.design_channel_width.setText(f"{ch.width*1000:.2f} mm")
        self.design_channel_height.setText(f"{ch.height*1000:.2f} mm")
        self.design_wall_thickness.setText(f"{ch.wall_thickness*1000:.2f} mm")
        self.design_hydraulic_dia.setText(f"{ch.hydraulic_diameter*1000:.2f} mm")
        self.design_coolant_flow.setText(f"{design.coolant_mass_flow:.1f} kg/s")
        
        # Calculate thermal summary
        if results:
            peak_tw = max(r.wall_temp_gas_side for r in results)
            peak_q = max(r.heat_flux for r in results)
            coolant_dt = results[-1].coolant_temp - results[0].coolant_temp
            min_margin_melt = min(r.margin_to_melting for r in results)
            min_margin_boil = min(r.margin_to_boiling for r in results)
            
            self.thermal_peak_tw.setText(f"{peak_tw:.0f} K")
            self.thermal_peak_q.setText(f"{peak_q/1e6:.1f} MW/m²")
            self.thermal_coolant_dt.setText(f"{coolant_dt:.0f} K")
            
            # Color code margins
            if min_margin_melt > 200:
                self.thermal_margin_melt.setText(f"{min_margin_melt:.0f} K ✓")
                self.thermal_margin_melt.setStyleSheet("color: #00ff88;")
            else:
                self.thermal_margin_melt.setText(f"{min_margin_melt:.0f} K ⚠")
                self.thermal_margin_melt.setStyleSheet("color: #ff6b6b;")
            
            if min_margin_boil > 50:
                self.thermal_margin_boil.setText(f"{min_margin_boil:.0f} K ✓")
                self.thermal_margin_boil.setStyleSheet("color: #00ff88;")
            else:
                self.thermal_margin_boil.setText(f"{min_margin_boil:.0f} K ⚠")
                self.thermal_margin_boil.setStyleSheet("color: #ff6b6b;")
            
            # Fill results table
            self.results_table.setRowCount(len(results))
            for i, r in enumerate(results):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"{r.axial_position*100:.1f} cm"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{r.wall_temp_gas_side:.0f} K"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{r.heat_flux/1e6:.2f}"))
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{r.coolant_temp:.0f} K"))
                self.results_table.setItem(i, 4, QTableWidgetItem(f"{r.margin_to_melting:.0f} K"))
    
    def _on_error(self, error: str):
        """Handle analysis error."""
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"Cooling analysis failed:\n{error}")

