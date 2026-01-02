"""
Optimization Tools Widget.

Provides UI for various optimization capabilities:
- Nozzle expansion ratio optimization
- Stage mass allocation optimization
- Engine design optimization
- Trajectory optimization (gravity turn)
"""

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QTabWidget, QProgressBar,
    QTextEdit, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt6.QtGui import QFont

from ...core.optimization import (
    optimize_nozzle_expansion_ratio,
    optimize_stage_mass_allocation,
    optimize_engine_parameters,
    optimize_gravity_turn,
    optimize_propellant_load,
    OptimizationResult,
    TrajectoryConstraints
)


class OptimizationWorker(QThread):
    """Background worker for optimization calculations."""
    
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, opt_type: str, params: dict):
        super().__init__()
        self.opt_type = opt_type
        self.params = params
    
    def run(self):
        try:
            self.progress.emit(10)
            
            if self.opt_type == "nozzle":
                result = optimize_nozzle_expansion_ratio(**self.params)
            elif self.opt_type == "stage_mass":
                result = optimize_stage_mass_allocation(**self.params)
            elif self.opt_type == "engine":
                result = optimize_engine_parameters(**self.params)
            elif self.opt_type == "trajectory":
                result = optimize_gravity_turn(**self.params)
            elif self.opt_type == "propellant":
                result = optimize_propellant_load(**self.params)
            else:
                raise ValueError(f"Unknown optimization type: {self.opt_type}")
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class NozzleOptimizationTab(QWidget):
    """Tab for nozzle expansion ratio optimization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._worker = None
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        from PyQt6.QtWidgets import QScrollArea, QFrame
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # Input parameters
        input_group = QGroupBox("Input Parameters")
        input_layout = QFormLayout(input_group)
        input_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        input_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        def add_row(layout, label, widget):
            lbl = QLabel(label)
            lbl.setMinimumWidth(120)
            layout.addRow(lbl, widget)

        self.chamber_pressure = QDoubleSpinBox()
        self.chamber_pressure.setRange(1, 50)
        self.chamber_pressure.setValue(7)
        self.chamber_pressure.setSuffix(" MPa")
        self.chamber_pressure.setDecimals(1)
        
        self.ambient_pressure = QDoubleSpinBox()
        self.ambient_pressure.setRange(0, 0.15)
        self.ambient_pressure.setValue(0.101325)
        self.ambient_pressure.setSuffix(" MPa")
        self.ambient_pressure.setDecimals(6)
        
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(1.1, 1.4)
        self.gamma.setValue(1.2)
        self.gamma.setDecimals(2)
        
        self.weight_vac = QDoubleSpinBox()
        self.weight_vac.setRange(0, 1)
        self.weight_vac.setValue(0.5)
        self.weight_vac.setDecimals(2)
        
        self.weight_sl = QDoubleSpinBox()
        self.weight_sl.setRange(0, 1)
        self.weight_sl.setValue(0.5)
        self.weight_sl.setDecimals(2)
        
        add_row(input_layout, "Chamber Pressure:", self.chamber_pressure)
        add_row(input_layout, "Ambient Pressure:", self.ambient_pressure)
        add_row(input_layout, "Gamma (γ):", self.gamma)
        add_row(input_layout, "Vacuum Weight:", self.weight_vac)
        add_row(input_layout, "Sea-Level Weight:", self.weight_sl)
        
        layout.addWidget(input_group)
        
        # Run button
        self.run_btn = QPushButton("Optimize Nozzle")
        self.run_btn.setObjectName("optimizeBtn")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run_optimization)
        layout.addWidget(self.run_btn)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Results
        results_group = QGroupBox("Optimization Results")
        results_layout = QFormLayout(results_group)
        results_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        results_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.result_epsilon = QLabel("--")
        self.result_epsilon.setObjectName("summaryValue")
        add_row(results_layout, "Optimal ε:", self.result_epsilon)
        
        self.result_mach = QLabel("--")
        self.result_mach.setObjectName("calcValue")
        add_row(results_layout, "Exit Mach:", self.result_mach)
        
        self.result_pe = QLabel("--")
        self.result_pe.setObjectName("calcValue")
        add_row(results_layout, "Exit Pressure:", self.result_pe)
        
        self.result_cf = QLabel("--")
        self.result_cf.setObjectName("calcValue")
        add_row(results_layout, "Weighted Cf:", self.result_cf)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _run_optimization(self):
        params = {
            'chamber_pressure': self.chamber_pressure.value() * 1e6,
            'ambient_pressure': self.ambient_pressure.value() * 1e6,
            'gamma': self.gamma.value(),
            'weight_vacuum': self.weight_vac.value(),
            'weight_sealevel': self.weight_sl.value()
        }
        
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        
        self._worker = OptimizationWorker("nozzle", params)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.error.connect(self._on_error)
        self._worker.start()
    
    def _on_finished(self, result: OptimizationResult):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        
        if result.success or result.optimal_value > 0:
            self.result_epsilon.setText(f"{result.optimal_params['area_ratio']:.1f}")
            self.result_mach.setText(f"{result.optimal_params['exit_mach']:.2f}")
            self.result_pe.setText(f"{result.optimal_params['exit_pressure']/1e3:.1f} kPa")
            self.result_cf.setText(f"{result.optimal_value:.3f}")
        else:
            QMessageBox.warning(self, "Optimization Failed", result.message)
    
    def _on_error(self, error: str):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Optimization failed: {error}")


class StageMassOptimizationTab(QWidget):
    """Tab for stage mass allocation optimization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._worker = None
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        from PyQt6.QtWidgets import QScrollArea, QFrame
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # Input parameters
        input_group = QGroupBox("Input Parameters")
        input_layout = QFormLayout(input_group)
        input_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        input_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        def add_row(layout, label, widget):
            lbl = QLabel(label)
            lbl.setMinimumWidth(120)
            layout.addRow(lbl, widget)

        self.total_prop = QDoubleSpinBox()
        self.total_prop.setRange(1000, 10000000)
        self.total_prop.setValue(100000)
        self.total_prop.setSuffix(" kg")
        self.total_prop.setDecimals(0)
        
        self.num_stages = QSpinBox()
        self.num_stages.setRange(1, 5)
        self.num_stages.setValue(2)
        self.num_stages.valueChanged.connect(self._update_isp_inputs)
        
        self.payload = QDoubleSpinBox()
        self.payload.setRange(0, 500000)
        self.payload.setValue(5000)
        self.payload.setSuffix(" kg")
        self.payload.setDecimals(0)
        
        add_row(input_layout, "Total Propellant:", self.total_prop)
        add_row(input_layout, "Number of Stages:", self.num_stages)
        add_row(input_layout, "Payload Mass:", self.payload)
        
        layout.addWidget(input_group)
        
        # Stage Isps
        self.isp_group = QGroupBox("Stage Specific Impulses")
        self.isp_layout = QGridLayout(self.isp_group)
        self.isp_inputs = []
        self._update_isp_inputs()
        layout.addWidget(self.isp_group)
        
        # Run button
        self.run_btn = QPushButton("Optimize Mass Distribution")
        self.run_btn.setObjectName("optimizeBtn")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run_optimization)
        layout.addWidget(self.run_btn)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Results
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(10)
        
        # Total delta-v
        dv_layout = QHBoxLayout()
        label_dv = QLabel("Total ΔV:")
        label_dv.setMinimumWidth(120)
        label_dv.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dv_layout.addWidget(label_dv)
        
        self.result_dv = QLabel("--")
        self.result_dv.setObjectName("summaryValue")
        dv_layout.addWidget(self.result_dv)
        dv_layout.addStretch()
        results_layout.addLayout(dv_layout)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Stage", "Propellant (kg)", "Fraction"])
        self.results_table.setMinimumHeight(180)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _update_isp_inputs(self):
        # Clear existing
        for widget in self.isp_inputs:
            widget.deleteLater()
        self.isp_inputs.clear()
        
        # Clear layout
        while self.isp_layout.count():
            item = self.isp_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        n = self.num_stages.value()
        default_isps = [310, 340, 360, 380, 400]
        
        for i in range(n):
            label = QLabel(f"Stage {i+1} Isp:")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.isp_layout.addWidget(label, i, 0)
            
            spin = QDoubleSpinBox()
            spin.setRange(200, 500)
            spin.setValue(default_isps[i] if i < len(default_isps) else 320)
            spin.setSuffix(" s")
            spin.setDecimals(0)
            self.isp_layout.addWidget(spin, i, 1)
            self.isp_inputs.append(spin)
    
    def _run_optimization(self):
        isps = [spin.value() for spin in self.isp_inputs]
        
        params = {
            'total_propellant': self.total_prop.value(),
            'num_stages': self.num_stages.value(),
            'payload_mass': self.payload.value(),
            'stage_isps': isps
        }
        
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        
        self._worker = OptimizationWorker("stage_mass", params)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.error.connect(self._on_error)
        self._worker.start()
    
    def _on_finished(self, result: OptimizationResult):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        
        if result.success:
            self.result_dv.setText(f"{result.optimal_params['total_delta_v']:,.0f} m/s")
            
            masses = result.optimal_params['propellant_masses']
            fractions = result.optimal_params['propellant_fractions']
            
            self.results_table.setRowCount(len(masses))
            for i, (mass, frac) in enumerate(zip(masses, fractions)):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"Stage {i+1}"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{mass:,.0f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{frac*100:.1f}%"))
        else:
            QMessageBox.warning(self, "Optimization Failed", result.message)
    
    def _on_error(self, error: str):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Optimization failed: {error}")


class EngineOptimizationTab(QWidget):
    """Tab for engine design optimization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._worker = None
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        from PyQt6.QtWidgets import QScrollArea, QFrame
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # Input parameters
        input_group = QGroupBox("Target Performance")
        input_layout = QGridLayout(input_group)
        input_layout.setSpacing(10)
        
        def add_grid_row(grid_layout, label, widget, row):
            lbl = QLabel(label)
            lbl.setMinimumWidth(110)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid_layout.addWidget(lbl, row, 0)
            grid_layout.addWidget(widget, row, 1)

        self.target_thrust = QDoubleSpinBox()
        self.target_thrust.setRange(10, 50000)
        self.target_thrust.setValue(1000)
        self.target_thrust.setSuffix(" kN")
        self.target_thrust.setDecimals(0)
        add_grid_row(input_layout, "Target Thrust:", self.target_thrust, 0)
        
        self.target_isp = QDoubleSpinBox()
        self.target_isp.setRange(200, 500)
        self.target_isp.setValue(350)
        self.target_isp.setSuffix(" s")
        self.target_isp.setDecimals(0)
        add_grid_row(input_layout, "Target Isp:", self.target_isp, 1)
        
        self.propellant = QComboBox()
        self.propellant.addItems(["LOX/CH4", "LOX/RP1", "LOX/LH2"])
        add_grid_row(input_layout, "Propellant:", self.propellant, 2)
        
        layout.addWidget(input_group)
        
        # Pressure range
        range_group = QGroupBox("Search Range")
        range_layout = QGridLayout(range_group)
        range_layout.setSpacing(10)
        
        range_layout.addWidget(QLabel("Pc Min:"), 0, 0, Qt.AlignmentFlag.AlignRight)
        self.pc_min = QDoubleSpinBox()
        self.pc_min.setRange(1, 50)
        self.pc_min.setValue(5)
        self.pc_min.setSuffix(" MPa")
        range_layout.addWidget(self.pc_min, 0, 1)
        
        range_layout.addWidget(QLabel("Pc Max:"), 0, 2, Qt.AlignmentFlag.AlignRight)
        self.pc_max = QDoubleSpinBox()
        self.pc_max.setRange(1, 50)
        self.pc_max.setValue(25)
        self.pc_max.setSuffix(" MPa")
        range_layout.addWidget(self.pc_max, 0, 3)
        
        range_layout.addWidget(QLabel("O/F Min:"), 1, 0, Qt.AlignmentFlag.AlignRight)
        self.of_min = QDoubleSpinBox()
        self.of_min.setRange(1, 10)
        self.of_min.setValue(2.5)
        range_layout.addWidget(self.of_min, 1, 1)
        
        range_layout.addWidget(QLabel("O/F Max:"), 1, 2, Qt.AlignmentFlag.AlignRight)
        self.of_max = QDoubleSpinBox()
        self.of_max.setRange(1, 10)
        self.of_max.setValue(4.0)
        range_layout.addWidget(self.of_max, 1, 3)
        
        layout.addWidget(range_group)
        
        # Run button
        self.run_btn = QPushButton("Optimize Engine Design")
        self.run_btn.setObjectName("optimizeBtn")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run_optimization)
        layout.addWidget(self.run_btn)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Results
        results_group = QGroupBox("Optimal Engine Parameters")
        results_layout = QFormLayout(results_group)
        results_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        results_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        labels = [
            ("Chamber Pressure:", "result_pc", "summaryValue"),
            ("Mixture Ratio:", "result_of", "calcValue"),
            ("Chamber Temp:", "result_tc", "calcValue"),
            ("Throat Diameter:", "result_dt", "calcValue"),
            ("C*:", "result_cstar", "calcValue"),
            ("Estimated Isp:", "result_isp", "summaryValue")
        ]
        
        for i, (text, attr, style) in enumerate(labels):
            lbl = QLabel(text)
            lbl.setMinimumWidth(120)
            results_layout.setWidget(i, QFormLayout.ItemRole.LabelRole, lbl)
            
            label = QLabel("--")
            label.setObjectName(style)
            setattr(self, attr, label)
            results_layout.setWidget(i, QFormLayout.ItemRole.FieldRole, label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _run_optimization(self):
        params = {
            'target_thrust': self.target_thrust.value() * 1000,
            'target_isp': self.target_isp.value(),
            'propellant_type': self.propellant.currentText(),
            'chamber_pressure_range': (self.pc_min.value() * 1e6, self.pc_max.value() * 1e6),
            'mixture_ratio_range': (self.of_min.value(), self.of_max.value())
        }
        
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        
        self._worker = OptimizationWorker("engine", params)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.error.connect(self._on_error)
        self._worker.start()
    
    def _on_finished(self, result: OptimizationResult):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        
        if result.success:
            p = result.optimal_params
            self.result_pc.setText(f"{p['chamber_pressure']/1e6:.1f} MPa")
            self.result_of.setText(f"{p['mixture_ratio']:.2f}")
            self.result_tc.setText(f"{p['chamber_temperature']:.0f} K")
            self.result_dt.setText(f"{p['throat_diameter']*1000:.1f} mm")
            self.result_cstar.setText(f"{p['c_star']:.0f} m/s")
            self.result_isp.setText(f"{p['estimated_isp']:.0f} s")
        else:
            QMessageBox.warning(self, "Optimization Failed", result.message)
    
    def _on_error(self, error: str):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Optimization failed: {error}")


class OptimizationWidget(QWidget):
    """
    Main optimization tools widget with tabs for different optimization types.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("Optimization Tools")
        header.setObjectName("sectionHeader")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        header.setFont(font)
        layout.addWidget(header)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(NozzleOptimizationTab(), "Nozzle ε")
        self.tabs.addTab(StageMassOptimizationTab(), "Stage Mass")
        self.tabs.addTab(EngineOptimizationTab(), "Engine Design")
        
        layout.addWidget(self.tabs)

