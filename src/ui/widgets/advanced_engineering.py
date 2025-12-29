"""
Advanced Engineering Widget - Mission Control Style.

Provides:
- MOC Nozzle Design visualization
- Engine Optimization controls
- Monte Carlo Reliability analysis

Phase 6 implementation with Mission Control styling.
"""

from typing import Optional, Dict
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QFrame,
    QFormLayout, QScrollArea, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from src.core.moc_solver import generate_mln_contour, export_contour_csv, export_mesh_vtk
from src.core.optimizer import (
    EngineOptimizer, MonteCarloAnalyzer, MonteCarloInput,
    OptimizationBounds, evaluate_engine_performance
)


class AdvancedCard(QGroupBox):
    """Styled card container for advanced engineering inputs."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        
        self.layout = QFormLayout(self)
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(16, 28, 16, 16)
        self.layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    
    def add_row(self, label: str, widget: QWidget):
        lbl = QLabel(label)
        lbl.setMinimumWidth(90)
        self.layout.addRow(lbl, widget)


def create_spinbox(min_val, max_val, default, suffix="", decimals=2, step=None):
    """Create a styled double spinbox."""
    spin = QDoubleSpinBox()
    spin.setRange(min_val, max_val)
    spin.setValue(default)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(f"  {suffix}")
    if step:
        spin.setSingleStep(step)
    return spin


def style_plot_axis(ax, title, bg_color='#0a0e14'):
    """Apply Mission Control styling to a matplotlib axis."""
    ax.set_facecolor(bg_color)
    ax.set_title(title, color='#ffffff', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#8899aa')
    ax.grid(True, color='#1a242e', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#2a3a4a')


class MOCDesignTab(QWidget):
    """MOC nozzle design sub-tab with Mission Control styling."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._contour = None
        self._mesh = None
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Left: Controls with scroll
        left_frame = QFrame()
        left_frame.setMaximumWidth(400)
        left_frame.setMinimumWidth(350)
        left_frame.setStyleSheet("background: #0a0e14;")
        left_main = QVBoxLayout(left_frame)
        left_main.setContentsMargins(0, 0, 0, 0)
        left_main.setSpacing(0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        container = QWidget()
        controls_layout = QVBoxLayout(container)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        
        # Design parameters card
        params_card = AdvancedCard("Design Parameters")
        
        self.m_exit_spin = create_spinbox(1.5, 10.0, 3.0, step=0.1)
        self.m_exit_spin.setToolTip("Design exit Mach number")
        params_card.add_row("Exit Mach:", self.m_exit_spin)
        
        self.gamma_spin = create_spinbox(1.1, 1.67, 1.2, step=0.01)
        self.gamma_spin.setToolTip("Ratio of specific heats γ")
        params_card.add_row("Gamma (γ):", self.gamma_spin)
        
        self.throat_r_spin = create_spinbox(0.01, 1.0, 0.05, "m", step=0.01)
        self.throat_r_spin.setToolTip("Throat radius")
        params_card.add_row("Throat R:", self.throat_r_spin)
        
        self.n_chars_spin = QSpinBox()
        self.n_chars_spin.setRange(5, 50)
        self.n_chars_spin.setValue(15)
        self.n_chars_spin.setToolTip("Number of characteristic lines")
        params_card.add_row("Char Lines:", self.n_chars_spin)
        
        controls_layout.addWidget(params_card)
        
        # Generate button
        self.generate_btn = QPushButton("🔬  GENERATE MLN CONTOUR")
        self.generate_btn.setObjectName("runButton")
        self.generate_btn.setMinimumHeight(48)
        self.generate_btn.clicked.connect(self._generate_contour)
        controls_layout.addWidget(self.generate_btn)
        
        # Export card
        export_card = AdvancedCard("Export")
        
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.export_csv_btn.setEnabled(False)
        export_card.layout.addRow(self.export_csv_btn)
        
        self.export_vtk_btn = QPushButton("Export VTK")
        self.export_vtk_btn.clicked.connect(self._export_vtk)
        self.export_vtk_btn.setEnabled(False)
        export_card.layout.addRow(self.export_vtk_btn)
        
        controls_layout.addWidget(export_card)
        
        # Results label
        self.results_label = QLabel("Results will appear here...")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("color: #8899aa; padding: 12px;")
        controls_layout.addWidget(self.results_label)
        
        controls_layout.addStretch()
        
        scroll.setWidget(container)
        left_main.addWidget(scroll)
        layout.addWidget(left_frame)
        
        # Right: Plot
        self.figure = Figure(figsize=(8, 6), facecolor='#0a0e14')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
    
    def _generate_contour(self):
        """Generate MLN contour using MOC."""
        M_exit = self.m_exit_spin.value()
        gamma = self.gamma_spin.value()
        throat_r = self.throat_r_spin.value()
        n_chars = self.n_chars_spin.value()
        
        try:
            self._contour, self._mesh = generate_mln_contour(
                M_exit=M_exit,
                gamma=gamma,
                throat_radius=throat_r,
                n_char_lines=n_chars
            )
            
            self._plot_results()
            self._update_results_label()
            
            self.export_csv_btn.setEnabled(True)
            self.export_vtk_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"MOC generation failed: {e}")
    
    def _plot_results(self):
        """Plot MOC mesh and contour with Mission Control styling."""
        self.figure.clear()
        
        ax = self.figure.add_subplot(1, 1, 1)
        style_plot_axis(ax, 'MINIMUM LENGTH NOZZLE (MOC)')
        
        # Plot characteristic lines
        if self._mesh and self._mesh.x_mesh is not None:
            for i in range(self._mesh.x_mesh.shape[0]):
                valid = ~np.isnan(self._mesh.x_mesh[i])
                ax.plot(self._mesh.x_mesh[i][valid], 
                       self._mesh.y_mesh[i][valid],
                       color='#2a3a4a', linewidth=0.5, alpha=0.7)
        
        # Plot contour
        if self._contour is not None:
            ax.plot(self._contour.x, self._contour.y, 
                   color='#00d4ff', linewidth=3, label='MLN Contour')
            ax.plot(self._contour.x, -self._contour.y, 
                   color='#00d4ff', linewidth=3)
            ax.axhline(y=0, color='#556677', linewidth=0.5, linestyle='--')
        
        ax.set_xlabel('Axial Position (m)', color='#8899aa')
        ax.set_ylabel('Radius (m)', color='#8899aa')
        ax.set_aspect('equal')
        ax.legend(facecolor='#141b22', edgecolor='#2a3a4a', labelcolor='#ffffff')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _update_results_label(self):
        """Update results text."""
        if self._contour:
            text = f"""<b style="color:#00d4ff;">MLN Design Results:</b><br>
            <span style="color:#8899aa;">Exit Mach:</span> <span style="color:#00ff9d;">{self._contour.M_exit:.2f}</span><br>
            <span style="color:#8899aa;">Throat R:</span> <span style="color:#00ff9d;">{self._contour.throat_radius*1000:.1f} mm</span><br>
            <span style="color:#8899aa;">Exit R:</span> <span style="color:#00ff9d;">{self._contour.exit_radius*1000:.1f} mm</span><br>
            <span style="color:#8899aa;">Length:</span> <span style="color:#00ff9d;">{self._contour.length*1000:.1f} mm</span><br>
            <span style="color:#8899aa;">ε:</span> <span style="color:#00ff9d;">{(self._contour.exit_radius/self._contour.throat_radius)**2:.1f}</span>
            """
            self.results_label.setText(text)
    
    def _export_csv(self):
        if not self._contour:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Contour", "mln_contour.csv", "CSV Files (*.csv)")
        if path:
            export_contour_csv(self._contour, path)
    
    def _export_vtk(self):
        if not self._mesh:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Mesh", "moc_mesh.vtk", "VTK Files (*.vtk)")
        if path:
            try:
                export_mesh_vtk(self._mesh, path)
            except ImportError:
                QMessageBox.warning(self, "VTK Export", "PyVista is required for VTK export.")


class OptimizationTab(QWidget):
    """Engine optimization sub-tab with Mission Control styling."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Left: Controls
        left_frame = QFrame()
        left_frame.setMaximumWidth(400)
        left_frame.setMinimumWidth(350)
        left_frame.setStyleSheet("background: #0a0e14;")
        left_main = QVBoxLayout(left_frame)
        left_main.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        container = QWidget()
        controls_layout = QVBoxLayout(container)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        
        # Objective card
        obj_card = AdvancedCard("Optimization Objective")
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(["Maximize Isp", "Maximize T/W Ratio"])
        obj_card.layout.addRow(self.objective_combo)
        controls_layout.addWidget(obj_card)
        
        # Bounds card
        bounds_card = AdvancedCard("Parameter Bounds")
        
        self.pc_min_spin = create_spinbox(1, 50, 5, "MPa")
        bounds_card.add_row("Pc Min:", self.pc_min_spin)
        
        self.pc_max_spin = create_spinbox(1, 50, 20, "MPa")
        bounds_card.add_row("Pc Max:", self.pc_max_spin)
        
        self.eps_min_spin = create_spinbox(5, 200, 20, decimals=0)
        bounds_card.add_row("ε Min:", self.eps_min_spin)
        
        self.eps_max_spin = create_spinbox(5, 500, 100, decimals=0)
        bounds_card.add_row("ε Max:", self.eps_max_spin)
        
        controls_layout.addWidget(bounds_card)
        
        # Fixed params card
        fixed_card = AdvancedCard("Fixed Parameters")
        
        self.gamma_spin = create_spinbox(1.1, 1.67, 1.2)
        fixed_card.add_row("Gamma:", self.gamma_spin)
        
        self.temp_spin = create_spinbox(1000, 5000, 3500, "K", decimals=0)
        fixed_card.add_row("T_chamber:", self.temp_spin)
        
        self.mw_spin = create_spinbox(1, 50, 18, "g/mol")
        fixed_card.add_row("Mean MW:", self.mw_spin)
        
        controls_layout.addWidget(fixed_card)
        
        # Optimize button
        self.optimize_btn = QPushButton("⚡  OPTIMIZE ENGINE")
        self.optimize_btn.setObjectName("runButton")
        self.optimize_btn.setMinimumHeight(48)
        self.optimize_btn.clicked.connect(self._run_optimization)
        controls_layout.addWidget(self.optimize_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        # Results table
        self.results_table = QTableWidget(5, 2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setMaximumHeight(200)
        controls_layout.addWidget(self.results_table)
        
        controls_layout.addStretch()
        
        scroll.setWidget(container)
        left_main.addWidget(scroll)
        layout.addWidget(left_frame)
        
        # Right: Plot
        self.figure = Figure(figsize=(6, 5), facecolor='#0a0e14')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
    
    def _run_optimization(self):
        objective = 'isp' if 'Isp' in self.objective_combo.currentText() else 'thrust_weight'
        
        bounds = OptimizationBounds(
            pc_range=(self.pc_min_spin.value() * 1e6, self.pc_max_spin.value() * 1e6),
            of_range=(3.0, 8.0),
            epsilon_range=(self.eps_min_spin.value(), self.eps_max_spin.value())
        )
        
        optimizer = EngineOptimizer(
            gamma=self.gamma_spin.value(),
            T_chamber=self.temp_spin.value(),
            mean_mw=self.mw_spin.value(),
            Pa=0
        )
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            self._result = optimizer.optimize(objective=objective, bounds=bounds, maxiter=100)
            self._update_results()
            self._plot_convergence()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Optimization failed: {e}")
        finally:
            self.progress_bar.setVisible(False)
    
    def _update_results(self):
        if not self._result:
            return
        
        rows = [
            ("Optimal Isp", f"{self._result.objective_value:.1f} s"),
            ("Chamber P", f"{self._result.optimal_params['Pc']/1e6:.2f} MPa"),
            ("Expansion ε", f"{self._result.optimal_params['epsilon']:.1f}"),
            ("O/F Ratio", f"{self._result.optimal_params['OF']:.2f}"),
            ("Converged", "✅ Yes" if self._result.success else "❌ No"),
        ]
        
        for i, (param, value) in enumerate(rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(param))
            self.results_table.setItem(i, 1, QTableWidgetItem(value))
    
    def _plot_convergence(self):
        self.figure.clear()
        
        if not self._result or not self._result.history:
            return
        
        ax = self.figure.add_subplot(1, 1, 1)
        style_plot_axis(ax, 'OPTIMIZATION CONVERGENCE')
        
        isp_history = [h['isp'] for h in self._result.history]
        ax.plot(isp_history, color='#00d4ff', linewidth=2)
        ax.set_xlabel('Iteration', color='#8899aa')
        ax.set_ylabel('Isp (s)', color='#8899aa')
        
        self.figure.tight_layout()
        self.canvas.draw()


class MonteCarloTab(QWidget):
    """Monte Carlo reliability analysis with Mission Control styling."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Left: Controls
        left_frame = QFrame()
        left_frame.setMaximumWidth(400)
        left_frame.setMinimumWidth(350)
        left_frame.setStyleSheet("background: #0a0e14;")
        left_main = QVBoxLayout(left_frame)
        left_main.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        container = QWidget()
        controls_layout = QVBoxLayout(container)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        
        # Nominal values card
        nom_card = AdvancedCard("Nominal Values")
        
        self.pc_spin = create_spinbox(1, 50, 10, "MPa")
        nom_card.add_row("Pc:", self.pc_spin)
        
        self.at_spin = create_spinbox(1, 1000, 100, "cm²")
        nom_card.add_row("At:", self.at_spin)
        
        self.eps_spin = create_spinbox(5, 200, 50, decimals=0)
        nom_card.add_row("ε:", self.eps_spin)
        
        controls_layout.addWidget(nom_card)
        
        # Uncertainties card
        unc_card = AdvancedCard("Uncertainties (σ)")
        
        self.pc_sigma_spin = create_spinbox(0.1, 10, 2.0, "%")
        unc_card.add_row("Pc σ:", self.pc_sigma_spin)
        
        self.at_sigma_spin = create_spinbox(0.1, 10, 1.0, "%")
        unc_card.add_row("At σ:", self.at_sigma_spin)
        
        controls_layout.addWidget(unc_card)
        
        # Samples card
        samples_card = AdvancedCard("Simulation")
        
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(100, 10000)
        self.n_samples_spin.setValue(1000)
        self.n_samples_spin.setSingleStep(100)
        samples_card.add_row("Samples:", self.n_samples_spin)
        
        controls_layout.addWidget(samples_card)
        
        # Run button
        self.run_btn = QPushButton("🎲  RUN MONTE CARLO")
        self.run_btn.setObjectName("runButton")
        self.run_btn.setMinimumHeight(48)
        self.run_btn.clicked.connect(self._run_monte_carlo)
        controls_layout.addWidget(self.run_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        # Stats table
        self.stats_table = QTableWidget(6, 2)
        self.stats_table.setHorizontalHeaderLabels(["Statistic", "Value"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setMaximumHeight(220)
        controls_layout.addWidget(self.stats_table)
        
        controls_layout.addStretch()
        
        scroll.setWidget(container)
        left_main.addWidget(scroll)
        layout.addWidget(left_frame)
        
        # Right: Histograms
        self.figure = Figure(figsize=(8, 6), facecolor='#0a0e14')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
    
    def _run_monte_carlo(self):
        inputs = MonteCarloInput(
            Pc_nominal=self.pc_spin.value() * 1e6,
            At_nominal=self.at_spin.value() * 1e-4,
            Pc_sigma=self.pc_sigma_spin.value() / 100,
            At_sigma=self.at_sigma_spin.value() / 100,
            epsilon=self.eps_spin.value(),
            gamma=1.2,
            T_chamber=3500,
            mean_mw=18
        )
        
        n_samples = self.n_samples_spin.value()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            mc = MonteCarloAnalyzer(n_workers=1)
            self._result = mc.run_sequential(inputs, n_samples=n_samples, seed=42)
            self._update_stats()
            self._plot_histograms()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Monte Carlo failed: {e}")
        finally:
            self.progress_bar.setVisible(False)
    
    def _update_stats(self):
        if not self._result:
            return
        
        rows = [
            ("Samples", f"{self._result.n_samples}"),
            ("Runtime", f"{self._result.runtime_seconds:.2f} s"),
            ("Thrust μ", f"{self._result.thrust_mean/1000:.1f} kN"),
            ("Thrust σ", f"{self._result.thrust_std/1000:.2f} kN"),
            ("Isp μ", f"{self._result.isp_mean:.1f} s"),
            ("Isp σ", f"{self._result.isp_std:.2f} s"),
        ]
        
        for i, (stat, value) in enumerate(rows):
            self.stats_table.setItem(i, 0, QTableWidgetItem(stat))
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))
    
    def _plot_histograms(self):
        self.figure.clear()
        
        if not self._result:
            return
        
        # Thrust histogram
        ax1 = self.figure.add_subplot(1, 2, 1)
        style_plot_axis(ax1, 'THRUST DISTRIBUTION')
        ax1.hist(self._result.thrust_distribution / 1000, bins=30, 
                color='#00d4ff', alpha=0.8, edgecolor='#33e0ff')
        ax1.axvline(self._result.thrust_mean / 1000, color='#ff3366', 
                   linestyle='--', linewidth=2, label=f'μ = {self._result.thrust_mean/1000:.1f}')
        ax1.set_xlabel('Thrust (kN)', color='#8899aa')
        ax1.set_ylabel('Frequency', color='#8899aa')
        ax1.legend(facecolor='#141b22', edgecolor='#2a3a4a', labelcolor='#ffffff')
        
        # Isp histogram
        ax2 = self.figure.add_subplot(1, 2, 2)
        style_plot_axis(ax2, 'ISP DISTRIBUTION')
        ax2.hist(self._result.isp_distribution, bins=30, 
                color='#00ff9d', alpha=0.8, edgecolor='#44ffaa')
        ax2.axvline(self._result.isp_mean, color='#ff3366', 
                   linestyle='--', linewidth=2, label=f'μ = {self._result.isp_mean:.1f}')
        ax2.set_xlabel('Isp (s)', color='#8899aa')
        ax2.set_ylabel('Frequency', color='#8899aa')
        ax2.legend(facecolor='#141b22', edgecolor='#2a3a4a', labelcolor='#ffffff')
        
        self.figure.tight_layout()
        self.canvas.draw()


class AdvancedEngineeringWidget(QWidget):
    """
    Advanced Engineering Widget with Mission Control styling.
    
    Contains sub-tabs for:
    - MOC Nozzle Design
    - Engine Optimization
    - Monte Carlo Reliability
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Sub-tabs
        self.tabs = QTabWidget()
        
        self.moc_tab = MOCDesignTab()
        self.tabs.addTab(self.moc_tab, "🔬 Nozzle Design (MOC)")
        
        self.opt_tab = OptimizationTab()
        self.tabs.addTab(self.opt_tab, "⚡ Optimization")
        
        self.mc_tab = MonteCarloTab()
        self.tabs.addTab(self.mc_tab, "🎲 Reliability")
        
        layout.addWidget(self.tabs)
