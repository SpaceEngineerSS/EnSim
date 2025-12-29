"""
Mission Analysis Widget - Thrust and Isp vs altitude visualization.
"""

from typing import Optional
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QPushButton
)
from PyQt6.QtCore import pyqtSignal


class MissionAnalysisWidget(QWidget):
    """
    Widget for mission/trajectory performance analysis.
    
    Shows thrust, Isp, and flow status vs altitude.
    """
    
    analysis_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mission_result = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the mission analysis UI."""
        layout = QHBoxLayout(self)
        
        # Left panel - inputs
        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)
        input_panel.setMaximumWidth(280)
        
        # Mission parameters
        mission_group = QGroupBox("Mission Parameters")
        mission_layout = QFormLayout(mission_group)
        
        self.max_alt_spin = QDoubleSpinBox()
        self.max_alt_spin.setRange(10, 200)
        self.max_alt_spin.setValue(100)
        self.max_alt_spin.setSuffix(" km")
        mission_layout.addRow("Max Altitude:", self.max_alt_spin)
        
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.5, 10)
        self.step_spin.setValue(1.0)
        self.step_spin.setSuffix(" km")
        mission_layout.addRow("Step Size:", self.step_spin)
        
        input_layout.addWidget(mission_group)
        
        # Analysis button
        self.analyze_btn = QPushButton("🚀 Run Mission Analysis")
        self.analyze_btn.clicked.connect(self.analysis_requested.emit)
        input_layout.addWidget(self.analyze_btn)
        
        # Results summary
        result_group = QGroupBox("Key Altitudes")
        result_layout = QFormLayout(result_group)
        
        self.opt_alt_label = QLabel("--")
        result_layout.addRow("Optimal Expansion:", self.opt_alt_label)
        
        self.sep_alt_label = QLabel("--")
        result_layout.addRow("Flow Separation:", self.sep_alt_label)
        
        self.max_thrust_label = QLabel("--")
        result_layout.addRow("Max Thrust:", self.max_thrust_label)
        
        self.max_isp_label = QLabel("--")
        result_layout.addRow("Max Isp:", self.max_isp_label)
        
        input_layout.addWidget(result_group)
        
        # Atmosphere info
        atm_group = QGroupBox("Atmosphere (Sample)")
        atm_layout = QFormLayout(atm_group)
        
        self.sl_p_label = QLabel("101.3 kPa")
        atm_layout.addRow("Sea Level P:", self.sl_p_label)
        
        self.alt10_p_label = QLabel("26.5 kPa")
        atm_layout.addRow("10 km P:", self.alt10_p_label)
        
        self.alt30_p_label = QLabel("1.2 kPa")
        atm_layout.addRow("30 km P:", self.alt30_p_label)
        
        input_layout.addWidget(atm_group)
        
        input_layout.addStretch()
        layout.addWidget(input_panel)
        
        # Right panel - plots
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        
        self.figure = Figure(figsize=(8, 8), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        layout.addWidget(plot_panel, stretch=1)
        
        self._create_empty_plots()
    
    def get_mission_params(self) -> dict:
        """Get mission parameters from UI."""
        return {
            'max_altitude': self.max_alt_spin.value() * 1000,  # km to m
            'step_size': self.step_spin.value() * 1000  # km to m
        }
    
    def _create_empty_plots(self):
        """Create empty subplot structure."""
        self.figure.clear()
        
        self.ax_thrust = self.figure.add_subplot(3, 1, 1)
        self.ax_isp = self.figure.add_subplot(3, 1, 2)
        self.ax_status = self.figure.add_subplot(3, 1, 3)
        
        for ax in [self.ax_thrust, self.ax_isp, self.ax_status]:
            ax.set_facecolor('#252525')
            ax.tick_params(colors='#888888')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.spines['right'].set_color('#444444')
            ax.grid(True, color='#333333', alpha=0.7)
            ax.text(0.5, 0.5, 'Run mission analysis',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#555555', fontsize=12)
        
        self.ax_thrust.set_title('Thrust vs Altitude', color='white', fontweight='bold')
        self.ax_isp.set_title('Specific Impulse vs Altitude', color='white', fontweight='bold')
        self.ax_status.set_title('Flow Status (Pe/Pa)', color='white', fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_plots(self, mission_result):
        """Update plots with mission analysis results."""
        self._mission_result = mission_result
        
        for ax in [self.ax_thrust, self.ax_isp, self.ax_status]:
            ax.clear()
        
        alt_km = mission_result.altitudes / 1000  # m to km
        
        # Thrust plot
        thrust_kN = mission_result.thrust / 1000  # N to kN
        self.ax_thrust.plot(alt_km, thrust_kN, color='#00ff88', linewidth=2)
        self.ax_thrust.fill_between(alt_km, thrust_kN, alpha=0.2, color='#00ff88')
        
        # Mark max thrust
        max_idx = np.argmax(mission_result.thrust)
        self.ax_thrust.axvline(x=alt_km[max_idx], color='#ffff00', linestyle='--', linewidth=1)
        self.ax_thrust.scatter([alt_km[max_idx]], [thrust_kN[max_idx]], 
                              color='#ffff00', s=100, zorder=5, marker='*')
        
        self.ax_thrust.set_xlabel('Altitude (km)', color='#cccccc')
        self.ax_thrust.set_ylabel('Thrust (kN)', color='#cccccc')
        self.ax_thrust.set_title('Thrust vs Altitude', color='white', fontweight='bold')
        
        # Isp plot
        self.ax_isp.plot(alt_km, mission_result.isp, color='#00a8ff', linewidth=2)
        self.ax_isp.fill_between(alt_km, mission_result.isp, alpha=0.2, color='#00a8ff')
        
        self.ax_isp.set_xlabel('Altitude (km)', color='#cccccc')
        self.ax_isp.set_ylabel('Isp (s)', color='#cccccc')
        self.ax_isp.set_title('Specific Impulse vs Altitude', color='white', fontweight='bold')
        
        # Flow status plot
        pe_pa = mission_result.pressure_ratio
        colors = []
        for status in mission_result.flow_status:
            if status == 'separated':
                colors.append('#ff4444')
            elif status == 'warning':
                colors.append('#ffaa00')
            else:
                colors.append('#00ff88')
        
        self.ax_status.scatter(alt_km, pe_pa, c=colors, s=20, alpha=0.8)
        self.ax_status.axhline(y=0.4, color='#ff4444', linestyle='--', linewidth=2, label='Separation (0.4)')
        self.ax_status.axhline(y=1.0, color='#00ff88', linestyle='--', linewidth=2, label='Optimal (1.0)')
        
        # Mark optimal expansion altitude
        if mission_result.optimal_altitude < mission_result.altitudes[-1]:
            self.ax_status.axvline(x=mission_result.optimal_altitude/1000, 
                                  color='#00ff88', linestyle=':', linewidth=2)
        
        self.ax_status.set_xlabel('Altitude (km)', color='#cccccc')
        self.ax_status.set_ylabel('Pe/Pa', color='#cccccc')
        self.ax_status.set_title('Flow Status (Pe/Pa)', color='white', fontweight='bold')
        self.ax_status.legend(facecolor='#2d2d2d', edgecolor='#444444', labelcolor='#cccccc')
        self.ax_status.set_yscale('log')
        
        for ax in [self.ax_thrust, self.ax_isp, self.ax_status]:
            ax.set_facecolor('#252525')
            ax.tick_params(colors='#888888')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.spines['right'].set_color('#444444')
            ax.grid(True, color='#333333', alpha=0.7)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update summary labels
        self.opt_alt_label.setText(f"{mission_result.optimal_altitude/1000:.1f} km")
        
        if mission_result.separation_altitude is not None:
            self.sep_alt_label.setText(f"< {mission_result.separation_altitude/1000:.1f} km")
            self.sep_alt_label.setStyleSheet("color: #ff4444;")
        else:
            self.sep_alt_label.setText("None (always attached)")
            self.sep_alt_label.setStyleSheet("color: #00ff88;")
        
        self.max_thrust_label.setText(f"{mission_result.max_thrust_altitude/1000:.1f} km")
        self.max_isp_label.setText(f"{mission_result.max_isp_altitude/1000:.1f} km")
