"""
Vehicle Designer Widget - Mission Control Style.

Rocket design and flight simulation UI with card-based layout
and scroll area for configuration panel.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class VehicleCard(QGroupBox):
    """Styled card container for vehicle component groups."""

    def __init__(self, title: str, icon: str = "", parent=None):
        super().__init__(parent)
        if icon:
            self.setTitle(f"{icon}  {title}")
        else:
            self.setTitle(title)

        self.layout = QFormLayout(self)
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(16, 28, 16, 16)
        self.layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

    def add_row(self, label: str, widget: QWidget):
        """Add a labeled row to the card."""
        lbl = QLabel(label)
        lbl.setMinimumWidth(90)
        self.layout.addRow(lbl, widget)


class VehicleDesignerWidget(QWidget):
    """
    Mission Control style vehicle designer.

    Features:
    - Card-based component configuration
    - Scrollable left panel
    - Live rocket diagram with CP/CG markers
    - Flight simulation results
    """

    launch_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rocket = None
        self._flight_result = None
        self._aero_result = None
        self._setup_ui()
        self._create_default_rocket()

    def _create_spinbox(self, min_val, max_val, default, suffix="", decimals=2, step=None):
        """Create a styled double spinbox."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(f"  {suffix}")
        if step:
            spin.setSingleStep(step)
        spin.valueChanged.connect(self._update_rocket)
        return spin

    def _setup_ui(self):
        """Build the vehicle designer UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # === Left Panel - Configuration ===
        left_frame = QFrame()
        left_frame.setMaximumWidth(420)
        left_frame.setMinimumWidth(380)
        left_frame.setStyleSheet("background: #0a0e14;")
        left_main = QVBoxLayout(left_frame)
        left_main.setContentsMargins(0, 0, 0, 0)
        left_main.setSpacing(0)

        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(12)
        container_layout.setContentsMargins(12, 12, 12, 12)

        # === NOSE CONE CARD ===
        nose_card = VehicleCard("Nose Cone", "🔺")

        self.nose_shape_combo = QComboBox()
        self.nose_shape_combo.addItems(["Ogive", "Conical", "Elliptical", "Parabolic"])
        self.nose_shape_combo.currentIndexChanged.connect(self._update_rocket)
        nose_card.add_row("Shape:", self.nose_shape_combo)

        self.nose_length_spin = self._create_spinbox(0.1, 2.0, 0.25, "m")
        nose_card.add_row("Length:", self.nose_length_spin)

        self.nose_mass_spin = self._create_spinbox(0.1, 10.0, 0.3, "kg")
        nose_card.add_row("Mass:", self.nose_mass_spin)

        container_layout.addWidget(nose_card)

        # === BODY TUBE CARD ===
        body_card = VehicleCard("Body Tube", "📦")

        self.body_length_spin = self._create_spinbox(0.2, 5.0, 1.0, "m")
        body_card.add_row("Length:", self.body_length_spin)

        self.body_diameter_spin = self._create_spinbox(0.05, 0.5, 0.1, "m", step=0.01)
        body_card.add_row("Diameter:", self.body_diameter_spin)

        self.body_mass_spin = self._create_spinbox(0.5, 50.0, 1.0, "kg")
        body_card.add_row("Mass:", self.body_mass_spin)

        container_layout.addWidget(body_card)

        # === FINS CARD ===
        fin_card = VehicleCard("Fins", "🔻")

        self.fin_count_spin = QSpinBox()
        self.fin_count_spin.setRange(3, 8)
        self.fin_count_spin.setValue(4)
        self.fin_count_spin.valueChanged.connect(self._update_rocket)
        fin_card.add_row("Count:", self.fin_count_spin)

        self.fin_root_spin = self._create_spinbox(0.05, 0.5, 0.12, "m")
        fin_card.add_row("Root Chord:", self.fin_root_spin)

        self.fin_span_spin = self._create_spinbox(0.02, 0.2, 0.06, "m", step=0.01)
        fin_card.add_row("Span:", self.fin_span_spin)

        self.fin_mass_spin = self._create_spinbox(0.1, 5.0, 0.2, "kg")
        fin_card.add_row("Total Mass:", self.fin_mass_spin)

        container_layout.addWidget(fin_card)

        # === PROPELLANT CARD ===
        prop_card = VehicleCard("Propellant", "🔥")

        self.fuel_mass_spin = self._create_spinbox(1.0, 100.0, 5.0, "kg", step=1.0)
        prop_card.add_row("Fuel Mass:", self.fuel_mass_spin)

        self.ox_mass_spin = self._create_spinbox(1.0, 500.0, 25.0, "kg", step=5.0)
        prop_card.add_row("Oxidizer:", self.ox_mass_spin)

        container_layout.addWidget(prop_card)

        # === RECOVERY CARD ===
        recovery_card = VehicleCard("Recovery", "🪂")

        self.chute_diameter_spin = self._create_spinbox(0.1, 5.0, 1.0, "m")
        self.chute_diameter_spin.valueChanged.disconnect()
        self.chute_diameter_spin.valueChanged.connect(self._update_recovery_estimate)
        recovery_card.add_row("Chute Dia:", self.chute_diameter_spin)

        self.deploy_combo = QComboBox()
        self.deploy_combo.addItems(["At Apogee", "At Altitude"])
        recovery_card.add_row("Deploy:", self.deploy_combo)

        self.descent_rate_label = QLabel("Descent: -- m/s")
        self.descent_rate_label.setStyleSheet("color: #8899aa;")
        recovery_card.layout.addRow(self.descent_rate_label)

        container_layout.addWidget(recovery_card)

        # === LAUNCH CONDITIONS CARD ===
        launch_card = VehicleCard("Launch Conditions", "🌪️")

        self.wind_speed_spin = self._create_spinbox(0.0, 30.0, 0.0, "m/s", step=1.0)
        self.wind_speed_spin.valueChanged.disconnect()
        launch_card.add_row("Wind:", self.wind_speed_spin)

        self.rail_length_spin = self._create_spinbox(0.5, 10.0, 1.5, "m")
        self.rail_length_spin.valueChanged.disconnect()
        launch_card.add_row("Rail Length:", self.rail_length_spin)

        self.angle_spin = self._create_spinbox(45.0, 90.0, 85.0, "°", decimals=1)
        self.angle_spin.valueChanged.disconnect()
        launch_card.add_row("Launch Angle:", self.angle_spin)

        container_layout.addWidget(launch_card)

        container_layout.addStretch()

        scroll.setWidget(container)
        left_main.addWidget(scroll, stretch=1)

        # === Bottom fixed section ===
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("background: #141b22; border-top: 1px solid #2a3a4a;")
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(16, 12, 16, 12)
        bottom_layout.setSpacing(12)

        # Stability indicator
        self.stability_label = QLabel("Stability Margin: -- cal")
        self.stability_label.setStyleSheet("""
            color: #00d4ff;
            font-size: 14pt;
            font-weight: bold;
            font-family: 'Consolas', 'Courier New', monospace;
        """)
        self.stability_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_layout.addWidget(self.stability_label)

        # Launch button
        self.launch_btn = QPushButton("🚀  LAUNCH SIMULATION")
        self.launch_btn.setObjectName("runButton")
        self.launch_btn.setMinimumHeight(52)
        self.launch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.launch_btn.clicked.connect(self.launch_requested.emit)
        bottom_layout.addWidget(self.launch_btn)

        left_main.addWidget(bottom_frame)

        layout.addWidget(left_frame)

        # === Right Panel - Visualization ===
        right_panel = QTabWidget()

        # Rocket diagram tab
        diagram_widget = QWidget()
        diagram_layout = QVBoxLayout(diagram_widget)
        diagram_layout.setContentsMargins(0, 0, 0, 0)

        self.diagram_figure = Figure(figsize=(6, 8), facecolor='#0a0e14')
        self.diagram_canvas = FigureCanvas(self.diagram_figure)
        diagram_layout.addWidget(self.diagram_canvas)

        right_panel.addTab(diagram_widget, "📐 Rocket Diagram")

        # Flight results tab
        flight_widget = QWidget()
        flight_layout = QVBoxLayout(flight_widget)
        flight_layout.setContentsMargins(0, 0, 0, 0)

        self.flight_figure = Figure(figsize=(8, 10), facecolor='#0a0e14')
        self.flight_canvas = FigureCanvas(self.flight_figure)
        flight_layout.addWidget(self.flight_canvas)

        right_panel.addTab(flight_widget, "📊 Flight Results")

        layout.addWidget(right_panel, stretch=1)

        self._init_diagram()

    def _create_default_rocket(self):
        """Create default rocket configuration."""
        from src.core.rocket import create_default_rocket
        self._rocket = create_default_rocket()
        self._update_diagram()

    def _update_rocket(self):
        """Update rocket from UI values."""
        if self._rocket is None:
            return

        from src.core.rocket import NoseShape

        shape_map = {
            0: NoseShape.OGIVE,
            1: NoseShape.CONICAL,
            2: NoseShape.ELLIPTICAL,
            3: NoseShape.PARABOLIC
        }

        # Update nose
        self._rocket.nose.shape = shape_map.get(self.nose_shape_combo.currentIndex(), NoseShape.OGIVE)
        self._rocket.nose.length = self.nose_length_spin.value()
        self._rocket.nose.diameter = self.body_diameter_spin.value()
        self._rocket.nose.mass = self.nose_mass_spin.value()

        # Update body
        self._rocket.body.length = self.body_length_spin.value()
        self._rocket.body.diameter = self.body_diameter_spin.value()
        self._rocket.body.mass = self.body_mass_spin.value()

        # Update fins
        self._rocket.fins.count = self.fin_count_spin.value()
        self._rocket.fins.fin.root_chord = self.fin_root_spin.value()
        self._rocket.fins.fin.span = self.fin_span_spin.value()
        self._rocket.fins.mass = self.fin_mass_spin.value()

        # Update engine
        self._rocket.engine.fuel_mass = self.fuel_mass_spin.value()
        self._rocket.engine.oxidizer_mass = self.ox_mass_spin.value()

        self._rocket._update_positions()
        self._update_diagram()

    def _init_diagram(self):
        """Initialize empty diagram."""
        self.diagram_figure.clear()
        self.ax_diagram = self.diagram_figure.add_subplot(111)
        self.ax_diagram.set_facecolor('#0a0e14')
        self.ax_diagram.text(0.5, 0.5, 'Configure rocket',
                           transform=self.ax_diagram.transAxes,
                           ha='center', va='center', color='#556677',
                           fontsize=14)
        self.diagram_canvas.draw()

    def _update_diagram(self):
        """Update rocket side-view diagram with Mission Control styling."""
        if self._rocket is None:
            return

        from src.core.aero import analyze_rocket

        self._aero_result = analyze_rocket(self._rocket, time=0.0)

        self.diagram_figure.clear()
        ax = self.diagram_figure.add_subplot(111)
        ax.set_facecolor('#0a0e14')

        # Dimensions
        L_nose = self._rocket.nose.length
        L_body = self._rocket.body.length
        D = self._rocket.body.diameter
        R = D / 2
        L_total = L_nose + L_body

        # Draw nose
        nose_x = [0, L_nose, L_nose]
        nose_y_top = [0, R, R]
        nose_y_bot = [0, -R, -R]

        nose_poly = Polygon(
            list(zip(nose_x, nose_y_top, strict=False)) + list(zip(nose_x[::-1], nose_y_bot[::-1], strict=False)),
            closed=True, facecolor='#00d4ff', edgecolor='#ffffff', linewidth=2, alpha=0.8
        )
        ax.add_patch(nose_poly)

        # Draw body
        body_rect = Rectangle(
            (L_nose, -R), L_body, D,
            facecolor='#0066aa', edgecolor='#ffffff', linewidth=2, alpha=0.8
        )
        ax.add_patch(body_rect)

        # Draw fins
        fin_pos = self._rocket.fins.position
        fin_root = self._rocket.fins.fin.root_chord
        fin_span = self._rocket.fins.fin.span

        for sign in [1, -1]:
            fin_poly = Polygon([
                (fin_pos, sign * R),
                (fin_pos + fin_root * 0.3, sign * (R + fin_span)),
                (fin_pos + fin_root, sign * (R + fin_span * 0.5)),
                (fin_pos + fin_root, sign * R)
            ], closed=True, facecolor='#ff6b35', edgecolor='#ffffff', linewidth=1, alpha=0.9)
            ax.add_patch(fin_poly)

        # Draw CG marker
        cg = self._aero_result.cg
        ax.plot(cg, 0, 'o', color='#00ff9d', markersize=16, zorder=10)
        ax.annotate('CG', (cg, R * 1.8), color='#00ff9d', fontsize=11,
                   ha='center', fontweight='bold')

        # Draw CP marker
        cp = self._aero_result.cp_total
        ax.plot(cp, 0, 's', color='#ff3366', markersize=14, zorder=10)
        ax.annotate('CP', (cp, -R * 2.0), color='#ff3366', fontsize=11,
                   ha='center', fontweight='bold')

        # Arrow from CG to CP
        ax.annotate('', xy=(cp, 0), xytext=(cg, 0),
                   arrowprops={"arrowstyle": '->', "color": '#ffffff', "lw": 2})

        # Scale reference
        ax.plot([0, 0.5], [-R - fin_span - 0.08, -R - fin_span - 0.08],
               'w-', linewidth=2)
        ax.text(0.25, -R - fin_span - 0.12, '0.5 m', color='#8899aa',
               ha='center', fontsize=10)

        # Styling
        margin = 0.1
        ax.set_xlim(-margin, L_total + margin)
        ax.set_ylim(-R - fin_span - 0.18, R + fin_span + 0.12)
        ax.set_aspect('equal')
        ax.axis('off')

        # Stability status
        margin_cal = self._aero_result.stability_margin
        if margin_cal >= 1.0:
            status = f"✅ STABLE ({margin_cal:.2f} cal)"
            color = '#00ff9d'
            label_color = '#00ff9d'
        else:
            status = f"⚠️ UNSTABLE ({margin_cal:.2f} cal)"
            color = '#ff3366'
            label_color = '#ff3366'

        ax.set_title(status, color=color, fontsize=16, fontweight='bold', pad=20)

        self.stability_label.setText(f"Margin: {margin_cal:.2f} cal")
        self.stability_label.setStyleSheet(f"""
            color: {label_color};
            font-size: 14pt;
            font-weight: bold;
            font-family: 'Consolas', 'Courier New', monospace;
        """)

        self.diagram_figure.tight_layout()
        self.diagram_canvas.draw()

    def get_rocket(self):
        """Get current rocket configuration."""
        return self._rocket

    def get_launch_conditions(self) -> dict:
        """Get launch condition parameters."""
        return {
            'wind_speed': self.wind_speed_spin.value(),
            'rail_length': self.rail_length_spin.value(),
            'launch_angle': self.angle_spin.value()
        }

    def get_recovery_params(self) -> dict:
        """Get recovery system parameters."""
        return {
            'chute_diameter': self.chute_diameter_spin.value(),
            'deploy_at_apogee': self.deploy_combo.currentIndex() == 0
        }

    def _update_recovery_estimate(self):
        """Update descent rate estimate."""
        if self._rocket is None:
            return

        try:
            from src.core.recovery import Parachute

            chute = Parachute(diameter=self.chute_diameter_spin.value())
            mass = self._rocket.dry_mass
            v_descent, is_safe = chute.is_safe_descent(mass)

            if is_safe:
                self.descent_rate_label.setText(f"Descent: {v_descent:.1f} m/s ✅")
                self.descent_rate_label.setStyleSheet("color: #00ff9d;")
            else:
                self.descent_rate_label.setText(f"Descent: {v_descent:.1f} m/s ⚠️")
                self.descent_rate_label.setStyleSheet("color: #ff6b35;")
        except Exception:
            self.descent_rate_label.setText("Descent: -- m/s")
            self.descent_rate_label.setStyleSheet("color: #8899aa;")

    def update_flight_plots(self, flight_result):
        """Update flight result plots with Mission Control styling."""
        self._flight_result = flight_result

        self.flight_figure.clear()

        # Create grid layout
        gs = self.flight_figure.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        plot_style = {
            'facecolor': '#0a0e14',
            'grid_color': '#1a242e',
            'text_color': '#8899aa',
            'title_color': '#ffffff',
            'line_colors': ['#00d4ff', '#00ff9d', '#ff6b35', '#ff3366']
        }

        def style_axis(ax, title):
            ax.set_facecolor(plot_style['facecolor'])
            ax.tick_params(colors=plot_style['text_color'])
            ax.set_title(title, color=plot_style['title_color'], fontweight='bold', fontsize=11)
            ax.grid(True, color=plot_style['grid_color'], alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        # Altitude
        ax1 = self.flight_figure.add_subplot(gs[0, 0])
        ax1.plot(flight_result.time, flight_result.altitude / 1000,
                color=plot_style['line_colors'][0], linewidth=2)
        ax1.axhline(y=flight_result.apogee_altitude / 1000,
                   color='#ff6b35', linestyle='--', alpha=0.8,
                   label=f'Apogee: {flight_result.apogee_altitude/1000:.1f} km')
        ax1.set_xlabel('Time (s)', color=plot_style['text_color'])
        ax1.set_ylabel('Altitude (km)', color=plot_style['text_color'])
        ax1.legend(facecolor='#141b22', edgecolor='#2a3a4a', labelcolor='#ffffff')
        style_axis(ax1, 'ALTITUDE')

        # Trajectory
        ax2 = self.flight_figure.add_subplot(gs[0, 1])
        ax2.plot(flight_result.range / 1000, flight_result.altitude / 1000,
                color=plot_style['line_colors'][1], linewidth=2)
        ax2.set_xlabel('Range (km)', color=plot_style['text_color'])
        ax2.set_ylabel('Altitude (km)', color=plot_style['text_color'])
        ax2.set_aspect('equal')
        style_axis(ax2, 'TRAJECTORY')

        # Stability
        ax3 = self.flight_figure.add_subplot(gs[1, 0])
        ax3.plot(flight_result.time, flight_result.stability_margin,
                color=plot_style['line_colors'][2], linewidth=2)
        ax3.axhline(y=1.0, color='#00ff9d', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Time (s)', color=plot_style['text_color'])
        ax3.set_ylabel('Margin (cal)', color=plot_style['text_color'])
        style_axis(ax3, 'STABILITY')

        # Dynamic Pressure
        ax4 = self.flight_figure.add_subplot(gs[1, 1])
        ax4.plot(flight_result.time, flight_result.q / 1000,
                color=plot_style['line_colors'][3], linewidth=2)
        ax4.set_xlabel('Time (s)', color=plot_style['text_color'])
        ax4.set_ylabel('Q (kPa)', color=plot_style['text_color'])
        style_axis(ax4, 'MAX Q')

        # Summary
        ax5 = self.flight_figure.add_subplot(gs[2, :])
        ax5.axis('off')

        summary = (
            f"APOGEE: {flight_result.apogee_altitude/1000:.2f} km   │   "
            f"RANGE: {flight_result.range[-1]/1000:.2f} km   │   "
            f"MAX VEL: {flight_result.max_velocity:.0f} m/s   │   "
            f"MAX G: {flight_result.max_acceleration:.1f}"
        )
        ax5.text(0.5, 0.5, summary, transform=ax5.transAxes,
                ha='center', va='center', color='#00ff9d', fontsize=12,
                fontweight='bold', fontfamily='Consolas',
                bbox={"boxstyle": 'round,pad=0.8', "facecolor": '#141b22',
                         "edgecolor": '#00d4ff', "linewidth": 2})

        self.flight_figure.subplots_adjust(
            left=0.1, right=0.95, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3
        )
        self.flight_canvas.draw()

    def update_flight_plots_6dof(self, flight_result):
        """
        Update flight result plots for 6-DOF FlightResult6DOF.

        Includes new plots for propellant mass, Euler angles, and flow status.
        """
        self._flight_result = flight_result

        self.flight_figure.clear()

        # Create grid layout: 3 rows x 2 columns
        gs = self.flight_figure.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        plot_style = {
            'facecolor': '#0a0e14',
            'grid_color': '#1a242e',
            'text_color': '#8899aa',
            'title_color': '#ffffff',
            'line_colors': ['#00d4ff', '#00ff9d', '#ff6b35', '#ff3366']
        }

        def style_axis(ax, title):
            ax.set_facecolor(plot_style['facecolor'])
            ax.tick_params(colors=plot_style['text_color'])
            ax.set_title(title, color=plot_style['title_color'], fontweight='bold', fontsize=11)
            ax.grid(True, color=plot_style['grid_color'], alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        # === Altitude vs Time ===
        ax1 = self.flight_figure.add_subplot(gs[0, 0])
        ax1.plot(flight_result.time, flight_result.position_z,
                color=plot_style['line_colors'][0], linewidth=2)
        ax1.axhline(y=flight_result.apogee_altitude,
                   color='#ff6b35', linestyle='--', alpha=0.8,
                   label=f'Apogee: {flight_result.apogee_altitude:.0f} m')
        ax1.set_xlabel('Time (s)', color=plot_style['text_color'])
        ax1.set_ylabel('Altitude (m)', color=plot_style['text_color'])
        ax1.legend(facecolor='#141b22', edgecolor='#2a3a4a', labelcolor='#ffffff', fontsize=9)
        style_axis(ax1, 'ALTITUDE')

        # === Propellant Mass ===
        ax2 = self.flight_figure.add_subplot(gs[0, 1])
        ax2.plot(flight_result.time, flight_result.propellant_mass,
                color=plot_style['line_colors'][1], linewidth=2)
        ax2.fill_between(flight_result.time, 0, flight_result.propellant_mass,
                        color=plot_style['line_colors'][1], alpha=0.3)
        ax2.set_xlabel('Time (s)', color=plot_style['text_color'])
        ax2.set_ylabel('Propellant (kg)', color=plot_style['text_color'])
        style_axis(ax2, 'PROPELLANT')

        # === Attitude (Euler angles) ===
        ax3 = self.flight_figure.add_subplot(gs[1, 0])
        ax3.plot(flight_result.time, flight_result.pitch,
                color=plot_style['line_colors'][0], linewidth=1.5, label='Pitch')
        ax3.plot(flight_result.time, flight_result.yaw,
                color=plot_style['line_colors'][2], linewidth=1.5, label='Yaw')
        ax3.set_xlabel('Time (s)', color=plot_style['text_color'])
        ax3.set_ylabel('Angle (deg)', color=plot_style['text_color'])
        ax3.legend(facecolor='#141b22', edgecolor='#2a3a4a', labelcolor='#ffffff', fontsize=9)
        style_axis(ax3, 'ATTITUDE')

        # === Velocity ===
        ax4 = self.flight_figure.add_subplot(gs[1, 1])
        velocity = np.sqrt(flight_result.velocity_x**2 +
                          flight_result.velocity_y**2 +
                          flight_result.velocity_z**2)
        ax4.plot(flight_result.time, velocity,
                color=plot_style['line_colors'][3], linewidth=2)
        ax4.axhline(y=flight_result.max_velocity,
                   color='#00d4ff', linestyle='--', alpha=0.6)
        ax4.set_xlabel('Time (s)', color=plot_style['text_color'])
        ax4.set_ylabel('Velocity (m/s)', color=plot_style['text_color'])
        style_axis(ax4, 'VELOCITY')

        # === Summary Box ===
        ax5 = self.flight_figure.add_subplot(gs[2, :])
        ax5.axis('off')

        summary = (
            f"APOGEE: {flight_result.apogee_altitude:.0f} m   │   "
            f"MAX VEL: {flight_result.max_velocity:.0f} m/s (M{flight_result.max_mach:.2f})   │   "
            f"BURNOUT: {flight_result.burnout_altitude:.0f} m   │   "
            f"FLIGHT: {flight_result.flight_time:.1f}s"
        )
        ax5.text(0.5, 0.5, summary, transform=ax5.transAxes,
                ha='center', va='center', color='#00ff9d', fontsize=11,
                fontweight='bold', fontfamily='Consolas',
                bbox={"boxstyle": 'round,pad=0.8', "facecolor": '#141b22',
                         "edgecolor": '#00d4ff', "linewidth": 2})

        self.flight_figure.subplots_adjust(
            left=0.1, right=0.95, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3
        )
        self.flight_canvas.draw()
