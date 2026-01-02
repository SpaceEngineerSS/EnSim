"""Main application window."""

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QFont, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ...core.constants import GAS_CONSTANT
from ...core.project_manager import ProjectManager
from ...core.propulsion import get_nozzle_profile
from ...utils.exporter import export_nozzle_profile_csv, export_report
from ..widgets.advanced_engineering import AdvancedEngineeringWidget
from ..widgets.graph_widget import PerformanceGraph
from ..widgets.input_panel import InputPanel
from ..widgets.view3d_widget import NozzleView3D
from ..widgets.staging_widget import MultiStageWidget
from ..widgets.optimization_widget import OptimizationWidget
from ..widgets.cooling_widget import CoolingAnalysisWidget
from ..widgets.propellant_presets_widget import PropellantPresetWidget
from ..widgets.unit_toggle_widget import UnitSystemBar
from ..workers import (
    CalculationWorker,
    MonteCarloParams,
    MonteCarloWorker,
    SimulationParams,
    SimulationResult,
)


class KPICard(QFrame):
    """A card widget displaying a single KPI value."""

    def __init__(self, title: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("kpiCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setObjectName("kpiTitle")
        layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel("---")
        self.value_label.setObjectName("kpiValue")
        layout.addWidget(self.value_label)

        # Unit
        self.unit_label = QLabel(unit)
        self.unit_label.setObjectName("kpiUnit")
        layout.addWidget(self.unit_label, 0, Qt.AlignmentFlag.AlignRight)

    def set_value(self, value: float, decimals: int = 1):
        """Update the displayed value."""
        self.value_label.setText(f"{value:.{decimals}f}")

    def reset(self):
        """Reset to placeholder."""
        self.value_label.setText("---")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EnSim - Rocket Engine Simulation")
        self.setMinimumSize(1200, 800)

        self._worker = None
        self._last_params = None
        self._last_result = None
        self._project = ProjectManager()
        self._first_run = True  # For tutorial overlay

        self._setup_menus()
        self._setup_toolbar()
        self._setup_ui()
        self._load_stylesheet()
        self._setup_shortcuts()
        self._update_title()


    def _setup_ui(self):
        """Build the main window layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # === Top: KPI Dashboard ===
        kpi_frame = QFrame()
        kpi_frame.setObjectName("kpiFrame")
        kpi_layout = QHBoxLayout(kpi_frame)
        kpi_layout.setContentsMargins(0, 0, 0, 0)
        kpi_layout.setSpacing(12)

        self.kpi_isp = KPICard("Vacuum Isp", "seconds")
        self.kpi_thrust = KPICard("Thrust", "kN")
        self.kpi_temp = KPICard("Chamber Temp", "K")
        self.kpi_mach = KPICard("Exit Mach", "")

        kpi_layout.addWidget(self.kpi_isp)
        kpi_layout.addWidget(self.kpi_thrust)
        kpi_layout.addWidget(self.kpi_temp)
        kpi_layout.addWidget(self.kpi_mach)

        main_layout.addWidget(kpi_frame)

        # === Bottom: Splitter with tabs ===
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Input Panel (Scrollable)
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.input_panel = InputPanel()
        scroll_area.setWidget(self.input_panel)
        scroll_area.setMinimumWidth(340)  # Slightly wider for scrollbar

        self.input_panel.run_clicked.connect(self._start_simulation)
        splitter.addWidget(scroll_area)

        # Right: Main Tabs - Organized into logical groups
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)

        # === TAB 1: OUTPUT (Log + Composition) ===
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(0)
        
        output_tabs = QTabWidget()
        output_tabs.setObjectName("subTabs")
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Simulation output will appear here...")
        self.log_output.setObjectName("logOutput")
        output_tabs.addTab(self.log_output, "Log")
        
        self.composition_output = QTextEdit()
        self.composition_output.setReadOnly(True)
        output_tabs.addTab(self.composition_output, "Composition")
        
        output_layout.addWidget(output_tabs)
        self.tabs.addTab(output_tab, "Output")

        # === TAB 2: RESULTS (Graphs + 3D) ===
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(0)
        
        results_tabs = QTabWidget()
        results_tabs.setObjectName("subTabs")
        
        self.graph_widget = PerformanceGraph()
        results_tabs.addTab(self.graph_widget, "Graphs")
        
        self.view3d_widget = NozzleView3D()
        results_tabs.addTab(self.view3d_widget, "3D View")
        
        results_layout.addWidget(results_tabs)
        self.tabs.addTab(results_tab, "Results")

        # === TAB 3: ENGINE (Thermal + Cooling + Propellants) ===
        engine_tab = QWidget()
        engine_layout = QVBoxLayout(engine_tab)
        engine_layout.setContentsMargins(0, 0, 0, 0)
        engine_layout.setSpacing(0)
        
        engine_tabs = QTabWidget()
        engine_tabs.setObjectName("subTabs")
        
        from src.ui.widgets.thermal_widget import ThermalAnalysisWidget
        self.thermal_widget = ThermalAnalysisWidget()
        self.thermal_widget.analysis_requested.connect(self._run_thermal_analysis)
        engine_tabs.addTab(self.thermal_widget, "Thermal")
        
        self.cooling_widget = CoolingAnalysisWidget()
        engine_tabs.addTab(self.cooling_widget, "Cooling")
        
        self.propellant_widget = PropellantPresetWidget()
        self.propellant_widget.preset_selected.connect(self._on_preset_selected)
        engine_tabs.addTab(self.propellant_widget, "Propellants")
        
        self.optimization_widget = OptimizationWidget()
        engine_tabs.addTab(self.optimization_widget, "Optimize")
        
        engine_layout.addWidget(engine_tabs)
        self.tabs.addTab(engine_tab, "Engine")

        # === TAB 4: VEHICLE (Mission + Vehicle + Multi-Stage) ===
        vehicle_tab = QWidget()
        vehicle_layout = QVBoxLayout(vehicle_tab)
        vehicle_layout.setContentsMargins(0, 0, 0, 0)
        vehicle_layout.setSpacing(0)
        
        vehicle_tabs = QTabWidget()
        vehicle_tabs.setObjectName("subTabs")
        
        from src.ui.widgets.mission_widget import MissionAnalysisWidget
        self.mission_widget = MissionAnalysisWidget()
        self.mission_widget.analysis_requested.connect(self._run_mission_analysis)
        vehicle_tabs.addTab(self.mission_widget, "Mission")
        
        from src.ui.widgets.vehicle_widget import VehicleDesignerWidget
        self.vehicle_widget = VehicleDesignerWidget()
        self.vehicle_widget.launch_requested.connect(self._run_flight_simulation)
        vehicle_tabs.addTab(self.vehicle_widget, "Design")
        
        self.staging_widget = MultiStageWidget()
        self.staging_widget.vehicle_changed.connect(self._on_vehicle_changed)
        vehicle_tabs.addTab(self.staging_widget, "Stages")
        
        vehicle_layout.addWidget(vehicle_tabs)
        self.tabs.addTab(vehicle_tab, "Vehicle")

        # === TAB 5: ADVANCED ===
        self.advanced_widget = AdvancedEngineeringWidget()
        self.tabs.addTab(self.advanced_widget, "Advanced")

        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 880])

        main_layout.addWidget(splitter, stretch=1)

        # Timeline Scrubber for simulation replay
        from src.ui.widgets.timeline_scrubber import ReplayControlBar
        self.replay_bar = ReplayControlBar()
        self.replay_bar.setVisible(False)  # Hidden by default, shown during replay
        self.replay_bar.position_changed.connect(self._on_replay_seek)
        self.replay_bar.playback_toggled.connect(self._on_replay_toggle)
        main_layout.addWidget(self.replay_bar)

        # Unit System Bar (above status bar)
        self.unit_bar = UnitSystemBar()
        self.unit_bar.unit_system_changed.connect(self._on_unit_system_changed)
        main_layout.addWidget(self.unit_bar)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # === NEW MODULE CALLBACKS (v2.1) ===
    
    def _on_vehicle_changed(self, vehicle):
        """Handle multi-stage vehicle configuration change."""
        if vehicle:
            self.status_bar.showMessage(
                f"Vehicle updated: {len(vehicle.stages)} stages, "
                f"ΔV = {vehicle.get_total_delta_v():,.0f} m/s",
                3000
            )
    
    def _on_preset_selected(self, preset):
        """Handle propellant preset selection."""
        if preset and hasattr(self, 'input_panel'):
            # Auto-fill O/F ratio in input panel
            if hasattr(self.input_panel, 'of_ratio_spin'):
                self.input_panel.of_ratio_spin.setValue(preset.of_ratio_optimal)
            self.status_bar.showMessage(
                f"Propellant preset applied: {preset.name} (Isp={preset.isp_vacuum:.0f}s)",
                3000
            )
    
    def _on_unit_system_changed(self, system):
        """Handle unit system change."""
        from ...utils.units import UnitSystem
        system_name = "SI (Metric)" if system == UnitSystem.SI else "Imperial (US)"
        self.status_bar.showMessage(f"Unit system changed to {system_name}", 3000)
    
    def _on_replay_seek(self, position: float):
        """Handle replay position change from timeline scrubber."""
        # position is 0.0 - 1.0
        if hasattr(self, '_replay_data') and self._replay_data:
            idx = int(position * (len(self._replay_data) - 1))
            self._display_replay_frame(idx)

    def _on_replay_toggle(self, playing: bool):
        """Handle play/pause toggle from timeline scrubber."""
        self.status_bar.showMessage(f"Replay: {'Playing' if playing else 'Paused'}", 2000)

    def _display_replay_frame(self, frame_idx: int):
        """Display a specific frame from replay data."""
        # Placeholder - implement based on actual replay data format
        pass

    def show_replay_controls(self, duration: float, recording_name: str = ""):
        """Show the replay control bar with a recording."""
        self.replay_bar.set_recording(duration, recording_name)
        self.replay_bar.setVisible(True)
        self.status_bar.showMessage(f"Replay loaded: {recording_name}", 3000)

    def hide_replay_controls(self):
        """Hide the replay control bar."""
        self.replay_bar.setVisible(False)
        self.replay_bar.timeline.stop()

    def _toggle_replay_bar(self, checked: bool):
        """Toggle replay bar visibility from View menu."""
        self.replay_bar.setVisible(checked)
        if checked:
            # Demo mode - show with sample duration
            self.replay_bar.set_recording(60.0, "Demo Replay")
            self.status_bar.showMessage("Replay timeline shown", 2000)
        else:
            self.replay_bar.timeline.stop()
            self.status_bar.showMessage("Replay timeline hidden", 2000)

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        self.action_new = QAction("&New Project", self)
        self.action_new.setShortcut(QKeySequence.StandardKey.New)
        self.action_new.triggered.connect(self._new_project)
        file_menu.addAction(self.action_new)

        self.action_open = QAction("&Open Project...", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._open_project)
        file_menu.addAction(self.action_open)

        self.action_save = QAction("&Save Project", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._save_project)
        file_menu.addAction(self.action_save)

        self.action_save_as = QAction("Save Project &As...", self)
        self.action_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.action_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(self.action_save_as)

        file_menu.addSeparator()

        self.action_export_csv = QAction("Export &CSV...", self)
        self.action_export_csv.triggered.connect(self._export_csv)
        file_menu.addAction(self.action_export_csv)

        self.action_export_report = QAction("Export &Report...", self)
        self.action_export_report.triggered.connect(self._export_report)
        file_menu.addAction(self.action_export_report)

        # Export 3D submenu (CAD Export feature)
        export_3d_menu = file_menu.addMenu("Export 3D Model")

        self.action_export_stl = QAction("Export STL...", self)
        self.action_export_stl.setToolTip("Export nozzle mesh to STL format (for 3D printing)")
        self.action_export_stl.triggered.connect(lambda: self._export_3d('stl'))
        export_3d_menu.addAction(self.action_export_stl)

        self.action_export_ply = QAction("Export PLY...", self)
        self.action_export_ply.setToolTip("Export nozzle mesh to PLY format (with colors)")
        self.action_export_ply.triggered.connect(lambda: self._export_3d('ply'))
        export_3d_menu.addAction(self.action_export_ply)

        self.action_export_obj = QAction("Export OBJ...", self)
        self.action_export_obj.setToolTip("Export nozzle mesh to OBJ format")
        self.action_export_obj.triggered.connect(lambda: self._export_3d('obj'))
        export_3d_menu.addAction(self.action_export_obj)

        file_menu.addSeparator()

        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)

        # Presets menu
        presets_menu = menubar.addMenu("&Presets")
        self._setup_presets_menu(presets_menu)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        self.action_sweep = QAction("&Parametric Sweep...", self)
        self.action_sweep.triggered.connect(self._show_sweep_dialog)
        analysis_menu.addAction(self.action_sweep)

        self.action_compare = QAction("📊 &Compare Designs...", self)
        self.action_compare.triggered.connect(self._show_compare_window)
        analysis_menu.addAction(self.action_compare)

        analysis_menu.addSeparator()

        self.action_propellant_db = QAction("🧪 &Propellant Database...", self)
        self.action_propellant_db.triggered.connect(self._show_propellant_editor)
        analysis_menu.addAction(self.action_propellant_db)

        analysis_menu.addSeparator()

        self.action_monte_carlo = QAction("🎲 &Monte Carlo Dispersion...", self)
        self.action_monte_carlo.setToolTip("Run landing dispersion analysis (N simulations)")
        self.action_monte_carlo.triggered.connect(self._run_monte_carlo_analysis)
        analysis_menu.addAction(self.action_monte_carlo)

        # View menu (Phase 7: Units)
        view_menu = menubar.addMenu("&View")

        self.action_units_si = QAction("Units: SI", self)
        self.action_units_si.setCheckable(True)
        self.action_units_si.setChecked(True)
        self.action_units_si.triggered.connect(lambda: self._set_unit_system('SI'))
        view_menu.addAction(self.action_units_si)

        self.action_units_imperial = QAction("Units: Imperial", self)
        self.action_units_imperial.setCheckable(True)
        self.action_units_imperial.triggered.connect(lambda: self._set_unit_system('Imperial'))
        view_menu.addAction(self.action_units_imperial)

        view_menu.addSeparator()

        # Replay controls
        self.action_show_replay = QAction("🎬 Show Replay Timeline", self)
        self.action_show_replay.setCheckable(True)
        self.action_show_replay.triggered.connect(self._toggle_replay_bar)
        view_menu.addAction(self.action_show_replay)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        self.action_about = QAction("&About EnSim", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)

    def _setup_presets_menu(self, menu):
        """Populate presets menu with engine presets."""
        from ...core.presets import ENGINE_PRESETS

        # Group by manufacturer
        manufacturers = {}
        for name, preset in ENGINE_PRESETS.items():
            if preset.manufacturer not in manufacturers:
                manufacturers[preset.manufacturer] = []
            manufacturers[preset.manufacturer].append((name, preset))

        for manufacturer, presets in sorted(manufacturers.items()):
            submenu = menu.addMenu(manufacturer)
            for name, preset in presets:
                action = QAction(name, self)
                action.setToolTip(preset.description)
                action.triggered.connect(lambda checked, p=preset: self._load_preset(p))
                submenu.addAction(action)

    def _setup_toolbar(self):
        """Setup main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(self.action_new)
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_save)
        toolbar.addSeparator()
        toolbar.addAction(self.action_export_csv)
        toolbar.addAction(self.action_export_report)
        toolbar.addSeparator()

        # Snapshot button (Phase 7)
        self.action_snapshot = QAction("📸 Snapshot", self)
        self.action_snapshot.setToolTip("Save current design for comparison")
        self.action_snapshot.triggered.connect(self._take_snapshot)
        toolbar.addAction(self.action_snapshot)

        toolbar.addAction(self.action_compare)
        toolbar.addSeparator()
        toolbar.addAction(self.action_about)

    def _update_title(self):
        """Update window title with project name."""
        modified = "*" if self._project.is_modified else ""
        self.setWindowTitle(f"EnSim - {self._project.project_name}{modified}")

    def _load_stylesheet(self):
        """Load the QSS stylesheet."""
        base_path = Path(__file__).parent.parent.parent.parent / "assets" / "styles"

        # Priority: aerospace.qss (new) > pro.qss > styles.qss
        for filename in ["aerospace.qss", "pro.qss"]:
            style_path = base_path / filename
            if style_path.exists():
                with open(style_path, encoding='utf-8') as f:
                    self.setStyleSheet(f.read())
                return

        # Fallback to old location
        default_path = Path(__file__).parent.parent.parent.parent / "assets" / "styles.qss"
        if default_path.exists():
            with open(default_path, encoding='utf-8') as f:
                self.setStyleSheet(f.read())

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for power users."""
        from PyQt6.QtGui import QShortcut

        # F5: Quick Run Simulation
        run_shortcut = QShortcut(QKeySequence("F5"), self)
        run_shortcut.activated.connect(self._start_simulation)

        # Ctrl+R: Run Simulation (alternative)
        run_alt_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        run_alt_shortcut.activated.connect(self._start_simulation)

        # Ctrl+E: Export Report
        export_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        export_shortcut.activated.connect(self._export_report)

        # Escape: Stop simulation (if running)
        stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        stop_shortcut.activated.connect(self._stop_simulation)

        # Ctrl+T: Switch to next tab
        tab_shortcut = QShortcut(QKeySequence("Ctrl+Tab"), self)
        tab_shortcut.activated.connect(lambda: self.tabs.setCurrentIndex(
            (self.tabs.currentIndex() + 1) % self.tabs.count()
        ))

        self.status_bar.showMessage("Shortcuts: F5=Run, Ctrl+E=Export, Ctrl+Tab=Switch Tab", 5000)

    def _stop_simulation(self):
        """Stop running simulation."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
            self.input_panel.set_enabled(True)
            self.status_bar.showMessage("Simulation stopped", 3000)
            self._log("⚠️ Simulation stopped by user")

    def _new_project(self):
        """Create a new project."""
        self._project.new_project()
        self._reset_kpis()
        self.log_output.clear()
        self.composition_output.clear()
        self.graph_widget.clear()
        self.view3d_widget.clear()
        self._update_title()
        self.status_bar.showMessage("New project created", 3000)

    def _open_project(self):
        """Open an existing project."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            ProjectManager.FILE_FILTER
        )
        if path:
            data = self._project.load(Path(path))
            if data:
                # Populate UI with loaded data
                self.input_panel.fuel_combo.setCurrentText(data.fuel)
                self.input_panel.oxidizer_combo.setCurrentText(data.oxidizer)
                self.input_panel.of_ratio_spin.setValue(data.of_ratio)
                self.input_panel.pressure_spin.setValue(data.chamber_pressure_bar)
                self.input_panel.throat_area_spin.setValue(data.throat_area_cm2)
                self.input_panel.expansion_spin.setValue(data.expansion_ratio)

                self._update_title()
                self.status_bar.showMessage(f"Opened: {path}", 3000)

    def _save_project(self):
        """Save current project."""
        if not self._project.current_path:
            self._save_project_as()
        else:
            self._collect_inputs()
            if self._project.save():
                self._update_title()
                self.status_bar.showMessage("Project saved", 3000)

    def _save_project_as(self):
        """Save project with new name."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            ProjectManager.FILE_FILTER
        )
        if path:
            self._collect_inputs()
            if self._project.save(Path(path)):
                self._update_title()
                self.status_bar.showMessage(f"Saved: {path}", 3000)

    def _collect_inputs(self):
        """Collect inputs from UI into project."""
        self._project.update_inputs(
            fuel=self.input_panel.fuel_combo.currentText(),
            oxidizer=self.input_panel.oxidizer_combo.currentText(),
            of_ratio=self.input_panel.of_ratio_spin.value(),
            chamber_pressure_bar=self.input_panel.pressure_spin.value(),
            throat_area_cm2=self.input_panel.throat_area_spin.value(),
            expansion_ratio=self.input_panel.expansion_spin.value(),
            ambient=self.input_panel.ambient_combo.currentText()
        )

    def _export_csv(self):
        """Export nozzle profile to CSV."""
        if not self._last_result:
            QMessageBox.warning(self, "Export", "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "nozzle_profile.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            R_specific = GAS_CONSTANT / (self._last_result.mean_mw / 1000.0)
            profile = get_nozzle_profile(
                gamma=self._last_result.gamma,
                T_chamber=self._last_result.temperature,
                P_chamber=self._last_params.chamber_pressure_bar * 1e5,
                exit_area_ratio=self._last_params.expansion_ratio,
                R_specific=R_specific
            )
            if export_nozzle_profile_csv(profile, Path(path)):
                self.status_bar.showMessage(f"Exported: {path}", 3000)

    def _export_report(self):
        """Export simulation report."""
        if not self._last_result:
            QMessageBox.warning(self, "Export", "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "simulation_report.md",
            "Markdown (*.md);;Text (*.txt);;All Files (*)"
        )
        if path:
            if export_report(
                filepath=Path(path),
                fuel=self._last_params.fuel,
                oxidizer=self._last_params.oxidizer,
                of_ratio=self._last_params.oxidizer_moles * 32.0 / (self._last_params.fuel_moles * 2.016),
                chamber_pressure_bar=self._last_params.chamber_pressure_bar,
                expansion_ratio=self._last_params.expansion_ratio,
                temperature=self._last_result.temperature,
                isp_vacuum=self._last_result.isp_vacuum,
                isp_sea_level=self._last_result.isp_sea_level,
                thrust_kN=self._last_result.thrust / 1000,
                c_star=self._last_result.c_star,
                gamma=self._last_result.gamma,
                mean_mw=self._last_result.mean_mw,
                species_fractions=self._last_result.species_fractions,
            ):
                self.status_bar.showMessage(f"Report exported: {path}", 3000)

    def _export_3d(self, format_type: str):
        """
        Export nozzle mesh to 3D CAD format.

        Args:
            format_type: 'stl', 'ply', or 'obj'
        """
        # Check if we have a mesh to export
        if not self.view3d_widget.has_mesh():
            QMessageBox.warning(
                self,
                "Export 3D",
                "No 3D mesh available.\n\nRun a simulation first to generate the nozzle geometry."
            )
            return

        # File dialog
        format_filters = {
            'stl': "STL Files (*.stl);;All Files (*)",
            'ply': "PLY Files (*.ply);;All Files (*)",
            'obj': "OBJ Files (*.obj);;All Files (*)",
        }

        default_names = {
            'stl': "nozzle_export.stl",
            'ply': "nozzle_export.ply",
            'obj': "nozzle_export.obj",
        }

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {format_type.upper()}",
            default_names.get(format_type, "nozzle.stl"),
            format_filters.get(format_type, "All Files (*)")
        )

        if not filepath:
            return

        # Export based on format
        success = False
        if format_type == 'stl':
            success = self.view3d_widget.export_stl(filepath)
        elif format_type == 'ply':
            success = self.view3d_widget.export_ply(filepath)
        elif format_type == 'obj':
            success = self.view3d_widget.export_obj(filepath)

        if success:
            self._log(f"✓ Exported 3D model to: {filepath}")
            self.status_bar.showMessage(f"Exported: {filepath}", 3000)
            QMessageBox.information(
                self,
                "Export Successful",
                f"Nozzle geometry exported to:\n{filepath}"
            )
        else:
            QMessageBox.warning(
                self,
                "Export Failed",
                f"Failed to export {format_type.upper()} file.\nCheck the console for details."
            )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About EnSim",
            """<h2>EnSim v1.0.0</h2>
            <p><b>Rocket Engine Simulation Suite</b></p>
            <p>Scientific engine based on NASA CEA<br>
            (Gordon & McBride Method)</p>
            <hr>
            <p>Features:</p>
            <ul>
            <li>Gibbs equilibrium with dissociation</li>
            <li>1-D isentropic nozzle flow</li>
            <li>Interactive 2D/3D visualization</li>
            </ul>
            <hr>
            <p><small>MIT License © 2024</small></p>"""
        )

    def _load_preset(self, preset):
        """Load an engine preset into the input fields."""

        # Set fuel
        idx = self.input_panel.fuel_combo.findText(preset.fuel)
        if idx >= 0:
            self.input_panel.fuel_combo.setCurrentIndex(idx)

        # Set oxidizer
        idx = self.input_panel.oxidizer_combo.findText(preset.oxidizer)
        if idx >= 0:
            self.input_panel.oxidizer_combo.setCurrentIndex(idx)

        # Set values
        self.input_panel.of_ratio_spin.setValue(preset.of_ratio)
        self.input_panel.pressure_spin.setValue(preset.chamber_pressure_bar)
        self.input_panel.throat_area_spin.setValue(preset.throat_area_cm2)
        self.input_panel.expansion_spin.setValue(preset.expansion_ratio)

        # Log
        self._log(f"Loaded preset: {preset.name}")
        self._log(f"  {preset.description}")
        if preset.reference_isp_vacuum:
            self._log(f"  Reference Isp (vac): {preset.reference_isp_vacuum} s")
        if preset.reference_thrust_kn:
            self._log(f"  Reference Thrust: {preset.reference_thrust_kn} kN")

        self.status_bar.showMessage(f"Loaded: {preset.name}", 3000)

    def _show_sweep_dialog(self):
        """Show parametric sweep analysis dialog."""
        if not self._last_result:
            QMessageBox.warning(self, "Sweep Analysis",
                               "Run a simulation first to establish baseline parameters.")
            return

        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Parametric Sweep Analysis")
        dialog.setMinimumWidth(350)

        layout = QFormLayout(dialog)

        # Parameter selection
        from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox
        param_combo = QComboBox()
        param_combo.addItems(["Chamber Pressure (bar)", "Expansion Ratio"])
        layout.addRow("Parameter:", param_combo)

        # Range
        start_spin = QDoubleSpinBox()
        start_spin.setRange(1, 500)
        start_spin.setValue(30.0)
        layout.addRow("Start:", start_spin)

        end_spin = QDoubleSpinBox()
        end_spin.setRange(1, 500)
        end_spin.setValue(200.0)
        layout.addRow("End:", end_spin)

        steps_spin = QSpinBox()
        steps_spin.setRange(5, 100)
        steps_spin.setValue(20)
        layout.addRow("Steps:", steps_spin)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._run_sweep_analysis(
                param_combo.currentText(),
                start_spin.value(),
                end_spin.value(),
                steps_spin.value()
            )

    def _run_sweep_analysis(self, param_text: str, start: float, end: float, steps: int):
        """Run the parametric sweep and show results."""

        from ...core.analysis import SweepConfig, run_sweep

        # Determine parameter key
        param_key = "chamber_pressure" if "Pressure" in param_text else "expansion_ratio"

        # Create config
        config = SweepConfig(
            parameter=param_key,
            start=start,
            end=end,
            steps=steps,
            fuel=self.input_panel.fuel_combo.currentText(),
            oxidizer=self.input_panel.oxidizer_combo.currentText(),
            base_of_ratio=self.input_panel.of_ratio_spin.value(),
            base_chamber_pressure_bar=self.input_panel.pressure_spin.value(),
            base_expansion_ratio=self.input_panel.expansion_spin.value(),
            base_throat_area_cm2=self.input_panel.throat_area_spin.value()
        )

        # Run sweep
        self._log(f"\nRunning sweep: {param_text} from {start} to {end} ({steps} steps)...")
        result = run_sweep(
            config,
            gamma=self._last_result.gamma,
            mean_mw=self._last_result.mean_mw,
            temperature=self._last_result.temperature
        )

        # Update graphs with sweep results
        self._show_sweep_results(result, param_text)
        self._log("Sweep complete!")

    def _show_sweep_results(self, result, param_label: str):
        """Display sweep results on a new graph."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Sweep Analysis Results")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(8, 6), facecolor='#1e1e1e')
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor('#252525')

        # Plot Isp vs parameter
        ax.plot(result.parameter_values, result.isp_vacuum,
                color='#00a8ff', linewidth=2, label='Isp (vacuum)')
        ax.plot(result.parameter_values, result.isp_sea_level,
                color='#00ff88', linewidth=2, linestyle='--', label='Isp (sea level)')

        ax.set_xlabel(param_label, color='#cccccc')
        ax.set_ylabel('Specific Impulse (s)', color='#cccccc')
        ax.set_title('Parametric Sweep Results', color='white', fontweight='bold')
        ax.tick_params(colors='#888888')
        ax.grid(True, color='#333333', alpha=0.7)
        ax.legend(facecolor='#2d2d2d', edgecolor='#444444', labelcolor='#cccccc')

        for spine in ax.spines.values():
            spine.set_color('#444444')

        fig.tight_layout()
        canvas.draw()

        dialog.exec()

    def _start_simulation(self):
        """Start a new simulation run."""
        # HOTFIX: Prevent rapid-fire thread spawning
        if self._worker is not None and self._worker.isRunning():
            self.status_bar.showMessage("Simulation already running!", 2000)
            return

        # Disable inputs
        self.input_panel.set_enabled(False)
        self.log_output.clear()
        self._reset_kpis()

        # Get O/F ratio and convert to moles
        of_ratio = self.input_panel.of_ratio_spin.value()
        fuel = self.input_panel.fuel_combo.currentText()
        oxidizer = self.input_panel.oxidizer_combo.currentText()

        # Calculate moles from O/F ratio (assuming 1 mol fuel base)
        mw = {"H2": 2.016, "CH4": 16.04, "O2": 32.0, "N2O": 44.01}
        fuel_moles = 2.0  # Base
        ox_moles = of_ratio * fuel_moles * mw.get(fuel, 16.0) / mw.get(oxidizer, 32.0)

        # Build params
        params = SimulationParams(
            fuel=fuel,
            oxidizer=oxidizer,
            fuel_moles=fuel_moles,
            oxidizer_moles=ox_moles,
            chamber_pressure_bar=self.input_panel.pressure_spin.value(),
            expansion_ratio=self.input_panel.expansion_spin.value(),
            ambient_pressure_bar=self.input_panel.get_ambient_pressure_bar(),
            throat_area_cm2=self.input_panel.throat_area_spin.value(),
            eta_cstar=self.input_panel.eta_cstar_spin.value(),
            eta_cf=self.input_panel.eta_cf_spin.value(),
            alpha_deg=self.input_panel.alpha_spin.value()
        )

        # Store params for visualization
        self._last_params = params

        # Log start
        self._log(f"Starting simulation: {fuel}/{oxidizer}")
        self._log(f"  O/F = {of_ratio:.2f}, Pc = {params.chamber_pressure_bar:.1f} bar")

        # Create and start worker
        self._worker = CalculationWorker(params)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_simulation_complete)
        self._worker.error.connect(self._on_simulation_error)
        self._worker.start()

        self.status_bar.showMessage("Simulation running...")

    def _log(self, message: str):
        """Append message to log output."""
        self.log_output.append(message)

    def _reset_kpis(self):
        """Reset all KPI cards."""
        self.kpi_isp.reset()
        self.kpi_thrust.reset()
        self.kpi_temp.reset()
        self.kpi_mach.reset()

    def _on_simulation_complete(self, result: SimulationResult):
        """Handle successful simulation completion."""
        # Store result for export
        self._last_result = result

        # Update KPIs
        self.kpi_isp.set_value(result.isp_vacuum, 1)
        self.kpi_thrust.set_value(result.thrust / 1000, 2)  # N to kN
        self.kpi_temp.set_value(result.temperature, 0)
        self.kpi_mach.set_value(result.exit_mach, 2)

        # Log final results
        self._log("\n" + "=" * 40)
        self._log("RESULTS:")
        self._log(f"  Vacuum Isp:    {result.isp_vacuum:.1f} s")
        self._log(f"  Sea Level Isp: {result.isp_sea_level:.1f} s")
        self._log(f"  C*:            {result.c_star:.1f} m/s")
        self._log(f"  Exit Velocity: {result.exit_velocity:.1f} m/s")
        self._log(f"  Thrust:        {result.thrust/1000:.2f} kN")
        if result.mass_flow_rate:
            self._log(f"  Mass Flow:     {result.mass_flow_rate:.3f} kg/s")
        self._log(f"  Converged:     {result.converged}")

        # Update composition tab
        comp_text = "Product Mole Fractions:\n\n"
        for species, frac in sorted(result.species_fractions.items(),
                                     key=lambda x: x[1], reverse=True):
            comp_text += f"  {species:8s}: {frac*100:6.2f}%\n"
        comp_text += f"\nMean MW: {result.mean_mw:.2f} g/mol"
        comp_text += f"\nGamma:   {result.gamma:.4f}"
        self.composition_output.setText(comp_text)

        # Update visualization widgets
        self._update_visualizations(result)

        # Re-enable inputs
        self.input_panel.set_enabled(True)
        self.status_bar.showMessage("Simulation complete", 5000)

    def _update_visualizations(self, result: SimulationResult):
        """Update graphs and 3D view with simulation results."""
        if not self._last_params:
            return

        P_chamber = self._last_params.chamber_pressure_bar * 1e5

        # Update 2D graphs
        self.graph_widget.update_plots(
            gamma=result.gamma,
            T_chamber=result.temperature,
            P_chamber=P_chamber,
            exit_area_ratio=self._last_params.expansion_ratio,
            mean_mw=result.mean_mw
        )

        # Update 3D view
        self.view3d_widget.update_view(
            exit_area_ratio=self._last_params.expansion_ratio,
            gamma=result.gamma,
            T_chamber=result.temperature,
            P_chamber=P_chamber,
            mean_mw=result.mean_mw,
            color_by='temperature'
        )

        # Switch to graphs tab
        self.tabs.setCurrentIndex(2)

    def _on_simulation_error(self, error_msg: str):
        """Handle simulation error."""
        self._log(f"\n❌ ERROR: {error_msg}")
        self.input_panel.set_enabled(True)
        self.status_bar.showMessage("Simulation failed", 5000)

    def closeEvent(self, event):
        """
        HOTFIX: Graceful exit - stop worker thread before closing.
        Prevents 'RuntimeError: wrapped C/C++ object deleted' crash.
        """
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)  # Wait max 3 seconds
        event.accept()

    def showEvent(self, event):
        """Show tutorial on first run."""
        super().showEvent(event)
        if self._first_run:
            self._first_run = False
            # Delayed show to ensure window is fully rendered
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, self._show_tutorial_if_needed)

    def _show_tutorial_if_needed(self):
        """Show tutorial overlay for first-time users."""
        from src.ui.tutorial_overlay import show_tutorial_if_first_run
        show_tutorial_if_first_run(self)


    def _run_thermal_analysis(self):
        """Run thermal analysis using Bartz model."""
        if self._last_result is None:
            self._log("❌ Run a simulation first before thermal analysis!")
            return

        try:
            from src.core.cooling import calculate_thermal_profile
            from src.core.materials import get_material

            # Get material properties
            mat_name = self.thermal_widget.get_material_name()
            mat = get_material(mat_name)

            # Get cooling parameters
            cool_params = self.thermal_widget.get_cooling_params()

            # Get throat diameter from throat area
            throat_area_m2 = self.input_panel.throat_area_spin.value() * 1e-4
            throat_diameter = 2 * np.sqrt(throat_area_m2 / np.pi)

            # Calculate thermal profile
            result = calculate_thermal_profile(
                T_chamber=self._last_result.temperature,
                P_chamber=self._last_params.chamber_pressure_bar * 1e5,
                c_star=self._last_result.c_star,
                gamma=self._last_result.gamma,
                throat_diameter=throat_diameter,
                expansion_ratio=self._last_params.expansion_ratio,
                wall_thickness=cool_params['wall_thickness'],
                wall_conductivity=mat.thermal_conductivity,
                coolant_temp=cool_params['coolant_temp'],
                coolant_htc=cool_params['coolant_htc'],
                material_limit=mat.max_service_temp
            )

            self.thermal_widget.update_plots(result)
            self._log(f"✅ Thermal analysis complete: Max T = {result.max_wall_temp:.0f}K")

        except Exception as e:
            self._log(f"❌ Thermal analysis error: {e}")

    def _run_mission_analysis(self):
        """Run mission/trajectory performance analysis."""
        if self._last_result is None:
            self._log("❌ Run a simulation first before mission analysis!")
            return

        try:
            from src.core.mission import simulate_ascent

            # Get mission parameters
            mission_params = self.mission_widget.get_mission_params()

            # Get throat area
            throat_area_m2 = self.input_panel.throat_area_spin.value() * 1e-4

            # Run ascent simulation
            result = simulate_ascent(
                T_chamber=self._last_result.temperature,
                P_chamber=self._last_params.chamber_pressure_bar * 1e5,
                gamma=self._last_result.gamma,
                mean_mw=self._last_result.mean_mw,
                expansion_ratio=self._last_params.expansion_ratio,
                throat_area=throat_area_m2,
                eta_cstar=self._last_params.eta_cstar,
                eta_cf=self._last_params.eta_cf,
                alpha_deg=self._last_params.alpha_deg,
                max_altitude=mission_params['max_altitude'],
                step_size=mission_params['step_size']
            )

            self.mission_widget.update_plots(result)
            self._log(f"✅ Mission analysis complete: Optimal altitude = {result.optimal_altitude/1000:.1f} km")

        except Exception as e:
            self._log(f"❌ Mission analysis error: {e}")

    def _run_flight_simulation(self):
        """Run 6-DOF flight simulation using engine results."""
        if self._last_result is None:
            self._log("❌ Run engine simulation first before flight!")
            QMessageBox.warning(
                self,
                "⚠️ Engine Simulation Required",
                "Please run an engine simulation first!\n\n"
                "1. Go to the main tab\n"
                "2. Configure engine parameters\n"
                "3. Click 'RUN SIMULATION'\n"
                "4. Then return here to launch flight simulation"
            )
            return

        try:
            from src.core.flight_6dof import simulate_flight_6dof

            # Get rocket from vehicle widget
            rocket = self.vehicle_widget.get_rocket()
            if rocket is None:
                self._log("❌ Configure rocket first!")
                return

            # Get engine parameters from last simulation
            throat_area_m2 = self.input_panel.throat_area_spin.value() * 1e-4

            # Calculate exit area from expansion ratio
            exit_area = throat_area_m2 * self._last_params.expansion_ratio

            # Get thrust and Isp from simulation results
            thrust_vac = self._last_result.thrust
            isp_vac = self._last_result.isp_vacuum

            # Mass flow rate from simulation or calculate
            g0 = 9.80665
            if self._last_result.mass_flow_rate is not None:
                mdot = self._last_result.mass_flow_rate
            else:
                mdot = thrust_vac / (isp_vac * g0) if isp_vac > 0 else 1.0

            prop_mass = rocket.engine.propellant_mass
            burn_time = prop_mass / mdot if mdot > 0 else 10.0

            # Update rocket engine with simulation results
            rocket.engine.thrust_vac = thrust_vac
            rocket.engine.isp_vac = isp_vac
            rocket.engine.mass_flow_rate = mdot
            rocket.engine.burn_time = burn_time

            self._log("🚀 Launching 6-DOF flight simulation...")
            self._log(f"  Thrust: {thrust_vac/1000:.1f} kN, Burn: {burn_time:.1f}s")

            # Get launch conditions
            launch_cond = self.vehicle_widget.get_launch_conditions()
            wind_speed = launch_cond.get('wind_speed', 0.0)
            rail_length = launch_cond.get('rail_length', 1.5)
            launch_angle = launch_cond.get('launch_angle', 85.0)

            if wind_speed > 0:
                self._log(f"  Wind: {wind_speed} m/s, Rail: {rail_length}m")

            # Run 6-DOF simulation with adaptive integration
            result = simulate_flight_6dof(
                rocket=rocket,
                thrust_vac=thrust_vac,
                isp_vac=isp_vac,
                burn_time=burn_time,
                exit_area=exit_area,
                dt=0.01,
                max_time=300.0,
                launch_angle_deg=launch_angle,
                wind_speed=wind_speed,
                rail_length=rail_length,
                use_adaptive=True,
                output_dt=0.01,  # 100Hz fixed output
                throttle=1.0
            )

            # Store flight result for visualization
            self._last_flight_result = result

            # Update plots with new 6-DOF data
            self.vehicle_widget.update_flight_plots_6dof(result)

            self._log("✅ 6-DOF Flight simulation complete:")
            self._log(f"  Apogee: {result.apogee_altitude:.1f} m ({result.apogee_altitude/1000:.2f} km)")
            self._log(f"  Max Velocity: {result.max_velocity:.0f} m/s (M{result.max_mach:.2f})")
            self._log(f"  Burnout Alt: {result.burnout_altitude:.1f} m")
            self._log(f"  Flight Time: {result.flight_time:.1f}s")

            # Log propellant usage
            if len(result.propellant_mass) > 0:
                prop_initial = result.propellant_mass[0]
                prop_final = result.propellant_mass[-1]
                self._log(f"  Propellant: {prop_initial:.1f} → {prop_final:.1f} kg")

        except Exception as e:
            self._log(f"❌ Flight simulation error: {e}")
            QMessageBox.critical(self, "Flight Error", str(e))
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Phase 7: Professional UX Features
    # =========================================================================

    def _take_snapshot(self):
        """Take a snapshot of current design for comparison."""
        if self._last_result is None:
            self._log("❌ Run a simulation first to take a snapshot!")
            return

        from datetime import datetime

        from .compare_window import DesignSnapshot, get_snapshot_manager

        # Generate name
        fuel = self.input_panel.fuel_combo.currentText()
        ox = self.input_panel.oxidizer_combo.currentText()
        pc = self.input_panel.pressure_spin.value()
        name = f"{fuel}/{ox} @ {pc:.0f}bar"

        snapshot = DesignSnapshot(
            name=name,
            timestamp=datetime.now(),
            fuel=fuel,
            oxidizer=ox,
            of_ratio=self.input_panel.of_ratio_spin.value(),
            chamber_pressure_bar=pc,
            expansion_ratio=self.input_panel.expansion_spin.value(),
            throat_area_cm2=self.input_panel.throat_area_spin.value(),
            isp_vacuum=self._last_result.isp_vacuum,
            isp_sea_level=self._last_result.isp_sea_level,
            thrust_kn=self._last_result.thrust / 1000,
            c_star=self._last_result.c_star,
            exit_velocity=self._last_result.exit_velocity,
            exit_mach=self._last_result.exit_mach,
            gamma=self._last_result.gamma,
            temperature=self._last_result.temperature,
            mean_mw=self._last_result.mean_mw
        )

        manager = get_snapshot_manager()
        manager.add(snapshot)

        self._log(f"📸 Snapshot saved: {name} ({manager.count()} total)")
        self.status_bar.showMessage(f"Snapshot saved: {name}", 3000)

    def _show_compare_window(self):
        """Show design comparison window."""
        from .compare_window import CompareWindow, get_snapshot_manager

        manager = get_snapshot_manager()
        if manager.count() < 1:
            QMessageBox.information(self, "Compare Designs",
                "Take at least one snapshot first using the 📸 button.")
            return

        dialog = CompareWindow(self)
        dialog.exec()

    def _show_propellant_editor(self):
        """Show propellant database editor."""
        from ..widgets.propellant_editor import PropellantEditorDialog

        dialog = PropellantEditorDialog(self)
        dialog.exec()

    def _set_unit_system(self, system: str):
        """Set the global unit system."""
        from ...core.units import UnitSystem, get_units

        units = get_units()

        if system == 'SI':
            units.set_system(UnitSystem.SI)
            self.action_units_si.setChecked(True)
            self.action_units_imperial.setChecked(False)
        else:
            units.set_system(UnitSystem.IMPERIAL)
            self.action_units_si.setChecked(False)
            self.action_units_imperial.setChecked(True)

        self.status_bar.showMessage(f"Units: {system}", 2000)
        self._log(f"🔄 Switched to {system} units")

    # =========================================================================
    # Monte Carlo Dispersion Analysis
    # =========================================================================

    def _run_monte_carlo_analysis(self):
        """Run Monte Carlo landing dispersion analysis."""
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QProgressDialog,
            QSpinBox,
        )

        # Check prerequisites
        if self._last_result is None:
            QMessageBox.warning(
                self,
                "⚠️ Engine Simulation Required",
                "Run an engine simulation first to get thrust/Isp values."
            )
            return

        # Show config dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Monte Carlo Dispersion Analysis")
        dialog.setMinimumWidth(320)

        layout = QFormLayout(dialog)

        num_sims_spin = QSpinBox()
        num_sims_spin.setRange(10, 1000)
        num_sims_spin.setValue(100)
        num_sims_spin.setSingleStep(10)
        layout.addRow("Simulations:", num_sims_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        num_sims = num_sims_spin.value()

        # Create progress dialog
        progress = QProgressDialog(
            f"Running {num_sims} simulations...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Monte Carlo Analysis")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        self._log(f"🎲 Starting Monte Carlo analysis (N={num_sims})...")

        # Create worker params
        params = MonteCarloParams(
            thrust_vac=self._last_result.thrust,
            isp_vac=self._last_result.isp_vacuum,
            burn_time=10.0,  # Default
            fuel_mass=5.0,
            oxidizer_mass=15.0,
            num_simulations=num_sims,
            thrust_sigma=0.02,
            isp_sigma=0.01,
            cd_sigma=0.05,
            seed=42
        )

        # Create and start worker
        self._mc_worker = MonteCarloWorker(params)

        def on_progress(pct):
            progress.setValue(pct)

        def on_finished(result):
            progress.close()
            self._on_monte_carlo_complete(result)

        def on_error(msg):
            progress.close()
            self._log(f"❌ Monte Carlo error: {msg}")
            QMessageBox.critical(self, "Monte Carlo Error", msg)

        self._mc_worker.progress.connect(on_progress)
        self._mc_worker.finished.connect(on_finished)
        self._mc_worker.error.connect(on_error)
        self._mc_worker.log.connect(self._log)
        self._mc_worker.start()

    def _on_monte_carlo_complete(self, result):
        """Handle Monte Carlo completion and display results."""
        self._log(f"✅ Monte Carlo complete: {result.num_simulations} runs")
        self._log(f"   CEP (50%): {result.cep_radius:.1f} m")

        # Format ellipse
        if result.ellipse_major and result.ellipse_minor:
            ellipse_str = f"{result.ellipse_major:.0f}m × {result.ellipse_minor:.0f}m"
        else:
            ellipse_str = "N/A"

        # Show result dialog
        QMessageBox.information(
            self,
            "🎲 Dispersion Analysis Complete",
            f"<h3>Monte Carlo Results (N={result.num_simulations})</h3>"
            f"<table style='font-size: 12pt;'>"
            f"<tr><td><b>CEP (50%):</b></td><td>{result.cep_radius:.1f} m</td></tr>"
            f"<tr><td><b>3σ Ellipse:</b></td><td>{ellipse_str}</td></tr>"
            f"<tr><td><b>Mean Apogee:</b></td><td>{result.mean_apogee:.0f} m</td></tr>"
            f"<tr><td><b>Success Rate:</b></td><td>{result.success_rate*100:.0f}%</td></tr>"
            f"</table>"
        )
