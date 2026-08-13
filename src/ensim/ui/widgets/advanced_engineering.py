"""Advanced nozzle-contour and reduced-order uncertainty tools."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ensim.core.engine_uq import EngineUQAnalyzer, EngineUQInput
from ensim.core.moc_solver import export_contour_csv, export_mesh_vtk, generate_mln_contour


def _spinbox(minimum, maximum, value, suffix="", decimals=2, step=None):
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(f" {suffix}")
    if step is not None:
        spin.setSingleStep(step)
    return spin


def _style_axis(axis, title):
    axis.set_facecolor("#0a0e14")
    axis.set_title(title, color="#ffffff", fontweight="bold")
    axis.tick_params(colors="#8899aa")
    axis.grid(True, color="#1a242e", alpha=0.5)
    for spine in axis.spines.values():
        spine.set_color("#2a3a4a")


class _InputCard(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(16, 28, 16, 16)
        self.form.setSpacing(12)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def add(self, label, widget):
        self.form.addRow(label, widget)
        self.setMinimumHeight(50 + self.form.rowCount() * 56)


class MOCDesignTab(QWidget):
    """Minimum-length planar nozzle design using characteristics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._contour = None
        self._mesh = None
        layout = QHBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(430)
        controls = QWidget()
        controls.setMaximumWidth(400)
        controls_layout = QVBoxLayout(controls)
        parameters = _InputCard("Design Parameters")
        self.exit_mach = _spinbox(1.5, 10.0, 3.0, decimals=2, step=0.1)
        self.gamma = _spinbox(1.01, 1.67, 1.2, decimals=3, step=0.01)
        self.throat_radius = _spinbox(0.001, 2.0, 0.05, "m", decimals=4, step=0.005)
        self.characteristics = QSpinBox()
        self.characteristics.setRange(5, 100)
        self.characteristics.setValue(20)
        parameters.add("Exit Mach:", self.exit_mach)
        parameters.add("γ:", self.gamma)
        parameters.add("Throat half-height:", self.throat_radius)
        parameters.add("Characteristic lines:", self.characteristics)
        controls_layout.addWidget(parameters)

        self.generate = QPushButton("GENERATE CONTOUR")
        self.generate.setObjectName("runButton")
        self.generate.clicked.connect(self._generate)
        controls_layout.addWidget(self.generate)

        export = _InputCard("Export")
        self.export_csv = QPushButton("Export contour CSV")
        self.export_vtk = QPushButton("Export characteristic mesh VTK")
        self.export_csv.setEnabled(False)
        self.export_vtk.setEnabled(False)
        self.export_csv.clicked.connect(self._save_csv)
        self.export_vtk.clicked.connect(self._save_vtk)
        export.add("", self.export_csv)
        export.add("", self.export_vtk)
        controls_layout.addWidget(export)

        self.summary = QLabel(
            "The method-of-characteristics solution assumes steady, inviscid, "
            "two-dimensional planar, calorically perfect supersonic flow."
        )
        self.summary.setWordWrap(True)
        self.summary.setObjectName("notesText")
        controls_layout.addWidget(self.summary)
        controls_layout.addStretch()
        scroll.setWidget(controls)
        layout.addWidget(scroll)

        self.figure = Figure(figsize=(8, 6), facecolor="#0a0e14")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

    def _generate(self):
        try:
            self._contour, self._mesh = generate_mln_contour(
                M_exit=self.exit_mach.value(),
                gamma=self.gamma.value(),
                throat_radius=self.throat_radius.value(),
                n_char_lines=self.characteristics.value(),
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            QMessageBox.critical(self, "MOC Design", str(error))
            return
        self.export_csv.setEnabled(True)
        self.export_vtk.setEnabled(True)
        self.summary.setText(
            f"Exit half-height: {self._contour.exit_radius * 1000:.1f} mm\n"
            f"Contour length: {self._contour.length * 1000:.1f} mm\n"
            f"Planar area ratio: {self._contour.exit_radius / self._contour.throat_radius:.2f}\n\n"
            "Assumptions: steady, inviscid, irrotational, two-dimensional planar, "
            "sharp throat and calorically perfect flow."
        )
        self.figure.clear()
        axis = self.figure.add_subplot(1, 1, 1)
        _style_axis(axis, "Planar Minimum-Length Nozzle")
        if self._mesh.x_mesh is not None:
            for row in range(self._mesh.x_mesh.shape[0]):
                valid = np.isfinite(self._mesh.x_mesh[row])
                axis.plot(
                    self._mesh.x_mesh[row][valid],
                    self._mesh.y_mesh[row][valid],
                    color="#2a5968",
                    linewidth=0.6,
                )
                axis.plot(
                    self._mesh.x_mesh[row][valid],
                    -self._mesh.y_mesh[row][valid],
                    color="#2a5968",
                    linewidth=0.6,
                )
            family_count = len(self._mesh.points)
            for diagonal in range(1, family_count + 1):
                points = [
                    self._mesh.points[family][diagonal - family]
                    for family in range(diagonal + 1)
                    if family < family_count and diagonal - family < len(self._mesh.points[family])
                ]
                x_values = [point.x for point in points]
                y_values = [point.y for point in points]
                axis.plot(x_values, y_values, color="#2a5968", linewidth=0.6)
                axis.plot(x_values, [-value for value in y_values], color="#2a5968", linewidth=0.6)
        axis.plot(self._contour.x, self._contour.y, color="#00c8ff", linewidth=2.5)
        axis.plot(self._contour.x, -self._contour.y, color="#00c8ff", linewidth=2.5)
        axis.set_xlabel("Axial position (m)", color="#8899aa")
        axis.set_ylabel("Transverse coordinate (m)", color="#8899aa")
        axis.set_aspect("equal")
        self.figure.tight_layout()
        self.canvas.draw()

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Contour", "contour.csv", "CSV (*.csv)")
        if path:
            export_contour_csv(self._contour, path)

    def _save_vtk(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Mesh", "mesh.vtk", "VTK (*.vtk)")
        if path:
            try:
                export_mesh_vtk(self._mesh, path)
            except ImportError as error:
                QMessageBox.warning(self, "VTK Export", str(error))


class _EngineUQWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, inputs, sample_count, seed, parent=None):
        super().__init__(parent)
        self.inputs = inputs
        self.sample_count = sample_count
        self.seed = seed

    def run(self):
        try:
            result = EngineUQAnalyzer(n_workers=1).run(
                self.inputs,
                n_samples=self.sample_count,
                seed=self.seed,
            )
            self.finished.emit(result)
        except (ValueError, RuntimeError) as error:
            self.failed.emit(str(error))


class EngineUQTab(QWidget):
    """Aleatory uncertainty propagation through the ideal frozen-nozzle model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._worker = None
        layout = QHBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(430)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        nominal = _InputCard("Nominal Ideal-Nozzle Inputs")
        self.pressure = _spinbox(0.1, 100.0, 10.0, "MPa", decimals=2)
        self.area = _spinbox(0.01, 10_000.0, 100.0, "cm²", decimals=2)
        self.expansion = _spinbox(1.1, 500.0, 50.0, decimals=1)
        self.gamma = _spinbox(1.01, 1.67, 1.2, decimals=3)
        self.temperature = _spinbox(300.0, 6000.0, 3500.0, "K", decimals=0)
        self.molecular_weight = _spinbox(1.0, 200.0, 18.0, "g/mol", decimals=2)
        nominal.add("Chamber pressure:", self.pressure)
        nominal.add("Throat area:", self.area)
        nominal.add("Expansion ratio:", self.expansion)
        nominal.add("γ:", self.gamma)
        nominal.add("Chamber temperature:", self.temperature)
        nominal.add("Mean molecular weight:", self.molecular_weight)
        controls_layout.addWidget(nominal)

        uncertainty = _InputCard("Independent One-Sigma Inputs")
        self.pressure_sigma = _spinbox(0.0, 100.0, 2.0, "%", decimals=2)
        self.area_sigma = _spinbox(0.0, 100.0, 1.0, "%", decimals=2)
        self.gamma_sigma = _spinbox(0.0, 25.0, 1.0, "%", decimals=2)
        uncertainty.add("Chamber pressure:", self.pressure_sigma)
        uncertainty.add("Throat area:", self.area_sigma)
        uncertainty.add("γ:", self.gamma_sigma)
        controls_layout.addWidget(uncertainty)

        sampling = _InputCard("Sampling")
        self.samples = QSpinBox()
        self.samples.setRange(100, 100_000)
        self.samples.setValue(1000)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(42)
        sampling.add("Samples:", self.samples)
        sampling.add("Random seed:", self.seed)
        controls_layout.addWidget(sampling)

        self.run_button = QPushButton("RUN UNCERTAINTY PROPAGATION")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._run)
        controls_layout.addWidget(self.run_button)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        controls_layout.addWidget(self.progress)

        self.statistics = QTableWidget(8, 2)
        self.statistics.setHorizontalHeaderLabels(["Statistic", "Value"])
        self.statistics.verticalHeader().setVisible(False)
        controls_layout.addWidget(self.statistics)
        note = QLabel(
            "Scope: ideal frozen, calorically perfect nozzle. Temperature, composition, "
            "model-form error, input correlation, separation, and hardware losses are not "
            "propagated. Reported intervals are conditional on the entered distributions."
        )
        note.setWordWrap(True)
        note.setObjectName("notesText")
        controls_layout.addWidget(note)
        controls_layout.addStretch()
        scroll.setWidget(controls)
        layout.addWidget(scroll)

        self.figure = Figure(figsize=(8, 6), facecolor="#0a0e14")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

    def _run(self):
        inputs = EngineUQInput(
            chamber_pressure=self.pressure.value() * 1e6,
            throat_area=self.area.value() * 1e-4,
            gamma=self.gamma.value(),
            chamber_temperature=self.temperature.value(),
            mean_molecular_weight=self.molecular_weight.value(),
            expansion_ratio=self.expansion.value(),
            chamber_pressure_sigma=self.pressure_sigma.value() / 100.0,
            throat_area_sigma=self.area_sigma.value() / 100.0,
            gamma_sigma=self.gamma_sigma.value() / 100.0,
        )
        self.run_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._worker = _EngineUQWorker(inputs, self.samples.value(), self.seed.value(), self)
        self._worker.finished.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self._worker.start()

    def _finished(self, result):
        self._result = result
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        rows = (
            ("Valid samples", f"{result.n_samples} / {result.n_requested}"),
            ("Runtime", f"{result.runtime_seconds:.2f} s"),
            ("Mean thrust", f"{result.thrust_mean / 1000:.2f} kN"),
            ("Thrust standard deviation", f"{result.thrust_std / 1000:.3f} kN"),
            ("Thrust 95th percentile", f"{result.thrust_p95 / 1000:.2f} kN"),
            ("Mean Isp", f"{result.isp_mean:.2f} s"),
            ("Isp standard deviation", f"{result.isp_std:.3f} s"),
            ("Isp 95th percentile", f"{result.isp_p95:.2f} s"),
        )
        for row, (label, value) in enumerate(rows):
            self.statistics.setItem(row, 0, QTableWidgetItem(label))
            self.statistics.setItem(row, 1, QTableWidgetItem(value))
        self._plot()

    def _failed(self, message):
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Engine UQ", message)

    def _plot(self):
        self.figure.clear()
        thrust_axis = self.figure.add_subplot(1, 2, 1)
        isp_axis = self.figure.add_subplot(1, 2, 2)
        _style_axis(thrust_axis, "Thrust Distribution")
        _style_axis(isp_axis, "Isp Distribution")
        thrust_axis.hist(self._result.thrust_distribution / 1000.0, bins=30, color="#00c8ff")
        isp_axis.hist(self._result.isp_distribution, bins=30, color="#00ff9f")
        thrust_axis.set_xlabel("Thrust (kN)", color="#8899aa")
        isp_axis.set_xlabel("Isp (s)", color="#8899aa")
        for axis in (thrust_axis, isp_axis):
            axis.set_ylabel("Count", color="#8899aa")
        self.figure.tight_layout()
        self.canvas.draw()


class AdvancedEngineeringWidget(QWidget):
    """Container for advanced nozzle geometry and engine UQ analyses."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.moc_tab = MOCDesignTab()
        self.uq_tab = EngineUQTab()
        self.tabs.addTab(self.moc_tab, "Planar Nozzle (MOC)")
        self.tabs.addTab(self.uq_tab, "Engine UQ")
        layout.addWidget(self.tabs)
