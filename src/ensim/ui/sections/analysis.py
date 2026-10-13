"""Analysis section: sweeps, optimisation, uncertainty, MOC contour, comparison."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ensim.core.analysis import SWEEP_PARAMETERS, sweep_engine
from ensim.core.engine_uq import EngineUQAnalyzer, EngineUQInput
from ensim.core.moc_solver import export_contour_csv, export_mesh_vtk, generate_mln_contour
from ensim.core.optimization import (
    TrajectoryConstraints,
    optimize_engine_parameters,
    optimize_gravity_turn,
    optimize_nozzle_expansion_ratio,
    optimize_stage_mass_allocation,
)
from ensim.utils.units import Quantity

from .. import theme
from ..units import UNITS, fmt
from ..widgets.controls import Choice, Form, Metric, NumberField, ResultTable, hint, primary_button
from ..widgets.plot import PlotCanvas
from ..workspace import Workspace
from .base import Section, caption, metric_row, tabs


def _side(form: QWidget) -> QWidget:
    side = QWidget()
    side.setFixedWidth(300)
    layout = QVBoxLayout(side)
    layout.setContentsMargins(18, 12, 12, 12)
    layout.addWidget(form)
    layout.addStretch()
    return side


class AnalysisSection(Section):
    title = "Analysis"
    icon_name = "analysis"

    def build(self) -> None:
        self.sweep = SweepView(self.workspace)
        self.optimize = OptimizeView(self.workspace)
        self.uncertainty = UncertaintyView(self.workspace)
        self.moc = MocView(self.workspace)
        self.compare = CompareView(self.workspace)
        self._root.addWidget(
            tabs(
                ("Sweep", self.sweep),
                ("Optimise", self.optimize),
                ("Uncertainty", self.uncertainty),
                ("MOC contour", self.moc),
                ("Compare", self.compare),
            )
        )

    def restyle(self) -> None:
        if self.built:
            for view in (self.sweep, self.optimize, self.uncertainty, self.moc, self.compare):
                view.plot.restyle()


class SweepView(QWidget):
    """One engine input varied, everything else from the Engine section."""

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self._result = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = Form()
        self.parameter = form.add(
            "Vary", Choice([(label, key) for key, label in SWEEP_PARAMETERS.items()])
        )
        self.start = form.add("From", NumberField(2.0, 0.01, 1e9, decimals=3, step=0.1))
        self.end = form.add("To", NumberField(4.0, 0.01, 1e9, decimals=3, step=0.1))
        self.steps = QSpinBox()
        self.steps.setRange(3, 200)
        self.steps.setValue(21)
        form.add("Points", self.steps)
        self.unit_hint = hint("")
        form.span(self.unit_hint)
        self.run_button = primary_button("Run sweep", "run")
        self.run_button.clicked.connect(self.run)
        form.span(self.run_button)
        form.span(hint("Chamber equilibrium is solved again at every point."))
        self.parameter.currentIndexChanged.connect(self._defaults)
        layout.addWidget(_side(form))
        self.plot = PlotCanvas(self._render, self._data, "Choose an input and run the sweep.")
        layout.addWidget(self.plot, 1)
        self._defaults()

    def _defaults(self) -> None:
        e = self.workspace.data.engine
        key = self.parameter.value()
        if key == "of_ratio":
            self.start.set_value_si(max(0.3, 0.6 * e.of_ratio))
            self.end.set_value_si(1.5 * e.of_ratio)
            self.unit_hint.setText("Mass ratio of oxidizer to fuel.")
        elif key == "chamber_pressure":
            self.start.set_value_si(max(2.0, 0.3 * e.chamber_pressure_bar))
            self.end.set_value_si(2.0 * e.chamber_pressure_bar)
            self.unit_hint.setText("Values in bar.")
        else:
            self.start.set_value_si(max(2.0, 0.3 * e.expansion_ratio))
            self.end.set_value_si(3.0 * e.expansion_ratio)
            self.unit_hint.setText("Exit to throat area.")

    def run(self) -> None:
        try:
            inputs = self.workspace.engine_inputs()
        except ValueError as error:
            self.workspace.message.emit(str(error))
            return
        key = self.parameter.value()
        values = np.linspace(self.start.value_si(), self.end.value_si(), self.steps.value())
        if key == "chamber_pressure":
            values = values * 1e5
        self.run_button.setEnabled(False)
        self.workspace.tasks.submit(
            "sweep",
            lambda ctx: sweep_engine(
                inputs, key, values, progress=ctx.progress, cancel=ctx.cancelled
            ),
            on_result=self._done,
            on_error=self._failed,
            on_cancel=lambda: self.run_button.setEnabled(True),
        )

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.workspace.message.emit(f"Sweep failed: {message}")

    def _done(self, result) -> None:
        self.run_button.setEnabled(True)
        self._result = result
        self.workspace.results["sweep"] = result
        self.plot.refresh()

    def _x(self):
        r = self._result
        return r.values / 1e5 if r.parameter == "chamber_pressure" else r.values

    def _data(self):
        r = self._result
        return {
            r.parameter: r.values, "chamber_temperature_k": r.chamber_temperature, "c_star_m_s": r.c_star,
            "isp_vacuum_s": r.isp_vacuum, "isp_sea_level_s": r.isp_sea_level, "thrust_vacuum_n": r.thrust_vacuum,
        }  # fmt: skip

    def _render(self, figure) -> None:
        p = theme.current()
        r = self._result
        x = self._x()
        label = SWEEP_PARAMETERS[r.parameter] + (
            "  [bar]" if r.parameter == "chamber_pressure" else ""
        )
        axes = figure.subplots(2, 2, sharex=True).ravel()
        axes[0].plot(x, r.isp_vacuum, color=p.signal, label="vacuum")
        axes[0].plot(x, r.isp_sea_level, color=p.series[1], label="sea level")
        axes[0].set_ylabel("Isp  [s]")
        axes[0].legend()
        axes[1].plot(
            x,
            [UNITS.show(t, Quantity.TEMPERATURE) for t in r.chamber_temperature],
            color=p.series[2],
        )
        axes[1].set_ylabel(f"Tc  [{UNITS.unit(Quantity.TEMPERATURE)}]")
        axes[2].plot(x, r.c_star, color=p.series[1])
        axes[2].set_ylabel("c*  [m/s]")
        axes[3].plot(x, r.molecular_weight, color=p.series[3])
        axes[3].set_ylabel("Molar mass  [g/mol]")
        best = int(np.nanargmax(r.isp_vacuum))
        for ax in axes:
            ax.axvline(x[best], color=p.line_strong, linewidth=0.8, linestyle=(0, (3, 3)))
        for ax in axes[2:]:
            ax.set_xlabel(label)
        axes[0].set_title(f"Peak vacuum Isp {r.isp_vacuum[best]:.1f} s at {fmt(x[best])}")


class OptimizeView(QWidget):
    """Four optimisation problems behind one selector."""

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self._result = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        selector = Form()
        self.problem = selector.add(
            "Problem",
            Choice(
                [
                    ("Nozzle area ratio", "nozzle"),
                    ("Engine design point", "engine"),
                    ("Stage propellant split", "stages"),
                    ("Gravity-turn ascent", "ascent"),
                ]
            ),
        )
        left_layout.addWidget(_side(selector))
        self.forms = QStackedWidget()
        self.forms.addWidget(_side(self._nozzle_form()))
        self.forms.addWidget(_side(self._engine_form()))
        self.forms.addWidget(_side(self._stage_form()))
        self.forms.addWidget(_side(self._ascent_form()))
        left_layout.addWidget(self.forms, 1)
        self.run_button = primary_button("Optimise", "run")
        self.run_button.clicked.connect(self.run)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(18, 0, 12, 14)
        holder_layout.addWidget(self.run_button)
        left_layout.addWidget(holder)
        left.setFixedWidth(300)
        self.problem.currentIndexChanged.connect(self.forms.setCurrentIndex)
        layout.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 12, 18, 12)
        self.message = caption("")
        right_layout.addWidget(self.message)
        self.table = ResultTable(["Quantity", "Value"])
        self.table.setMaximumHeight(300)
        right_layout.addWidget(self.table)
        self.plot = PlotCanvas(self._render, None, "Pick a problem and optimise.")
        right_layout.addWidget(self.plot, 1)
        layout.addWidget(right, 1)

    def _nozzle_form(self) -> QWidget:
        form = Form()
        self.n_weight = form.add(
            "Vacuum weight",
            NumberField(0.7, 0.0, 1.0, decimals=2, step=0.05),
            "Share of the objective given to vacuum Cf",
        )
        self.n_alt = form.add(
            "Lower operating altitude",
            NumberField(0.0, 0.0, 60_000.0, Quantity.LENGTH, decimals=0, step=500.0),
        )
        form.span(
            hint(
                "Maximises a weighted thrust coefficient between vacuum and the lower altitude, using the engine's chamber pressure and gamma."
            )
        )
        return form

    def _engine_form(self) -> QWidget:
        form = Form()
        self.e_pair = form.add(
            "Propellants",
            Choice(
                [
                    (k, k)
                    for k in ("LOX/CH4", "LOX/RP1", "LOX/LH2", "NTO/MMH", "NTO/UDMH", "NTO/N2H4")
                ]
            ),
        )
        self.e_thrust = form.add(
            "Target thrust", NumberField(1e6, 1e2, 1e8, Quantity.FORCE, decimals=1, step=1e4)
        )
        self.e_isp = form.add(
            "Target Isp", NumberField(340.0, 150.0, 480.0, decimals=1, step=1.0, suffix="s")
        )
        self.e_pc_lo = form.add(
            "Pc from", NumberField(50e5, 5e5, 4e7, Quantity.PRESSURE, decimals=1, step=5e5)
        )
        self.e_pc_hi = form.add(
            "Pc to", NumberField(250e5, 5e5, 4e7, Quantity.PRESSURE, decimals=1, step=5e5)
        )
        self.e_of_lo = form.add("O/F from", NumberField(2.4, 0.3, 20.0, decimals=2, step=0.1))
        self.e_of_hi = form.add("O/F to", NumberField(3.8, 0.3, 20.0, decimals=2, step=0.1))
        self.e_eps = form.add("Area ratio", NumberField(40.0, 2.0, 300.0, decimals=1, step=1.0))
        return form

    def _stage_form(self) -> QWidget:
        form = Form()
        self.s_total = form.add(
            "Total propellant",
            NumberField(400_000.0, 100.0, 5e6, Quantity.MASS, decimals=0, step=1000.0),
        )
        self.s_payload = form.add(
            "Payload", NumberField(10_000.0, 1.0, 2e5, Quantity.MASS, decimals=0, step=500.0)
        )
        self.s_isps = form.add("Isp per stage", QLineEdit("311, 348"))
        self.s_structure = form.add("Dry fraction per stage", QLineEdit("0.06, 0.08"))
        form.span(
            hint("Comma-separated, bottom stage first. Dry fraction is dry mass over stage mass.")
        )
        return form

    def _ascent_form(self) -> QWidget:
        form = Form()
        self.a_mass = form.add(
            "Liftoff mass", NumberField(30_000.0, 10.0, 5e6, Quantity.MASS, decimals=0, step=1000.0)
        )
        self.a_propellant = form.add(
            "Propellant", NumberField(24_000.0, 1.0, 5e6, Quantity.MASS, decimals=0, step=1000.0)
        )
        self.a_thrust = form.add(
            "Thrust", NumberField(500e3, 1e2, 1e8, Quantity.FORCE, decimals=1, step=1e4)
        )
        self.a_isp = form.add(
            "Isp", NumberField(300.0, 100.0, 480.0, decimals=1, step=1.0, suffix="s")
        )
        self.a_target = form.add(
            "Target altitude",
            NumberField(150_000.0, 1000.0, 1e6, Quantity.LENGTH, decimals=0, step=5000.0),
        )
        self.a_qmax = form.add(
            "q limit",
            NumberField(35_000.0, 1000.0, 2e5, Quantity.PRESSURE, decimals=0, step=1000.0),
        )
        form.span(
            hint(
                "Planar point-mass ascent; optimises kick-off altitude, kick angle and pitch rate."
            )
        )
        return form

    def run(self) -> None:
        problem = self.problem.value()
        ws = self.workspace
        try:
            if problem == "nozzle":
                if ws.result is None:
                    raise ValueError(
                        "Run the engine first; the chamber pressure and gamma come from it"
                    )
                from ensim.core.mission import atmosphere_properties

                ambient = atmosphere_properties(self.n_alt.value_si())[1]
                args = {
                    "chamber_pressure": ws.result.inputs.chamber_pressure,
                    "ambient_pressure": ambient,
                    "gamma": ws.result.gamma,
                    "weight_vacuum": self.n_weight.value_si(),
                    "weight_sealevel": 1.0 - self.n_weight.value_si(),
                }
                work = lambda ctx: optimize_nozzle_expansion_ratio(**args)  # noqa: E731
            elif problem == "engine":
                args = {
                    "target_thrust": self.e_thrust.value_si(),
                    "target_isp": self.e_isp.value_si(),
                    "propellant_type": self.e_pair.value(),
                    "chamber_pressure_range": (self.e_pc_lo.value_si(), self.e_pc_hi.value_si()),
                    "mixture_ratio_range": (self.e_of_lo.value_si(), self.e_of_hi.value_si()),
                    "expansion_ratio": self.e_eps.value_si(),
                }
                work = lambda ctx: optimize_engine_parameters(**args)  # noqa: E731
            elif problem == "stages":
                isps = [float(v) for v in self.s_isps.text().split(",") if v.strip()]
                structure = [float(v) for v in self.s_structure.text().split(",") if v.strip()]
                if len(isps) != len(structure) or len(isps) < 2:
                    raise ValueError(
                        "Give one Isp and one dry fraction per stage, at least two stages"
                    )
                args = {
                    "total_propellant": self.s_total.value_si(),
                    "num_stages": len(isps),
                    "payload_mass": self.s_payload.value_si(),
                    "stage_isps": isps,
                    "structural_coefficients": structure,
                }
                work = lambda ctx: optimize_stage_mass_allocation(**args)  # noqa: E731
            else:
                constraints = TrajectoryConstraints(
                    target_altitude=self.a_target.value_si(),
                    max_dynamic_pressure=self.a_qmax.value_si(),
                )
                args = {
                    "vehicle_mass": self.a_mass.value_si(),
                    "thrust": self.a_thrust.value_si(),
                    "isp": self.a_isp.value_si(),
                    "propellant_mass": self.a_propellant.value_si(),
                    "constraints": constraints,
                }
                work = lambda ctx: optimize_gravity_turn(**args)  # noqa: E731
        except ValueError as error:
            self.message.setText(str(error))
            return
        self.run_button.setEnabled(False)
        self.message.setText("Optimising…")
        self.workspace.tasks.submit(
            f"optimise-{problem}", work, on_result=self._done, on_error=self._failed
        )

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.message.setText(f"Failed: {message}")

    def _done(self, result) -> None:
        self.run_button.setEnabled(True)
        self._result = result
        self.message.setText(
            ("Converged. " if result.success else "Did not fully converge. ") + str(result.message)
        )
        rows = []
        for key, value in result.optimal_params.items():
            if isinstance(value, (list, tuple)):
                value = ", ".join(fmt(float(v)) for v in value)
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                value = str(value)
            rows.append([key.replace("_", " "), value if isinstance(value, str) else float(value)])
        self.table.set_rows(rows)
        self.plot.refresh(has_data=bool(result.convergence_history))

    def _render(self, figure) -> None:
        p = theme.current()
        history = np.asarray(self._result.convergence_history, dtype=float)
        ax = figure.add_subplot(111)
        ax.plot(np.arange(1, len(history) + 1), history, color=p.signal, marker="o", markersize=2)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Objective")
        ax.set_title("Convergence")


class UncertaintyView(QWidget):
    """Monte Carlo on the ideal-nozzle relations around the current engine."""

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self._result = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = Form()
        self.pc_sigma = form.add(
            "Chamber pressure 1-sigma",
            NumberField(0.02, 0.0, 0.3, decimals=3, step=0.005, suffix="x"),
        )
        self.at_sigma = form.add(
            "Throat area 1-sigma", NumberField(0.01, 0.0, 0.3, decimals=3, step=0.005, suffix="x")
        )
        self.gamma_sigma = form.add(
            "Gamma 1-sigma", NumberField(0.01, 0.0, 0.1, decimals=3, step=0.002, suffix="x")
        )
        self.samples = QSpinBox()
        self.samples.setRange(100, 200_000)
        self.samples.setValue(5000)
        form.add("Samples", self.samples)
        self.threshold = form.add(
            "Minimum thrust",
            NumberField(0.0, 0.0, 1e9, Quantity.FORCE, decimals=1),
            "Reliability = share of samples above this; 0 to skip",
        )
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(7)
        form.add("Seed", self.seed)
        self.run_button = primary_button("Propagate", "run")
        self.run_button.clicked.connect(self.run)
        form.span(self.run_button)
        form.span(
            hint("Varies Pc, At and gamma around the last engine run with Tc and molar mass held.")
        )
        layout.addWidget(_side(form))
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.m_thrust = Metric("Thrust mean", Quantity.FORCE)
        self.m_thrust_sd = Metric("Thrust 1-sigma", Quantity.FORCE)
        self.m_isp = Metric("Isp mean", unit="s")
        self.m_isp_sd = Metric("Isp 1-sigma", unit="s", digits=3)
        self.m_rel = Metric("Reliability", unit="", digits=4)
        right_layout.addWidget(
            metric_row(self.m_thrust, self.m_thrust_sd, self.m_isp, self.m_isp_sd, self.m_rel)
        )
        self.plot = PlotCanvas(self._render, self._data, "Run the engine, then propagate.")
        right_layout.addWidget(self.plot, 1)
        layout.addWidget(right, 1)

    def run(self) -> None:
        r = self.workspace.result
        if r is None:
            self.workspace.message.emit("Run the engine first")
            return
        inputs = EngineUQInput(
            chamber_pressure=r.inputs.chamber_pressure,
            throat_area=r.inputs.throat_area,
            gamma=r.gamma,
            chamber_temperature=r.chamber_temperature,
            mean_molecular_weight=r.molecular_weight,
            expansion_ratio=r.inputs.expansion_ratio,
            chamber_pressure_sigma=self.pc_sigma.value_si(),
            throat_area_sigma=self.at_sigma.value_si(),
            gamma_sigma=self.gamma_sigma.value_si(),
        )
        threshold = self.threshold.value_si() or None
        samples, seed = self.samples.value(), self.seed.value()
        self.run_button.setEnabled(False)
        self.workspace.tasks.submit(
            "uncertainty",
            lambda ctx: EngineUQAnalyzer(n_workers=1).run(
                inputs, samples, r.inputs.ambient_pressure, threshold, seed
            ),
            on_result=self._done,
            on_error=self._failed,
        )

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.workspace.message.emit(f"Uncertainty propagation failed: {message}")

    def _done(self, result) -> None:
        self.run_button.setEnabled(True)
        self._result = result
        self.m_thrust.set_value(result.thrust_mean)
        self.m_thrust_sd.set_value(result.thrust_std)
        self.m_isp.set_value(result.isp_mean)
        self.m_isp_sd.set_value(result.isp_std)
        self.m_rel.set_value(result.reliability)
        self.plot.refresh()

    def _data(self):
        r = self._result
        return {
            "thrust_n": r.thrust_distribution,
            "isp_s": r.isp_distribution,
            "c_star_m_s": r.cstar_distribution,
        }

    def _render(self, figure) -> None:
        p = theme.current()
        r = self._result
        ax1, ax2 = figure.subplots(1, 2)
        ax1.hist(
            [UNITS.show(t, Quantity.FORCE) for t in r.thrust_distribution],
            bins=50,
            color=p.signal,
            alpha=0.85,
        )
        ax1.set_xlabel(f"Thrust  [{UNITS.unit(Quantity.FORCE)}]")
        ax1.set_ylabel("Samples")
        ax2.hist(r.isp_distribution, bins=50, color=p.series[1], alpha=0.85)
        ax2.set_xlabel("Isp  [s]")
        ax1.set_title(f"{r.n_samples} samples")


class MocView(QWidget):
    """Planar minimum-length nozzle by the method of characteristics."""

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self._contour = self._mesh = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = Form()
        self.mach = form.add("Exit Mach", NumberField(3.0, 1.1, 8.0, decimals=3, step=0.1))
        self.gamma = form.add("Gamma", NumberField(1.2, 1.05, 1.67, decimals=4, step=0.01))
        self.lines = QSpinBox()
        self.lines.setRange(4, 120)
        self.lines.setValue(30)
        form.add("Characteristics", self.lines)
        self.from_engine = QPushButton("Use engine exit Mach and gamma")
        self.from_engine.clicked.connect(self._from_engine)
        form.span(self.from_engine)
        self.run_button = primary_button("Generate contour", "run")
        self.run_button.clicked.connect(self.run)
        form.span(self.run_button)
        export_csv = QPushButton("Export contour CSV…")
        export_vtk = QPushButton("Export mesh VTK…")
        export_csv.clicked.connect(self._export_csv)
        export_vtk.clicked.connect(self._export_vtk)
        form.span(export_csv)
        form.span(export_vtk)
        form.span(
            hint("Planar, inviscid, calorically perfect, sharp throat. Not an axisymmetric bell.")
        )
        layout.addWidget(_side(form))
        self.plot = PlotCanvas(self._render, self._data, "Generate a contour.")
        layout.addWidget(self.plot, 1)

    def _from_engine(self) -> None:
        r = self.workspace.result
        if r is None:
            self.workspace.message.emit("Run the engine first")
            return
        self.mach.set_value_si(min(8.0, r.vacuum.exit_mach))
        self.gamma.set_value_si(r.gamma)

    def run(self) -> None:
        mach, gamma, lines = self.mach.value_si(), self.gamma.value_si(), self.lines.value()
        self.run_button.setEnabled(False)
        self.workspace.tasks.submit(
            "moc",
            lambda ctx: generate_mln_contour(
                M_exit=mach, gamma=gamma, throat_radius=1.0, n_char_lines=lines
            ),
            on_result=self._done,
            on_error=self._failed,
        )

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.workspace.message.emit(f"MOC failed: {message}")

    def _done(self, outcome) -> None:
        self.run_button.setEnabled(True)
        self._contour, self._mesh = outcome
        self.plot.refresh()

    def _data(self):
        c = self._contour
        return {"x_over_rt": c.x, "y_over_rt": c.y, "mach": c.M, "theta_rad": c.theta}

    def _render(self, figure) -> None:
        from matplotlib.collections import LineCollection

        p = theme.current()
        c, mesh = self._contour, self._mesh
        ax = figure.add_subplot(111)
        segments = []
        families = mesh.points
        for diagonal in range(1, len(families) + 1):
            pts = [
                families[f][diagonal - f]
                for f in range(diagonal + 1)
                if f < len(families) and diagonal - f < len(families[f])
            ]
            if len(pts) > 1:
                segments.append([(pt.x, pt.y) for pt in pts])
        for family in families:
            if len(family) > 1:
                segments.append([(pt.x, pt.y) for pt in family])
        mirrored = [[(x, -y) for x, y in seg] for seg in segments]
        ax.add_collection(LineCollection(segments + mirrored, colors=p.line_strong, linewidths=0.5))
        ax.plot(c.x, c.y, color=p.signal, linewidth=1.6)
        ax.plot(c.x, -c.y, color=p.signal, linewidth=1.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x / throat half-height")
        ax.set_ylabel("y / throat half-height")
        ax.set_title(
            f"Exit Mach {c.M_exit:.2f}, length {c.length:.2f}, area ratio {c.exit_radius / c.throat_radius:.2f}"
        )

    def _export_csv(self) -> None:
        if self._contour is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export contour", "moc_contour.csv", "CSV (*.csv)"
        )
        if path:
            export_contour_csv(self._contour, path)

    def _export_vtk(self) -> None:
        if self._mesh is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export mesh", "moc_mesh.vtk", "VTK (*.vtk)")
        if path:
            export_mesh_vtk(self._mesh, path)


class CompareView(QWidget):
    """Pinned engine results side by side."""

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self._pins: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        bar = QHBoxLayout()
        self.label = QLineEdit()
        self.label.setPlaceholderText("Label for the current engine result")
        pin = QPushButton("Pin current result")
        pin.clicked.connect(self.pin)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        bar.addWidget(self.label, 1)
        bar.addWidget(pin)
        bar.addWidget(clear)
        layout.addLayout(bar)
        self.table = ResultTable(
            [
                "Label",
                "Propellants",
                "O/F",
                "Pc  [bar]",
                "Area ratio",
                "Tc  [K]",
                "c*  [m/s]",
                "Isp vac  [s]",
                "Isp SL  [s]",
            ]
        )
        self.table.setMaximumHeight(220)
        layout.addWidget(self.table)
        self.plot = PlotCanvas(
            self._render, None, "Pin two or more engine results to compare them."
        )
        layout.addWidget(self.plot, 1)

    def pin(self) -> None:
        try:
            snap = self.workspace.snapshot()
        except ValueError as error:
            self.workspace.message.emit(str(error))
            return
        snap["label"] = self.label.text().strip() or f"Result {len(self._pins) + 1}"
        self._pins.append(snap)
        self.label.clear()
        self._refresh()

    def clear(self) -> None:
        self._pins.clear()
        self._refresh()

    def _refresh(self) -> None:
        rows = []
        for s in self._pins:
            i = s["inputs"]
            rows.append([
                s["label"], f"{i['fuel']}/{i['oxidizer']}", float(i["of_ratio"]), i["chamber_pressure"] / 1e5,
                float(i["expansion_ratio"]), s["chamber_temperature"], s["c_star"], s["isp_vacuum"],
                s["isp_sea_level"] if s["isp_sea_level"] is not None else "–",
            ])  # fmt: skip
        self.table.set_rows(rows)
        self.plot.refresh(has_data=len(self._pins) > 1)

    def _render(self, figure) -> None:
        p = theme.current()
        labels = [s["label"] for s in self._pins]
        vac = [s["isp_vacuum"] for s in self._pins]
        sl = [s["isp_sea_level"] or 0.0 for s in self._pins]
        ax = figure.add_subplot(111)
        index = np.arange(len(labels))
        ax.bar(index - 0.18, vac, width=0.36, color=p.signal, label="vacuum")
        ax.bar(index + 0.18, sl, width=0.36, color=p.series[1], label="sea level")
        ax.set_xticks(index, labels)
        ax.set_ylabel("Isp  [s]")
        ax.set_ylim(min(v for v in sl + vac if v > 0) * 0.85, max(vac) * 1.05)
        ax.legend()
        ax.grid(axis="x", visible=False)
