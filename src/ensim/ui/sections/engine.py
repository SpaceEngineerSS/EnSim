"""Engine section: operating point in, performance, nozzle and chamber out."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QSplitter, QVBoxLayout, QWidget

from ensim.core.constants import P_ATM
from ensim.core.engine import EngineResult
from ensim.core.mission import atmosphere_properties
from ensim.core.presets import ENGINE_PRESETS
from ensim.core.propellants import FUELS, OXIDIZERS, PROPELLANTS
from ensim.utils.units import Quantity, UnitSystem

from .. import theme
from ..units import UNITS, fmt
from ..widgets.controls import Choice, Form, Metric, NumberField, ResultTable, connect_all, hint
from ..widgets.plot import PlotCanvas
from .base import Section, caption, metric_row, tabs
from .chamber import CoolingView, StabilityView


def _propellant_items(keys) -> list[tuple[str, str, str]]:
    items = []
    for key in keys:
        p = PROPELLANTS[key]
        state = f"liquid at {p.temperature:.1f} K" if p.phase == "liquid" else "vapour at 298 K"
        items.append((key, key, f"{p.name}, {state}"))
    return items


class EngineSection(Section):
    title = "Engine"
    icon_name = "engine"
    run_label = "Run engine"

    def build(self) -> None:
        self._loading = False
        form = Form()

        self.preset = Choice([("Custom", None)] + [(name, name) for name in ENGINE_PRESETS])
        form.add("Preset", self.preset)
        self.preset.currentIndexChanged.connect(self._apply_preset)

        form.heading("Propellants")
        self.fuel = form.add("Fuel", Choice(_propellant_items(FUELS)))
        self.oxidizer = form.add("Oxidizer", Choice(_propellant_items(OXIDIZERS)))
        self.of_ratio = form.add(
            "O/F (mass)",
            NumberField(2.6, 0.2, 30.0, decimals=3, step=0.1),
            "Oxidizer-to-fuel mass ratio",
        )

        form.heading("Chamber")
        self.pc = form.add(
            "Chamber pressure",
            NumberField(70e5, 2e5, 5e7, Quantity.PRESSURE, decimals=2, step=5e5),
        )
        self.throat = form.add(
            "Throat area",
            NumberField(1e-2, 1e-6, 1.0, Quantity.AREA, decimals=3, step=1e-4),
            "Sets thrust and mass flow; Isp does not depend on it",
        )
        self.contraction = form.add(
            "Contraction ratio",
            NumberField(4.0, 1.5, 12.0, decimals=2, step=0.5),
            "Chamber to throat area",
        )
        self.l_star = form.add(
            "L*",
            NumberField(1.0, 0.2, 5.0, Quantity.LENGTH, decimals=2, step=0.05),
            "Characteristic length: chamber volume over throat area",
        )

        form.heading("Nozzle")
        self.eps = form.add(
            "Area ratio", NumberField(25.0, 1.5, 400.0, decimals=1, step=1.0), "Exit to throat area"
        )
        self.pa = form.add(
            "Ambient pressure",
            NumberField(0.0, 0.0, 2e5, Quantity.PRESSURE, decimals=3, step=1e4),
            "Back pressure for the 'operating' column; vacuum and sea level are always shown",
        )
        self.alpha = form.add(
            "Divergence half-angle",
            NumberField(15.0, 0.0, 30.0, decimals=1, step=1.0, suffix="deg"),
            "Conical divergence loss (1 + cos a) / 2 on the momentum thrust",
        )

        form.heading("Efficiency")
        self.eta_cstar = form.add(
            "c* efficiency", NumberField(0.97, 0.7, 1.0, decimals=3, step=0.005)
        )
        self.eta_cf = form.add("Cf efficiency", NumberField(0.98, 0.7, 1.0, decimals=3, step=0.005))

        self._fields = [
            self.fuel, self.oxidizer, self.of_ratio, self.pc, self.throat, self.contraction,
            self.l_star, self.eps, self.pa, self.alpha, self.eta_cstar, self.eta_cf,
        ]  # fmt: skip
        connect_all(self._fields, self._store)

        self.split(form, self._build_views())
        self.workspace.engine_result.connect(self._show_result)
        self.workspace.engine_started.connect(lambda: self.set_running(True))
        self.workspace.engine_failed.connect(lambda _m: self.set_running(False))
        self.workspace.project_loaded.connect(self.load_from_project)

    # views

    def _build_views(self) -> QWidget:
        self.m_isp_vac = Metric("Isp vacuum", unit="s")
        self.m_isp_sl = Metric("Isp sea level", unit="s")
        self.m_thrust = Metric("Thrust vacuum", Quantity.FORCE)
        self.m_cstar = Metric("c*", Quantity.VELOCITY)
        self.m_tc = Metric("Chamber temperature", Quantity.TEMPERATURE)
        self.m_mdot = Metric("Mass flow", Quantity.MASS_FLOW)

        performance = QWidget()
        layout = QVBoxLayout(performance)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            metric_row(
                self.m_isp_vac, self.m_isp_sl, self.m_thrust, self.m_cstar, self.m_tc, self.m_mdot
            )
        )
        split = QSplitter(Qt.Orientation.Horizontal)
        self.profile_plot = PlotCanvas(self._render_profile, self._profile_data)
        split.addWidget(self.profile_plot)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 0, 18, 12)
        right_layout.addWidget(caption("Operating points"))
        self.points = ResultTable(["", "Vacuum", "Sea level", "Operating"])
        right_layout.addWidget(self.points, 1)
        self.regime = hint("")
        right_layout.addWidget(self.regime)
        split.addWidget(right)
        split.setSizes([640, 360])
        layout.addWidget(split, 1)

        self.contour_plot = PlotCanvas(self._render_contour, self._contour_data)
        nozzle = QWidget()
        nozzle_layout = QVBoxLayout(nozzle)
        nozzle_layout.setContentsMargins(0, 0, 0, 0)
        nozzle_layout.addWidget(self.contour_plot, 1)
        bar = QHBoxLayout()
        bar.setContentsMargins(18, 4, 18, 10)
        self.contour_info = caption("")
        export = QPushButton("Export contour…")
        export.clicked.connect(self._export_contour)
        self.view3d_button = QPushButton("3D view")
        self.view3d_button.clicked.connect(self._open_3d)
        bar.addWidget(self.contour_info, 1)
        bar.addWidget(self.view3d_button)
        bar.addWidget(export)
        nozzle_layout.addLayout(bar)

        self.composition_plot = PlotCanvas(self._render_composition, self._composition_data)
        self.altitude_plot = PlotCanvas(self._render_altitude, self._altitude_data)
        self.cooling = CoolingView(self.workspace)
        self.stability = StabilityView(self.workspace)

        self.views = tabs(
            ("Performance", performance),
            ("Nozzle", nozzle),
            ("Composition", self.composition_plot),
            ("Altitude", self.altitude_plot),
            ("Cooling", self.cooling),
            ("Stability", self.stability),
        )
        return self.views

    # inputs <-> project

    def load_from_project(self) -> None:
        if not self.built:
            return
        e = self.workspace.data.engine
        self._loading = True
        try:
            self.fuel.set_value(e.fuel)
            self.oxidizer.set_value(e.oxidizer)
            self.of_ratio.set_value_si(e.of_ratio)
            self.pc.set_value_si(e.chamber_pressure_bar * 1e5)
            self.throat.set_value_si(e.throat_area_cm2 * 1e-4)
            self.contraction.set_value_si(e.contraction_ratio)
            self.l_star.set_value_si(e.characteristic_length)
            self.eps.set_value_si(e.expansion_ratio)
            self.pa.set_value_si(e.ambient_pressure_bar * 1e5)
            self.alpha.set_value_si(e.alpha_deg)
            self.eta_cstar.set_value_si(e.eta_cstar)
            self.eta_cf.set_value_si(e.eta_cf)
        finally:
            self._loading = False
        self._clear_results()

    def _store(self) -> None:
        if self._loading:
            return
        self.workspace.update(
            "engine",
            fuel=self.fuel.value(),
            oxidizer=self.oxidizer.value(),
            of_ratio=self.of_ratio.value_si(),
            chamber_pressure_bar=self.pc.value_si() / 1e5,
            throat_area_cm2=self.throat.value_si() * 1e4,
            contraction_ratio=self.contraction.value_si(),
            characteristic_length=self.l_star.value_si(),
            expansion_ratio=self.eps.value_si(),
            ambient_pressure_bar=self.pa.value_si() / 1e5,
            alpha_deg=self.alpha.value_si(),
            eta_cstar=self.eta_cstar.value_si(),
            eta_cf=self.eta_cf.value_si(),
        )
        if self.preset.currentIndex() != 0:
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(0)
            self.preset.blockSignals(False)

    def _apply_preset(self) -> None:
        name = self.preset.value()
        if name is None:
            return
        p = ENGINE_PRESETS[name]
        self.workspace.update(
            "engine",
            fuel=p.fuel,
            oxidizer=p.oxidizer,
            of_ratio=p.of_ratio,
            chamber_pressure_bar=p.chamber_pressure_bar,
            expansion_ratio=p.expansion_ratio,
            throat_area_cm2=p.throat_area_cm2,
        )
        self.preset.blockSignals(True)
        self.load_from_project()
        self.preset.setCurrentIndex(self.preset.findData(name))
        self.preset.blockSignals(False)

    # run

    def run(self) -> None:
        self.workspace.run_engine()

    def _clear_results(self) -> None:
        for metric in (
            self.m_isp_vac,
            self.m_isp_sl,
            self.m_thrust,
            self.m_cstar,
            self.m_tc,
            self.m_mdot,
        ):
            metric.set_value(None)
        self.points.set_rows([])
        self.regime.setText("")
        self.contour_info.setText("")
        for plot in (
            self.profile_plot,
            self.contour_plot,
            self.composition_plot,
            self.altitude_plot,
        ):
            plot.refresh(has_data=False)

    def _show_result(self, result: EngineResult) -> None:
        self.set_running(False)
        sl = result.sea_level
        self.m_isp_vac.set_value(result.vacuum.isp)
        self.m_isp_sl.set_value(sl.isp if sl else None)
        self.m_thrust.set_value(result.vacuum.thrust)
        self.m_cstar.set_value(result.vacuum.c_star)
        self.m_tc.set_value(result.chamber_temperature)
        self.m_mdot.set_value(result.vacuum.mass_flow_rate)

        columns = [result.vacuum, sl, result.operating]
        ambient = [0.0, P_ATM, result.inputs.ambient_pressure]

        def row(label, getter, quantity=None):
            values = []
            for perf in columns:
                if perf is None:
                    values.append("–")
                    continue
                value = getter(perf)
                values.append(fmt(UNITS.show(value, quantity) if quantity else value))
            unit = f"  [{UNITS.unit(quantity)}]" if quantity else ""
            return [label + unit, *values]

        self.points.set_rows(
            [
                ["Ambient  [" + UNITS.unit(Quantity.PRESSURE) + "]"]
                + [fmt(UNITS.show(p, Quantity.PRESSURE)) for p in ambient],
                row("Isp  [s]", lambda p: p.isp),
                row("Thrust", lambda p: p.thrust, Quantity.FORCE),
                row("Cf", lambda p: p.c_f),
                row("Exit pressure", lambda p: p.exit_pressure, Quantity.PRESSURE),
                row("Exit velocity", lambda p: p.exit_velocity, Quantity.VELOCITY),
                row("Exit Mach", lambda p: p.exit_mach),
            ]
        )
        exit_pressure = result.vacuum.exit_pressure
        if (
            result.inputs.ambient_pressure > 0
            and exit_pressure / result.inputs.ambient_pressure < 0.4
        ):
            self.regime.setObjectName("Warn")
            self.regime.setText(
                "At the operating back pressure the nozzle is strongly over-expanded "
                f"(pe/pa = {exit_pressure / result.inputs.ambient_pressure:.2f} < 0.4). "
                "The flow would separate; the attached-flow thrust shown is optimistic."
            )
        else:
            self.regime.setObjectName("Hint")
            self.regime.setText(
                "Composition frozen at the chamber; Isp includes the c*, Cf and divergence factors."
            )
        self.regime.style().polish(self.regime)
        run = self.workspace.run
        if run is not None:
            c = run.contour
            self.contour_info.setText(
                f"Throat {fmt(UNITS.show(2 * c.throat_radius, Quantity.LENGTH))} "
                f"{UNITS.unit(Quantity.LENGTH)} dia.   Chamber {fmt(UNITS.show(c.chamber_length, Quantity.LENGTH))} "
                f"{UNITS.unit(Quantity.LENGTH)} long   Nozzle {fmt(UNITS.show(c.nozzle_length, Quantity.LENGTH))} "
                f"{UNITS.unit(Quantity.LENGTH)} long (80 % bell)"
            )
        for plot in (
            self.profile_plot,
            self.contour_plot,
            self.composition_plot,
            self.altitude_plot,
        ):
            plot.refresh()
        self.cooling.engine_changed()
        self.stability.engine_changed()

    def restyle(self) -> None:
        if not self.built:
            return
        for plot in (
            self.profile_plot,
            self.contour_plot,
            self.composition_plot,
            self.altitude_plot,
        ):
            plot.restyle()
        self.cooling.restyle()

    # plots

    def _profile_data(self) -> dict[str, np.ndarray]:
        run = self.workspace.run
        return {
            "x_m": run.contour.x,
            "radius_m": run.contour.r,
            "pressure_pa": run.profile["pressure"],
            "temperature_k": run.profile["temperature"],
            "mach": run.profile["mach"],
            "velocity_m_s": run.profile["velocity"],
        }

    def _render_profile(self, figure) -> None:
        run = self.workspace.run
        p = theme.current()
        x = UNITS.show(1.0, Quantity.LENGTH) * run.contour.x
        axes = figure.subplots(3, 1, sharex=True)
        pressure = run.profile["pressure"] / run.result.inputs.chamber_pressure
        axes[0].plot(x, pressure, color=p.signal)
        axes[0].set_ylabel("p / pc")
        axes[1].plot(
            x,
            UNITS.show(1.0, Quantity.TEMPERATURE) * 0
            + np.array([UNITS.show(t, Quantity.TEMPERATURE) for t in run.profile["temperature"]]),
            color=p.series[1],
        )
        axes[1].set_ylabel(f"T  [{UNITS.unit(Quantity.TEMPERATURE)}]")
        axes[2].plot(x, run.profile["mach"], color=p.series[2])
        axes[2].set_ylabel("Mach")
        axes[2].set_xlabel(f"Axial position from throat  [{UNITS.unit(Quantity.LENGTH)}]")
        for ax in axes:
            ax.axvline(0.0, color=p.line_strong, linewidth=0.8)
        axes[0].set_title("Frozen expansion along the wall contour")

    def _contour_data(self) -> dict[str, np.ndarray]:
        run = self.workspace.run
        return {"x_m": run.contour.x, "radius_m": run.contour.r}

    def _render_contour(self, figure) -> None:
        run = self.workspace.run
        p = theme.current()
        scale = UNITS.show(1.0, Quantity.LENGTH)
        x, r = run.contour.x * scale, run.contour.r * scale
        ax = figure.add_subplot(111)
        ax.fill_between(x, r, -r, color=p.raised, linewidth=0)
        ax.plot(x, r, color=p.signal)
        ax.plot(x, -r, color=p.signal)
        ax.axhline(0.0, color=p.line_strong, linewidth=0.6, linestyle=(0, (6, 3, 1, 3)))
        ax.axvline(0.0, color=p.line_strong, linewidth=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"x  [{UNITS.unit(Quantity.LENGTH)}]")
        ax.set_ylabel(f"r  [{UNITS.unit(Quantity.LENGTH)}]")
        ax.set_title("Chamber and nozzle wall")
        ax.grid(False)

    def _composition_data(self) -> dict[str, np.ndarray]:
        comp = self.workspace.result.composition
        return {"species": np.array(list(comp)), "mole_fraction": np.array(list(comp.values()))}

    def _render_composition(self, figure) -> None:
        p = theme.current()
        comp = [(k, v) for k, v in self.workspace.result.composition.items() if v >= 1e-4][:14]
        names = [k for k, _ in comp][::-1]
        values = [v for _, v in comp][::-1]
        ax = figure.add_subplot(111)
        bars = ax.barh(names, values, color=p.series[1], height=0.6)
        bars[-1].set_color(p.signal)
        ax.set_xscale("log")
        ax.set_xlabel("Mole fraction in the chamber")
        ax.grid(axis="y", visible=False)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value * 1.15,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                fontsize=8,
                color=p.muted,
                family=theme.MONO,
            )
        ax.set_title("Equilibrium composition, species above 1e-4")

    def _altitude_table(self):
        result = self.workspace.result
        altitudes = np.linspace(0.0, 80_000.0, 161)
        thrust, isp, ratio = [], [], []
        for h in altitudes:
            pa = atmosphere_properties(float(h))[1]
            perf = result.performance_at(pa)
            thrust.append(perf.thrust if perf else np.nan)
            isp.append(perf.isp if perf else np.nan)
            ratio.append(result.vacuum.exit_pressure / pa)
        return altitudes, np.array(thrust), np.array(isp), np.array(ratio)

    def _altitude_data(self) -> dict[str, np.ndarray]:
        h, thrust, isp, ratio = self._altitude_table()
        return {"altitude_m": h, "thrust_n": thrust, "isp_s": isp, "pe_over_pa": ratio}

    def _render_altitude(self, figure) -> None:
        p = theme.current()
        h, thrust, isp, ratio = self._altitude_table()
        hs = (
            h
            * UNITS.show(1.0, Quantity.LENGTH)
            / (1000.0 if UNITS.system == UnitSystem.SI else 1.0)
        )
        h_unit = "km" if UNITS.system == UnitSystem.SI else UNITS.unit(Quantity.LENGTH)
        ax1, ax2 = figure.subplots(1, 2, sharey=True)
        ax1.plot([UNITS.show(t, Quantity.FORCE) for t in thrust], hs, color=p.signal)
        ax1.set_xlabel(f"Thrust  [{UNITS.unit(Quantity.FORCE)}]")
        ax1.set_ylabel(f"Altitude  [{h_unit}]")
        ax2.plot(isp, hs, color=p.series[1])
        ax2.set_xlabel("Isp  [s]")
        separated = ratio < 0.4
        if separated.any():
            limit = hs[np.argmax(~separated)] if (~separated).any() else hs[-1]
            for ax in (ax1, ax2):
                ax.axhspan(hs[0], limit, color=p.warn, alpha=0.12, linewidth=0)
            ax1.text(
                0.02,
                0.02,
                "pe/pa < 0.4: separation likely",
                transform=ax1.transAxes,
                fontsize=8,
                color=p.warn,
            )
        matched = np.argmin(np.abs(np.log(ratio)))
        for ax in (ax1, ax2):
            ax.axhline(hs[matched], color=p.line_strong, linewidth=0.8, linestyle=(0, (3, 3)))
        ax2.text(
            0.98,
            hs[matched],
            " matched expansion",
            ha="right",
            va="bottom",
            fontsize=8,
            color=p.muted,
            transform=ax2.get_yaxis_transform(),
        )
        ax1.set_title("US Standard Atmosphere 1976")

    # nozzle exports

    def _export_contour(self) -> None:
        run = self.workspace.run
        if run is None:
            return
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export contour", "contour.csv", "CSV (*.csv);;STL mesh (*.stl)"
        )
        if not path:
            return
        if path.lower().endswith(".stl"):
            from ..widgets.view3d import revolve_to_stl

            revolve_to_stl(run.contour.x, run.contour.r, path)
        else:
            np.savetxt(
                path,
                np.column_stack([run.contour.x, run.contour.r, run.contour.area_ratio]),
                delimiter=",",
                header="x_m,radius_m,area_ratio",
                comments="",
            )
        self.workspace.message.emit(f"Contour written to {path}")

    def _open_3d(self) -> None:
        run = self.workspace.run
        if run is None:
            self.workspace.message.emit("Run the engine first")
            return
        from ..widgets.view3d import show_nozzle_3d

        show_nozzle_3d(self, run)
