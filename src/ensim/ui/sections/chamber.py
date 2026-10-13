"""Chamber views inside the Engine section: regenerative cooling and acoustics."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QSpinBox, QVBoxLayout, QWidget

from ensim.core.cooling import (
    COOLANT_DATABASE,
    COOLING_MATERIALS,
    CoolantType,
    CoolingChannel,
    CoolingSystemDesign,
    analyze_cooling_system,
)
from ensim.core.instability import analyze_combustion_instability
from ensim.utils.units import Quantity

from .. import theme
from ..units import UNITS, fmt
from ..widgets.controls import Choice, Form, Metric, NumberField, ResultTable, hint, primary_button
from ..widgets.plot import PlotCanvas
from ..workspace import Workspace
from .base import caption, metric_row

FUEL_COOLANT = {
    "RP-1": CoolantType.RP1,
    "LH2": CoolantType.LH2,
    "LCH4": CoolantType.LCH4,
    "N2H4": CoolantType.N2H4,
    "MMH": CoolantType.MMH,
}


class CoolingView(QWidget):
    """Regenerative cooling of the current chamber contour."""

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self._results = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        form = Form()
        self.coolant = form.add(
            "Coolant",
            Choice([(t.name, t) for t in COOLANT_DATABASE]),
            "Defaults to the fuel when it is in the coolant table",
        )
        self.flow_fraction = form.add(
            "Coolant flow",
            NumberField(1.0, 0.05, 1.0, decimals=2, step=0.05, suffix="x fuel"),
            "Fraction of the fuel mass flow routed through the jacket",
        )
        self.inlet_t = form.add(
            "Inlet temperature", NumberField(293.15, 15.0, 600.0, Quantity.TEMPERATURE, decimals=1)
        )
        self.inlet_p_ratio = form.add(
            "Inlet pressure", NumberField(1.5, 1.05, 4.0, decimals=2, step=0.05, suffix="x pc")
        )
        self.counterflow = form.span(QCheckBox("Coolant enters at the nozzle exit"))
        self.counterflow.setChecked(True)
        form.heading("Channels")
        self.channels = QSpinBox()
        self.channels.setRange(8, 600)
        self.channels.setValue(120)
        form.add("Number", self.channels)
        self.width = form.add(
            "Width", NumberField(1.5e-3, 2e-4, 2e-2, Quantity.LENGTH, decimals=5, units=_mm())
        )
        self.height = form.add(
            "Height", NumberField(3e-3, 2e-4, 3e-2, Quantity.LENGTH, decimals=5, units=_mm())
        )
        self.wall = form.add(
            "Hot-wall thickness",
            NumberField(1e-3, 2e-4, 1e-2, Quantity.LENGTH, decimals=5, units=_mm()),
        )
        self.land = form.add(
            "Land width", NumberField(1e-3, 2e-4, 1e-2, Quantity.LENGTH, decimals=5, units=_mm())
        )
        form.heading("Wall")
        self.material = form.add("Material", Choice([(name, name) for name in COOLING_MATERIALS]))
        self.material.set_value("GRCop-84")
        self.run_button = primary_button("Analyse cooling", "run")
        self.run_button.clicked.connect(self.run)
        form.span(self.run_button)
        form.span(hint("Uses the chamber contour and gas state from the last engine run."))
        side = QWidget()
        side.setFixedWidth(300)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(18, 12, 12, 12)
        side_layout.addWidget(form)
        side_layout.addStretch()
        layout.addWidget(side)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.m_wall = Metric("Peak wall temperature", Quantity.TEMPERATURE)
        self.m_margin = Metric("Margin to melting", unit="K", digits=3)
        self.m_flux = Metric("Peak heat flux", Quantity.HEAT_FLUX)
        self.m_dt = Metric("Coolant temperature rise", unit="K", digits=3)
        self.m_dp = Metric("Coolant pressure drop", Quantity.PRESSURE)
        self.m_re = Metric("Coolant Reynolds number", unit="", digits=3)
        right_layout.addWidget(
            metric_row(self.m_wall, self.m_margin, self.m_flux, self.m_dt, self.m_dp, self.m_re)
        )
        self.plot = PlotCanvas(self._render, self._data, "Run the engine, then analyse cooling.")
        right_layout.addWidget(self.plot, 1)
        self.note = hint("")
        self.note.setContentsMargins(18, 0, 18, 10)
        right_layout.addWidget(self.note)
        layout.addWidget(right, 1)

    def engine_changed(self) -> None:
        engine = self.workspace.data.engine
        run = self.workspace.run
        if run is not None:
            # As many channels as fit around the throat at the current width and land
            pitch = self.width.value_si() + self.land.value_si()
            self.channels.setValue(max(8, int(2.0 * np.pi * run.contour.throat_radius / pitch)))
        coolant = FUEL_COOLANT.get(engine.fuel)
        if coolant is not None:
            self.coolant.set_value(coolant)
            self.inlet_t.set_value_si(COOLANT_DATABASE[coolant].reference_inlet_temperature)
        self._results = None
        self.plot.refresh(has_data=False)

    def restyle(self) -> None:
        self.plot.restyle()

    def run(self) -> None:
        run = self.workspace.run
        if run is None:
            self.workspace.message.emit("Run the engine first")
            return
        result = run.result
        mdot = result.vacuum.mass_flow_rate
        fuel_flow = mdot / (1.0 + result.inputs.of_ratio)
        k, melting = COOLING_MATERIALS[self.material.value()]
        design = CoolingSystemDesign(
            channels=CoolingChannel(
                width=self.width.value_si(),
                height=self.height.value_si(),
                wall_thickness=self.wall.value_si(),
                land_width=self.land.value_si(),
                num_channels=self.channels.value(),
                length=float(run.contour.x[-1] - run.contour.x[0]),
            ),
            coolant=self.coolant.value(),
            coolant_inlet_temp=self.inlet_t.value_si(),
            coolant_inlet_pressure=self.inlet_p_ratio.value_si() * result.inputs.chamber_pressure,
            coolant_mass_flow=self.flow_fraction.value_si() * fuel_flow,
            wall_material=self.material.value(),
            wall_thermal_conductivity=k,
            wall_melting_point=melting,
            coolant_flow_from_exit=self.counterflow.isChecked(),
        )
        profile = list(zip(run.contour.x.tolist(), (2.0 * run.contour.r).tolist(), strict=True))
        conditions = {
            "T_chamber": result.chamber_temperature,
            "P_chamber": result.inputs.chamber_pressure,
            "gamma": result.gamma,
            "c_star": result.vacuum.c_star,
            "molecular_weight": result.molecular_weight,
            "throat_radius_of_curvature": 1.5 * run.contour.throat_radius,
        }
        self.run_button.setEnabled(False)
        self.workspace.tasks.submit(
            "cooling",
            lambda ctx: (
                design,
                analyze_cooling_system(design, profile, conditions, num_stations=80),
            ),
            on_result=self._done,
            on_error=self._failed,
        )

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.workspace.message.emit(f"Cooling analysis failed: {message}")

    def _done(self, outcome) -> None:
        self.run_button.setEnabled(True)
        design, results = outcome
        self._design, self._results = design, sorted(results, key=lambda r: r.axial_position)
        wall = np.array([r.wall_temp_gas_side for r in results])
        coolant_t = np.array([r.coolant_temp for r in results])
        coolant_p = np.array([r.coolant_pressure for r in results])
        flux = np.array([r.heat_flux for r in results])
        self.m_wall.set_value(float(wall.max()))
        self.m_margin.set_value(float(design.wall_melting_point - wall.max()))
        self.m_flux.set_value(float(flux.max()))
        self.m_dt.set_value(float(coolant_t.max() - coolant_t.min()))
        self.m_dp.set_value(float(coolant_p.max() - coolant_p.min()))
        props = COOLANT_DATABASE[design.coolant]
        reynolds = (
            props.density
            * results[0].coolant_velocity
            * design.channels.hydraulic_diameter
            / props.dynamic_viscosity
        )
        self.m_re.set_value(float(reynolds))
        warnings = []
        if wall.max() > design.wall_melting_point:
            warnings.append("the hot wall exceeds the melting point")
        if reynolds < 2300.0:
            warnings.append(
                "the coolant flow is laminar (Re < 2300); use fewer or smaller channels"
            )
        if coolant_t.max() > props.critical_temp and coolant_p.min() < props.critical_pressure:
            warnings.append(
                "the coolant passes its critical temperature below critical pressure (boiling)"
            )
        self.note.setObjectName("Warn" if warnings else "Hint")
        self.note.setText(
            ("Check: " + "; ".join(warnings) + ".")
            if warnings
            else "Single-phase coolant with constant properties; Bartz gas side, Gnielinski coolant side."
        )
        self.note.style().polish(self.note)
        self.workspace.results["cooling"] = self._results
        self.plot.refresh()

    def _data(self) -> dict[str, np.ndarray]:
        r = self._results
        return {
            "x_m": np.array([s.axial_position for s in r]),
            "wall_gas_k": np.array([s.wall_temp_gas_side for s in r]),
            "wall_coolant_k": np.array([s.wall_temp_coolant_side for s in r]),
            "coolant_k": np.array([s.coolant_temp for s in r]),
            "heat_flux_w_m2": np.array([s.heat_flux for s in r]),
            "coolant_pressure_pa": np.array([s.coolant_pressure for s in r]),
        }

    def _render(self, figure) -> None:
        p = theme.current()
        d = self._data()
        x = d["x_m"] * UNITS.show(1.0, Quantity.LENGTH)
        t = lambda arr: np.array([UNITS.show(v, Quantity.TEMPERATURE) for v in arr])  # noqa: E731
        ax1, ax2 = figure.subplots(2, 1, sharex=True)
        ax1.plot(x, t(d["wall_gas_k"]), color=p.signal, label="hot wall")
        ax1.plot(x, t(d["wall_coolant_k"]), color=p.series[1], label="cold wall")
        ax1.plot(x, t(d["coolant_k"]), color=p.series[2], label="coolant")
        ax1.axhline(
            UNITS.show(self._design.wall_melting_point, Quantity.TEMPERATURE),
            color=p.error,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
        )
        ax1.text(
            x[0],
            UNITS.show(self._design.wall_melting_point, Quantity.TEMPERATURE),
            f" melting, {self._design.wall_material}",
            va="bottom",
            fontsize=8,
            color=p.error,
        )
        ax1.set_ylabel(f"T  [{UNITS.unit(Quantity.TEMPERATURE)}]")
        ax1.legend(loc="upper left", ncols=3)
        ax2.plot(
            x, [UNITS.show(q, Quantity.HEAT_FLUX) for q in d["heat_flux_w_m2"]], color=p.signal
        )
        ax2.set_ylabel(f"q  [{UNITS.unit(Quantity.HEAT_FLUX)}]")
        ax2.set_xlabel(f"Axial position from throat  [{UNITS.unit(Quantity.LENGTH)}]")
        for ax in (ax1, ax2):
            ax.axvline(0.0, color=p.line_strong, linewidth=0.8)


class StabilityView(QWidget):
    """Acoustic modes of the chamber and the feed-system (chug) estimate."""

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        self.summary = caption("Run the engine to compute the chamber modes.")
        layout.addWidget(self.summary)
        self.table = ResultTable(["Mode", "Frequency", "Wavelength", "Description"])
        layout.addWidget(self.table, 1)
        layout.addWidget(
            hint(
                "Frequencies follow from the chamber geometry and the frozen speed of sound. "
                "Whether a mode grows depends on the injector and combustion response, which "
                "EnSim does not model, so no stability verdict is given."
            )
        )

    def engine_changed(self) -> None:
        run = self.workspace.run
        if run is None:
            return
        result = run.result
        instability = analyze_combustion_instability(
            chamber_length=run.contour.chamber_length,
            chamber_diameter=2.0 * run.contour.chamber_radius,
            gamma=result.gamma,
            T_chamber=result.chamber_temperature,
            mean_mw=result.molecular_weight,
        )
        rows = [
            [
                m.name,
                f"{fmt(m.frequency)} Hz",
                f"{fmt(UNITS.show(m.wavelength, Quantity.LENGTH))} {UNITS.unit(Quantity.LENGTH)}",
                m.description,
            ]
            for m in sorted(instability.all_modes, key=lambda m: m.frequency)[:12]
        ]
        self.table.set_rows(rows)
        chug = instability.chugging_frequency
        self.summary.setText(
            f"Sound speed {fmt(instability.speed_of_sound)} m/s   "
            f"Chamber {fmt(UNITS.show(run.contour.chamber_length, Quantity.LENGTH))} x "
            f"{fmt(UNITS.show(2 * run.contour.chamber_radius, Quantity.LENGTH))} {UNITS.unit(Quantity.LENGTH)}"
            + (f"   Chug estimate {fmt(chug)} Hz" if chug else "")
        )
        self.workspace.results["stability"] = instability


def _mm():
    from ensim.utils.units import UnitSystem

    return {UnitSystem.SI: "mm", UnitSystem.IMPERIAL: "in"}
