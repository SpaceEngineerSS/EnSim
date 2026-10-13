"""Flight section: 6-DOF trajectory of the project's vehicle and its dispersion."""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QSpinBox, QVBoxLayout, QWidget

from ensim.core.flight_6dof import FlightResult6DOF, simulate_flight_6dof
from ensim.core.monte_carlo import DispersionConfig, DispersionResult, run_monte_carlo
from ensim.utils.units import Quantity, UnitSystem

from .. import theme
from ..units import UNITS, fmt
from ..widgets.controls import (
    Form,
    Metric,
    NumberField,
    ResultTable,
    connect_all,
    hint,
    primary_button,
)
from ..widgets.plot import PlotCanvas, decimate
from .base import Section, caption, metric_row, tabs


def _flight_kwargs(workspace) -> dict:
    """Arguments for simulate_flight_6dof from the project and the last engine run."""
    result = workspace.result
    rocket = workspace.rocket()
    env = workspace.data.environment
    rec = workspace.data.recovery
    inputs = result.inputs
    return {
        "rocket": rocket,
        "thrust_vac": result.vacuum.thrust,
        "isp_vac": result.vacuum.isp,
        "burn_time": rocket.engine.propellant_mass / result.vacuum.mass_flow_rate,
        "exit_area": inputs.throat_area * inputs.expansion_ratio,
        "nozzle_exit_pressure": result.vacuum.exit_pressure,
        "launch_angle_deg": env.launch_angle,
        "launch_azimuth_deg": env.launch_azimuth,
        "rail_length": env.rail_length,
        "wind_speed": env.wind_speed,
        "wind_direction_deg": env.wind_direction,
        "use_wgs84": env.use_wgs84,
        "launch_latitude_deg": env.latitude,
        "launch_longitude_deg": env.longitude,
        "launch_altitude_m": env.launch_altitude,
        "chute_diameter": rec.main_diameter,
        "chute_cd": rec.main_cd,
        "deploy_at_apogee": not rec.dual_deploy,
        "main_deploy_altitude_m": rec.main_deploy_altitude,
        "drogue_diameter": rec.drogue_diameter if rec.dual_deploy else 0.0,
        "drogue_cd": rec.drogue_cd,
    }


class FlightSection(Section):
    title = "Flight"
    icon_name = "flight"
    run_label = "Fly"

    def build(self) -> None:
        self._loading = False
        self._result: FlightResult6DOF | None = None
        self._pending = False
        form = Form()
        form.heading("Launch")
        self.elevation = form.add(
            "Rail elevation", NumberField(87.0, 30.0, 90.0, decimals=1, step=0.5, suffix="deg")
        )
        self.azimuth = form.add(
            "Azimuth",
            NumberField(0.0, 0.0, 360.0, decimals=1, step=5.0, suffix="deg"),
            "Clockwise from north",
        )
        self.rail = form.add(
            "Rail length", NumberField(3.0, 0.0, 50.0, Quantity.LENGTH, decimals=2, step=0.5)
        )
        form.heading("Weather")
        self.wind = form.add(
            "Wind at 10 m", NumberField(0.0, 0.0, 40.0, Quantity.VELOCITY, decimals=1, step=0.5)
        )
        self.wind_dir = form.add(
            "Wind from", NumberField(270.0, 0.0, 360.0, decimals=0, step=10.0, suffix="deg")
        )
        form.heading("Site")
        self.site_alt = form.add(
            "Altitude", NumberField(0.0, -400.0, 5000.0, Quantity.LENGTH, decimals=0, step=10.0)
        )
        self.wgs84 = form.span(QCheckBox("Round, rotating Earth (WGS-84, J2)"))
        self.lat = form.add(
            "Latitude", NumberField(0.0, -90.0, 90.0, decimals=4, step=0.1, suffix="deg")
        )
        self.lon = form.add(
            "Longitude", NumberField(0.0, -180.0, 180.0, decimals=4, step=0.1, suffix="deg")
        )
        form.heading("Integration")
        self.adaptive = form.span(QCheckBox("Adaptive step (Dormand-Prince 5(4))"))
        self.dt = form.add("Step", NumberField(0.01, 1e-4, 0.1, decimals=4, step=0.005, suffix="s"))
        self.max_time = form.add(
            "Stop after", NumberField(600.0, 5.0, 7200.0, decimals=0, step=30.0, suffix="s")
        )
        fields = [
            self.elevation,
            self.azimuth,
            self.rail,
            self.wind,
            self.wind_dir,
            self.site_alt,
            self.wgs84,
            self.lat,
            self.lon,
        ]
        connect_all(fields, self._store)
        self.wgs84.toggled.connect(self._toggle_site)

        self.dispersion = DispersionView(self)
        self.split(
            form,
            tabs(
                ("Summary", self._summary_view()),
                ("Trajectory", self._trajectory_view()),
                ("Histories", self._history_view()),
                ("Dispersion", self.dispersion),
            ),
        )
        self.workspace.project_loaded.connect(self.load_from_project)
        self.workspace.engine_result.connect(self._engine_ready)
        self.workspace.engine_failed.connect(lambda _m: self._abort_pending())

    def _toggle_site(self, on: bool) -> None:
        self.lat.setEnabled(on)
        self.lon.setEnabled(on)

    def _summary_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.m_apogee = Metric("Apogee", Quantity.LENGTH)
        self.m_speed = Metric("Max speed", Quantity.VELOCITY)
        self.m_mach = Metric("Max Mach", unit="", digits=3)
        self.m_accel = Metric("Max acceleration", unit="g", digits=3)
        self.m_q = Metric("Max q", Quantity.PRESSURE)
        self.m_range = Metric("Landing distance", Quantity.LENGTH)
        layout.addWidget(
            metric_row(
                self.m_apogee, self.m_speed, self.m_mach, self.m_accel, self.m_q, self.m_range
            )
        )
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(18, 0, 18, 12)
        body_layout.addWidget(caption("Events"))
        self.events = ResultTable(["Event", "Time  [s]", "Altitude", "Speed"])
        body_layout.addWidget(self.events, 1)
        self.flight_note = hint("Run the engine and set up the vehicle, then fly (F5).")
        body_layout.addWidget(self.flight_note)
        layout.addWidget(body, 1)
        return page

    def _trajectory_view(self) -> QWidget:
        self.trajectory_plot = PlotCanvas(
            self._render_trajectory, self._trajectory_data, "Fly to see the trajectory."
        )
        return self.trajectory_plot

    def _history_view(self) -> QWidget:
        self.history_plot = PlotCanvas(
            self._render_history, self._history_data, "Fly to see the time histories."
        )
        return self.history_plot

    # project

    def load_from_project(self) -> None:
        if not self.built:
            return
        env = self.workspace.data.environment
        self._loading = True
        try:
            self.elevation.set_value_si(env.launch_angle)
            self.azimuth.set_value_si(env.launch_azimuth)
            self.rail.set_value_si(env.rail_length)
            self.wind.set_value_si(env.wind_speed)
            self.wind_dir.set_value_si(env.wind_direction)
            self.site_alt.set_value_si(env.launch_altitude)
            self.wgs84.setChecked(env.use_wgs84)
            self.lat.set_value_si(env.latitude)
            self.lon.set_value_si(env.longitude)
            self._toggle_site(env.use_wgs84)
        finally:
            self._loading = False

    def _store(self) -> None:
        if self._loading:
            return
        self.workspace.update(
            "environment",
            launch_angle=self.elevation.value_si(),
            launch_azimuth=self.azimuth.value_si(),
            rail_length=self.rail.value_si(),
            wind_speed=self.wind.value_si(),
            wind_direction=self.wind_dir.value_si(),
            launch_altitude=self.site_alt.value_si(),
            use_wgs84=self.wgs84.isChecked(),
            latitude=self.lat.value_si(),
            longitude=self.lon.value_si(),
        )

    # run

    def run(self) -> None:
        if self.workspace.result is None:
            self._pending = True
            self.set_running(True)
            self.workspace.message.emit("Running the engine first")
            self.workspace.run_engine()
            return
        try:
            kwargs = _flight_kwargs(self.workspace)
        except ValueError as error:
            self.workspace.message.emit(str(error))
            return
        kwargs.update(
            dt=self.dt.value_si(),
            max_time=self.max_time.value_si(),
            use_adaptive=self.adaptive.isChecked(),
            output_dt=max(self.dt.value_si(), 0.01),
        )
        self.set_running(True)

        def work(ctx):
            return simulate_flight_6dof(
                **kwargs,
                cancel=ctx.cancelled,
                progress=lambda f: ctx.progress(f, f"t = {f * kwargs['max_time']:.0f} s"),
            )

        self.workspace.tasks.submit(
            "flight",
            work,
            on_result=self._done,
            on_error=self._failed,
            on_cancel=lambda: self.set_running(False),
        )

    def _engine_ready(self, _result) -> None:
        if self._pending:
            self._pending = False
            self.set_running(False)
            self.run()

    def _abort_pending(self) -> None:
        if self._pending:
            self._pending = False
            self.set_running(False)

    def _failed(self, message: str) -> None:
        self.set_running(False)
        self.flight_note.setText(f"Flight failed: {message}")
        self.workspace.message.emit(f"Flight failed: {message}")

    def _done(self, result: FlightResult6DOF) -> None:
        self.set_running(False)
        self._result = result
        self.workspace.results["flight"] = result
        self.workspace.data.last_apogee = result.apogee_altitude
        self.m_apogee.set_value(result.apogee_altitude)
        self.m_speed.set_value(result.max_velocity)
        self.m_mach.set_value(result.max_mach)
        self.m_accel.set_value(result.max_acceleration)
        self.m_q.set_value(result.max_q)
        distance = np.hypot(result.landing_x, result.landing_y) if result.ground_impact else None
        self.m_range.set_value(distance)

        def at(t):
            i = min(int(np.searchsorted(result.time, t)), len(result.time) - 1)
            return [
                fmt(UNITS.show(result.position_z[i], Quantity.LENGTH))
                + " "
                + UNITS.unit(Quantity.LENGTH),
                fmt(UNITS.show(result.velocity_magnitude[i], Quantity.VELOCITY))
                + " "
                + UNITS.unit(Quantity.VELOCITY),
            ]

        rows = [
            ["Liftoff", result.liftoff_time, *at(result.liftoff_time)],
            ["Burnout", result.burnout_time, *at(result.burnout_time)],
        ]
        if result.apogee_reached:
            rows.append(
                [
                    "Apogee",
                    result.apogee_time,
                    fmt(UNITS.show(result.apogee_altitude, Quantity.LENGTH))
                    + " "
                    + UNITS.unit(Quantity.LENGTH),
                    at(result.apogee_time)[1],
                ]
            )
        if result.drogue_deploy_time is not None:
            rows.append(["Drogue", result.drogue_deploy_time, *at(result.drogue_deploy_time)])
        if result.parachute_deploy_time is not None:
            rows.append(
                ["Main parachute", result.parachute_deploy_time, *at(result.parachute_deploy_time)]
            )
        rows.append(
            [
                "Landing" if result.ground_impact else "End of run",
                result.flight_time,
                *at(result.flight_time),
            ]
        )
        self.events.set_rows(rows)
        notes = []
        if not result.success:
            notes.append(result.abort_reason or "The vehicle did not lift off.")
        if not result.ground_impact:
            notes.append("Still airborne at the time limit.")
        if result.forced_min_steps:
            notes.append(f"{result.forced_min_steps} adaptive steps hit the minimum step size.")
        self.flight_note.setText(
            "  ".join(notes) or "Point-mass atmosphere US76; Barrowman aerodynamics; rigid body."
        )
        self.trajectory_plot.refresh()
        self.history_plot.refresh()

    def restyle(self) -> None:
        if self.built:
            self.trajectory_plot.restyle()
            self.history_plot.restyle()
            self.dispersion.plot.restyle()

    # plots

    def _length(self, values):
        return np.asarray(values) * UNITS.show(1.0, Quantity.LENGTH)

    def _trajectory_data(self):
        r = self._result
        return {
            "t_s": r.time,
            "east_m": r.position_x,
            "north_m": r.position_y,
            "up_m": r.position_z,
        }

    def _render_trajectory(self, figure) -> None:
        p = theme.current()
        r = self._result
        unit = UNITS.unit(Quantity.LENGTH)
        downrange = np.hypot(r.position_x, r.position_y)
        ax1, ax2 = figure.subplots(1, 2, width_ratios=[1.6, 1.0])
        x, z = decimate(self._length(downrange), self._length(r.position_z))
        ax1.plot(x, z, color=p.signal)
        if r.apogee_reached:
            i = int(np.searchsorted(r.time, r.apogee_time))
            ax1.plot(
                self._length(downrange[i]),
                self._length(r.apogee_altitude),
                "o",
                markersize=4,
                color=p.text,
            )
        ax1.set_xlabel(f"Ground distance  [{unit}]")
        ax1.set_ylabel(f"Altitude  [{unit}]")
        ax1.set_title("Profile")
        e, n = decimate(self._length(r.position_x), self._length(r.position_y))
        ax2.plot(e, n, color=p.series[1])
        ax2.plot([0], [0], marker="^", color=p.text, markersize=6)
        if r.ground_impact:
            ax2.plot(
                self._length(r.landing_x),
                self._length(r.landing_y),
                "x",
                color=p.signal,
                markersize=7,
            )
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.set_xlabel(f"East  [{unit}]")
        ax2.set_ylabel(f"North  [{unit}]")
        ax2.set_title("Ground track")

    def _history_data(self):
        r = self._result
        return {
            "t_s": r.time, "altitude_m": r.position_z, "speed_m_s": r.velocity_magnitude,
            "mach": r.mach, "dynamic_pressure_pa": r.dynamic_pressure, "thrust_n": r.thrust,
            "drag_n": r.drag, "angle_of_attack_deg": r.angle_of_attack, "mass_kg": r.mass,
        }  # fmt: skip

    def _render_history(self, figure) -> None:
        p = theme.current()
        r = self._result
        axes = figure.subplots(2, 2, sharex=True).ravel()
        series = [
            (
                r.position_z * UNITS.show(1.0, Quantity.LENGTH),
                f"Altitude  [{UNITS.unit(Quantity.LENGTH)}]",
                p.signal,
            ),
            (r.mach, "Mach", p.series[1]),
            (
                np.array([UNITS.show(f, Quantity.FORCE) for f in r.thrust]),
                f"Thrust  [{UNITS.unit(Quantity.FORCE)}]",
                p.series[2],
            ),
            (r.angle_of_attack, "Angle of attack  [deg]", p.series[3]),
        ]
        for ax, (values, label, color) in zip(axes, series, strict=True):
            t, v = decimate(r.time, values)
            ax.plot(t, v, color=color)
            ax.set_ylabel(label)
        axes[2].plot(
            *decimate(r.time, np.array([UNITS.show(f, Quantity.FORCE) for f in r.drag])),
            color=p.muted,
            linewidth=1.0,
        )
        axes[2].legend(["thrust", "drag"], loc="upper right")
        for ax in axes[2:]:
            ax.set_xlabel("Time  [s]")
        for ax in axes:
            if r.apogee_reached:
                ax.axvline(r.apogee_time, color=p.line_strong, linewidth=0.8, linestyle=(0, (3, 3)))


class DispersionView(QWidget):
    """Monte Carlo landing dispersion."""

    def __init__(self, section: FlightSection):
        super().__init__()
        self.section = section
        self._result: DispersionResult | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = Form()
        self.runs = QSpinBox()
        self.runs.setRange(10, 5000)
        self.runs.setValue(100)
        form.add("Runs", self.runs)
        self.thrust_sigma = form.add(
            "Thrust 1-sigma", NumberField(0.02, 0.0, 0.3, decimals=3, step=0.005, suffix="x")
        )
        self.isp_sigma = form.add(
            "Isp 1-sigma", NumberField(0.01, 0.0, 0.2, decimals=3, step=0.005, suffix="x")
        )
        self.cd_sigma = form.add(
            "Cd 1-sigma", NumberField(0.05, 0.0, 0.5, decimals=3, step=0.01, suffix="x")
        )
        self.angle_sigma = form.add(
            "Elevation 1-sigma", NumberField(0.5, 0.0, 5.0, decimals=2, step=0.1, suffix="deg")
        )
        self.fin_max = form.add(
            "Fin misalignment",
            NumberField(0.25, 0.0, 3.0, decimals=2, step=0.05, suffix="deg"),
            "Uniform within plus/minus this angle",
        )
        self.wind_mean = form.add(
            "Wind mean", NumberField(3.0, 0.0, 30.0, Quantity.VELOCITY, decimals=1, step=0.5)
        )
        self.wind_sigma = form.add(
            "Wind 1-sigma", NumberField(1.5, 0.0, 15.0, Quantity.VELOCITY, decimals=1, step=0.5)
        )
        self.random_dir = form.span(QCheckBox("Random wind direction"))
        self.random_dir.setChecked(True)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(2026)
        form.add("Seed", self.seed)
        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(max(1, (os.cpu_count() or 2) - 1))
        form.add(
            "Processes", self.workers, "1 runs inside the application; more use separate processes"
        )
        self.run_button = primary_button("Run dispersion", "run")
        self.run_button.clicked.connect(self.run)
        form.span(self.run_button)
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
        self.m_cep = Metric("CEP (50 %)", Quantity.LENGTH)
        self.m_major = Metric("99.7 % ellipse, semi-major", Quantity.LENGTH)
        self.m_apogee = Metric("Apogee mean", Quantity.LENGTH)
        self.m_apogee_sd = Metric("Apogee 1-sigma", Quantity.LENGTH)
        self.m_landed = Metric("Landed", unit="runs", digits=4)
        right_layout.addWidget(
            metric_row(self.m_cep, self.m_major, self.m_apogee, self.m_apogee_sd, self.m_landed)
        )
        self.plot = PlotCanvas(self._render, self._data, "Fly once, then run the dispersion.")
        right_layout.addWidget(self.plot, 1)
        layout.addWidget(right, 1)

    def run(self) -> None:
        ws = self.section.workspace
        if ws.result is None:
            ws.message.emit("Run the engine first")
            return
        kwargs = _flight_kwargs(ws)
        config = DispersionConfig(
            num_simulations=self.runs.value(),
            thrust_sigma=self.thrust_sigma.value_si(),
            isp_sigma=self.isp_sigma.value_si(),
            cd_sigma=self.cd_sigma.value_si(),
            launch_angle_sigma=self.angle_sigma.value_si(),
            fin_misalignment_max=self.fin_max.value_si(),
            wind_speed_mean=self.wind_mean.value_si(),
            wind_speed_sigma=self.wind_sigma.value_si(),
            randomize_wind_dir=self.random_dir.isChecked(),
            wind_direction_mean=ws.data.environment.wind_direction,
            seed=self.seed.value(),
            n_workers=self.workers.value(),
        )
        self.run_button.setEnabled(False)

        def work(ctx):
            return run_monte_carlo(
                kwargs["rocket"],
                kwargs["thrust_vac"],
                kwargs["isp_vac"],
                kwargs["burn_time"],
                config=config,
                launch_angle_deg=kwargs["launch_angle_deg"],
                launch_azimuth_deg=kwargs["launch_azimuth_deg"],
                exit_area=kwargs["exit_area"],
                dt=0.02,
                max_time=self.section.max_time.value_si(),
                rail_length=kwargs["rail_length"],
                chute_diameter=kwargs["chute_diameter"],
                deploy_at_apogee=kwargs["deploy_at_apogee"],
                progress=lambda done, total: ctx.progress(done / total, f"{done} of {total} runs"),
                cancel=ctx.cancelled,
                flight_options={
                    key: kwargs[key]
                    for key in (
                        "chute_cd",
                        "main_deploy_altitude_m",
                        "drogue_diameter",
                        "drogue_cd",
                        "nozzle_exit_pressure",
                        "use_wgs84",
                        "launch_latitude_deg",
                        "launch_longitude_deg",
                        "launch_altitude_m",
                    )
                },
            )

        ws.tasks.submit(
            "dispersion",
            work,
            on_result=self._done,
            on_error=self._failed,
            on_cancel=lambda: self.run_button.setEnabled(True),
        )

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.section.workspace.message.emit(f"Dispersion failed: {message}")

    def _done(self, result: DispersionResult) -> None:
        self.run_button.setEnabled(True)
        self._result = result
        self.section.workspace.results["dispersion"] = result
        landed = result.n_landed > 1
        self.m_cep.set_value(result.cep_radius if landed else None)
        self.m_major.set_value(result.ellipse_semi_major if landed else None)
        self.m_apogee.set_value(result.apogee_mean)
        self.m_apogee_sd.set_value(result.apogee_std)
        self.m_landed.set_value(float(result.n_landed))
        self.plot.refresh()

    def _data(self):
        r = self._result
        return {
            "landing_east_m": r.landing_points[:, 0],
            "landing_north_m": r.landing_points[:, 1],
            "apogee_m": r.apogees,
        }

    def _render(self, figure) -> None:
        from matplotlib.patches import Circle, Ellipse

        p = theme.current()
        r = self._result
        scale = UNITS.show(1.0, Quantity.LENGTH)
        unit = UNITS.unit(Quantity.LENGTH)
        ax1, ax2 = figure.subplots(1, 2, width_ratios=[1.3, 1.0])
        if r.n_landed > 1:
            pts = r.landing_points * scale
            ax1.scatter(pts[:, 0], pts[:, 1], s=8, color=p.series[1], alpha=0.7, linewidths=0)
            centre = (r.landing_x_mean * scale, r.landing_y_mean * scale)
            ax1.add_patch(
                Circle(centre, r.cep_radius * scale, fill=False, color=p.text, linewidth=1.0)
            )
            ax1.add_patch(
                Ellipse(
                    centre,
                    2 * r.ellipse_semi_major * scale,
                    2 * r.ellipse_semi_minor * scale,
                    angle=r.ellipse_angle_deg,
                    fill=False,
                    color=p.signal,
                    linewidth=1.2,
                )
            )
            ax1.plot([0], [0], marker="^", color=p.text, markersize=6)
            ax1.set_aspect("equal", adjustable="datalim")
            ax1.legend(["landings", "CEP", "99.7 % ellipse"], loc="upper right")
        else:
            ax1.text(
                0.5,
                0.5,
                "Fewer than two runs landed\nbefore the time limit.",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                color=p.muted,
                fontsize=9,
            )
        ax1.set_xlabel(f"East  [{unit}]")
        ax1.set_ylabel(f"North  [{unit}]")
        ax1.set_title(f"Landing points, {r.n_landed} of {r.num_simulations}")
        divisor = 1000.0 if UNITS.system == UnitSystem.SI else 1.0
        ax2.hist(
            r.apogees * scale / divisor,
            bins=min(30, max(8, len(r.apogees) // 5)),
            color=p.signal,
            alpha=0.85,
        )
        ax2.set_xlabel(f"Apogee  [{'km' if divisor > 1 else unit}]")
        ax2.set_ylabel("Runs")
        ax2.set_title("Apogee spread")
