"""Vehicle section: airframe, mass, stability and multi-stage delta-v."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ensim.core.aero import analyze_rocket
from ensim.core.constants import G0
from ensim.core.rocket import NoseShape
from ensim.core.staging import (
    create_custom_vehicle,
    create_heavy_lift_reference,
    create_medium_lift_reference,
)
from ensim.utils.units import Quantity

from .. import theme
from ..units import UNITS, fmt
from ..widgets.controls import Choice, Form, Metric, NumberField, ResultTable, connect_all, hint
from ..widgets.plot import PlotCanvas
from .base import Section, caption, metric_row, tabs


def nose_profile(
    shape: NoseShape, length: float, radius: float, n: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Radius along a nose cone, x from the tip."""
    x = np.linspace(0.0, length, n)
    if shape is NoseShape.CONICAL:
        r = radius * x / length
    elif shape is NoseShape.OGIVE:
        rho = (radius**2 + length**2) / (2.0 * radius)
        r = np.sqrt(np.maximum(rho**2 - (length - x) ** 2, 0.0)) + radius - rho
    elif shape is NoseShape.ELLIPTICAL:
        r = radius * np.sqrt(np.maximum(1.0 - ((length - x) / length) ** 2, 0.0))
    elif shape is NoseShape.PARABOLIC:
        u = x / length
        r = radius * (2.0 * u - u * u)
    else:  # LD-Haack (Von Karman)
        theta = np.arccos(1.0 - 2.0 * x / length)
        r = radius / np.sqrt(np.pi) * np.sqrt(theta - np.sin(2.0 * theta) / 2.0)
    return x, np.maximum(r, 0.0)


class VehicleSection(Section):
    title = "Vehicle"
    icon_name = "vehicle"

    def build(self) -> None:
        self._loading = False
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(150)
        self._redraw.timeout.connect(self._update_design)
        length = Quantity.LENGTH
        mass = Quantity.MASS

        form = Form()
        self.name = form.add("Name", QLineEdit())
        form.heading("Nose")
        self.nose_shape = form.add(
            "Shape",
            Choice(
                [
                    (s.value.capitalize() if s is not NoseShape.HAACK else "LD-Haack", s)
                    for s in NoseShape
                ]
            ),
        )
        self.nose_length = form.add(
            "Length", NumberField(0.45, 0.05, 5.0, length, decimals=3, step=0.01)
        )
        self.nose_mass = form.add(
            "Mass", NumberField(0.7, 0.01, 500.0, mass, decimals=3, step=0.05)
        )
        form.heading("Body")
        self.body_length = form.add(
            "Length", NumberField(1.8, 0.1, 30.0, length, decimals=3, step=0.05)
        )
        self.body_diameter = form.add(
            "Diameter", NumberField(0.12, 0.02, 5.0, length, decimals=4, step=0.005)
        )
        self.body_mass = form.add(
            "Mass", NumberField(2.5, 0.05, 5000.0, mass, decimals=3, step=0.1)
        )
        self.cd = form.add(
            "Axial drag coefficient", NumberField(0.45, 0.05, 2.0, decimals=3, step=0.01)
        )
        form.heading("Fins")
        self.fin_count = QSpinBox()
        self.fin_count.setRange(3, 8)
        form.add("Count", self.fin_count)
        self.fin_root = form.add(
            "Root chord", NumberField(0.26, 0.01, 3.0, length, decimals=3, step=0.01)
        )
        self.fin_tip = form.add(
            "Tip chord", NumberField(0.10, 0.0, 3.0, length, decimals=3, step=0.01)
        )
        self.fin_span = form.add(
            "Span", NumberField(0.16, 0.01, 3.0, length, decimals=3, step=0.01)
        )
        self.fin_sweep = form.add(
            "Leading-edge sweep", NumberField(35.0, 0.0, 70.0, decimals=1, step=1.0, suffix="deg")
        )
        self.fin_mass = form.add(
            "Mass (all fins)", NumberField(0.6, 0.01, 500.0, mass, decimals=3, step=0.05)
        )
        form.heading("Propulsion")
        self.engine_mass = form.add(
            "Engine and tanks, dry", NumberField(2.0, 0.01, 50_000.0, mass, decimals=3, step=0.1)
        )
        self.tank_length = form.add(
            "Tank length", NumberField(0.45, 0.05, 30.0, length, decimals=3, step=0.01)
        )
        self.propellant = form.add(
            "Propellant",
            NumberField(3.2, 0.01, 500_000.0, mass, decimals=3, step=0.1),
            "Split into fuel and oxidizer at the engine's O/F",
        )
        form.heading("Recovery")
        self.dual = form.span(QCheckBox("Dual deployment (drogue at apogee)"))
        self.drogue_d = form.add(
            "Drogue diameter", NumberField(0.45, 0.05, 20.0, length, decimals=3, step=0.05)
        )
        self.drogue_cd = form.add("Drogue Cd", NumberField(1.2, 0.3, 2.5, decimals=2, step=0.05))
        self.main_d = form.add(
            "Main diameter", NumberField(1.5, 0.1, 60.0, length, decimals=3, step=0.05)
        )
        self.main_cd = form.add("Main Cd", NumberField(1.5, 0.3, 2.5, decimals=2, step=0.05))
        self.main_alt = form.add(
            "Main deploys at", NumberField(300.0, 30.0, 10_000.0, length, decimals=0, step=50.0)
        )

        self._fields = [
            self.nose_shape, self.nose_length, self.nose_mass, self.body_length, self.body_diameter,
            self.body_mass, self.cd, self.fin_count, self.fin_root, self.fin_tip, self.fin_span,
            self.fin_sweep, self.fin_mass, self.engine_mass, self.tank_length, self.propellant,
            self.dual, self.drogue_d, self.drogue_cd, self.main_d, self.main_cd, self.main_alt,
        ]  # fmt: skip
        connect_all(self._fields, self._store)
        self.fin_count.valueChanged.connect(lambda _v: self._store())
        self.name.editingFinished.connect(self._store)
        self.dual.toggled.connect(self._toggle_drogue)

        self.stages = StagesView()
        self.split(form, tabs(("Design", self._design_view()), ("Stages", self.stages)))
        self.workspace.project_loaded.connect(self.load_from_project)
        self.workspace.engine_result.connect(lambda _r: self._redraw.start())

    def _design_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.m_margin = Metric("Static margin, full", unit="cal", digits=3)
        self.m_margin_dry = Metric("Static margin, empty", unit="cal", digits=3)
        self.m_wet = Metric("Liftoff mass", Quantity.MASS)
        self.m_tw = Metric("Thrust / weight", unit="", digits=3)
        self.m_burn = Metric("Burn time", unit="s", digits=3)
        layout.addWidget(
            metric_row(self.m_margin, self.m_margin_dry, self.m_wet, self.m_tw, self.m_burn)
        )
        self.diagram = PlotCanvas(self._render, self._diagram_data, "")
        layout.addWidget(self.diagram, 1)
        self.design_note = hint("")
        self.design_note.setContentsMargins(18, 0, 18, 10)
        layout.addWidget(self.design_note)
        return page

    def _toggle_drogue(self, on: bool) -> None:
        self.drogue_d.setEnabled(on)
        self.drogue_cd.setEnabled(on)

    # project

    def load_from_project(self) -> None:
        if not self.built:
            return
        rd = self.workspace.data.rocket
        rec = self.workspace.data.recovery
        self._loading = True
        try:
            self.name.setText(rd.name)
            self.name.setCursorPosition(0)
            try:
                self.nose_shape.set_value(NoseShape(rd.nose_shape))
            except ValueError:
                self.nose_shape.set_value(NoseShape.OGIVE)
            for field, value in (
                (self.nose_length, rd.nose_length), (self.nose_mass, rd.nose_mass),
                (self.body_length, rd.body_length), (self.body_diameter, rd.body_diameter),
                (self.body_mass, rd.body_mass), (self.cd, rd.axial_drag_coefficient),
                (self.fin_root, rd.fin_root_chord), (self.fin_tip, rd.fin_tip_chord),
                (self.fin_span, rd.fin_span), (self.fin_sweep, rd.fin_sweep_angle),
                (self.fin_mass, rd.fin_mass), (self.engine_mass, rd.engine_mass_dry),
                (self.tank_length, rd.tank_length), (self.propellant, rd.fuel_mass + rd.oxidizer_mass),
                (self.drogue_d, rec.drogue_diameter), (self.drogue_cd, rec.drogue_cd),
                (self.main_d, rec.main_diameter), (self.main_cd, rec.main_cd),
                (self.main_alt, rec.main_deploy_altitude),
            ):  # fmt: skip
                field.set_value_si(value)
            self.fin_count.setValue(rd.fin_count)
            self.dual.setChecked(rec.dual_deploy)
            self._toggle_drogue(rec.dual_deploy)
        finally:
            self._loading = False
        self._redraw.stop()
        self._update_design()

    def _store(self) -> None:
        if self._loading:
            return
        of = max(self.workspace.data.engine.of_ratio, 1e-6)
        propellant = self.propellant.value_si()
        self.workspace.update(
            "rocket",
            name=self.name.text() or "Rocket",
            nose_shape=self.nose_shape.value().value,
            nose_length=self.nose_length.value_si(),
            nose_diameter=self.body_diameter.value_si(),
            nose_mass=self.nose_mass.value_si(),
            body_length=self.body_length.value_si(),
            body_diameter=self.body_diameter.value_si(),
            body_mass=self.body_mass.value_si(),
            axial_drag_coefficient=self.cd.value_si(),
            fin_count=self.fin_count.value(),
            fin_root_chord=self.fin_root.value_si(),
            fin_tip_chord=self.fin_tip.value_si(),
            fin_span=self.fin_span.value_si(),
            fin_sweep_angle=self.fin_sweep.value_si(),
            fin_mass=self.fin_mass.value_si(),
            engine_mass_dry=self.engine_mass.value_si(),
            tank_length=self.tank_length.value_si(),
            fuel_mass=propellant / (1.0 + of),
            oxidizer_mass=propellant * of / (1.0 + of),
        )
        self.workspace.update(
            "recovery",
            dual_deploy=self.dual.isChecked(),
            drogue_diameter=self.drogue_d.value_si(),
            drogue_cd=self.drogue_cd.value_si(),
            main_diameter=self.main_d.value_si(),
            main_cd=self.main_cd.value_si(),
            main_deploy_altitude=self.main_alt.value_si(),
        )
        self._redraw.start()

    # design view

    def _update_design(self) -> None:
        try:
            self._rocket = self.workspace.rocket()
            self._aero = analyze_rocket(self._rocket)
        except ValueError as error:
            self.design_note.setText(str(error))
            return
        rocket = self._rocket
        cp = self._aero.cp_total
        dry_margin = (cp - rocket.get_cg_from_propellant(0.0)) / rocket.reference_diameter
        self.m_margin.set_value(self._aero.stability_margin)
        self.m_margin_dry.set_value(dry_margin)
        self.m_wet.set_value(rocket.wet_mass)
        result = self.workspace.result
        notes = []
        if result is not None and result.sea_level is not None:
            thrust = result.sea_level.thrust
            self.m_tw.set_value(thrust / (rocket.wet_mass * G0))
            self.m_burn.set_value(rocket.engine.propellant_mass / result.vacuum.mass_flow_rate)
            if thrust < rocket.wet_mass * G0:
                notes.append("Sea-level thrust is below the liftoff weight.")
        else:
            self.m_tw.set_value(None)
            self.m_burn.set_value(None)
            notes.append("Run the engine for thrust-to-weight and burn time.")
        if self._aero.stability_margin < 1.0:
            notes.append(
                "Margin below 1 caliber at liftoff: move mass forward or enlarge the fins."
            )
        elif self._aero.stability_margin > 4.0:
            notes.append("Margin above 4 calibers: the vehicle will weathercock strongly in wind.")
        self.design_note.setText("  ".join(notes) or "Barrowman static stability at Mach < 0.5.")
        self.diagram.refresh()

    def _diagram_data(self) -> dict[str, np.ndarray]:
        x, r = self._outline()
        return {"x_m": x, "radius_m": r}

    def _outline(self) -> tuple[np.ndarray, np.ndarray]:
        rocket = self._rocket
        radius = rocket.reference_diameter / 2.0
        xn, rn = nose_profile(rocket.nose.shape, rocket.nose.length, radius)
        end = rocket.total_length
        return np.concatenate([xn, [rocket.nose.length, end]]), np.concatenate(
            [rn, [radius, radius]]
        )

    def _render(self, figure) -> None:
        p = theme.current()
        rocket = self._rocket
        scale = UNITS.show(1.0, Quantity.LENGTH)
        unit = UNITS.unit(Quantity.LENGTH)
        x, r = self._outline()
        ax = figure.add_subplot(111)
        ax.fill_between(x * scale, r * scale, -r * scale, color=p.raised, linewidth=0)
        ax.plot(x * scale, r * scale, color=p.text, linewidth=1.2)
        ax.plot(x * scale, -r * scale, color=p.text, linewidth=1.2)
        end = rocket.total_length
        ax.plot([end * scale] * 2, [-r[-1] * scale, r[-1] * scale], color=p.text, linewidth=1.2)
        ax.plot(
            [rocket.nose.length * scale] * 2,
            [-r[-1] * scale, r[-1] * scale],
            color=p.line_strong,
            linewidth=0.8,
        )

        fin = rocket.fins.fin
        root_le = rocket.fins.position
        sweep = np.tan(np.radians(fin.sweep_angle)) * fin.span
        radius = rocket.reference_diameter / 2.0
        for sign in (1.0, -1.0):
            fx = np.array(
                [
                    root_le,
                    root_le + sweep,
                    root_le + sweep + fin.tip_chord,
                    root_le + fin.root_chord,
                ]
            )
            fy = sign * np.array([radius, radius + fin.span, radius + fin.span, radius])
            ax.fill(fx * scale, fy * scale, color=p.surface, edgecolor=p.text, linewidth=1.2)

        tank_start = rocket.engine.position - rocket.engine.tank_length / 2.0
        ax.add_patch(
            __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (tank_start * scale, -0.6 * radius * scale),
                rocket.engine.tank_length * scale,
                1.2 * radius * scale,
                facecolor="none",
                edgecolor=p.muted,
                linewidth=0.8,
                linestyle=(0, (3, 2)),
            )
        )
        cg_full = rocket.get_cg_from_propellant(rocket.engine.propellant_mass)
        cg_dry = rocket.get_cg_from_propellant(0.0)
        cp = self._aero.cp_total
        marker_y = (radius + fin.span) * 1.25 * scale
        for value, label, color in (
            (cg_full, "CG", p.text),
            (cg_dry, "CG empty", p.muted),
            (cp, "CP", p.signal),
        ):
            ax.plot(
                [value * scale],
                [0.0],
                marker="o",
                markersize=7,
                markerfacecolor=p.bg,
                markeredgecolor=color,
                markeredgewidth=1.5,
            )
            ax.annotate(
                label,
                (value * scale, 0.0),
                (value * scale, marker_y),
                ha="center",
                fontsize=8,
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6},
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"Station from nose tip  [{unit}]")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.set_title(
            f"{rocket.name}   {fmt(UNITS.show(rocket.total_length, Quantity.LENGTH))} {unit} long, "
            f"{fmt(UNITS.show(rocket.reference_diameter, Quantity.LENGTH))} {unit} diameter"
        )

    def restyle(self) -> None:
        if self.built:
            self.diagram.restyle()
            self.stages.plot.restyle()


class StagesView(QWidget):
    """Serial staging and ideal delta-v, edited as a table."""

    COLUMNS = [
        "Stage",
        "Dry  [t]",
        "Propellant  [t]",
        "Engines",
        "Thrust/engine vac  [kN]",
        "Isp SL  [s]",
        "Isp vac  [s]",
    ]

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        top = QHBoxLayout()
        self.preset = Choice(
            [
                ("Medium-lift reference", "medium"),
                ("Heavy-lift reference", "heavy"),
                ("Two-stage small launcher", "small"),
            ]
        )
        load = QPushButton("Load")
        load.clicked.connect(self._load_preset)
        add = QPushButton("Add stage")
        add.clicked.connect(lambda: self._add_row(["Stage", 5.0, 50.0, 1, 500.0, 0.0, 330.0]))
        remove = QPushButton("Remove last")
        remove.clicked.connect(
            lambda: self.table.setRowCount(max(1, self.table.rowCount() - 1)) or self._compute()
        )
        self.payload = NumberField(1000.0, 0.0, 200_000.0, Quantity.MASS, decimals=1, step=100.0)
        top.addWidget(self.preset)
        top.addWidget(load)
        top.addSpacing(18)
        top.addWidget(add)
        top.addWidget(remove)
        top.addStretch()
        top.addWidget(caption("Payload"))
        top.addWidget(self.payload)
        layout.addLayout(top)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMaximumHeight(170)
        self.table.itemChanged.connect(lambda _item: self._compute())
        layout.addWidget(self.table)

        self.total = Metric("Ideal delta-v", Quantity.VELOCITY)
        self.liftoff = Metric("Liftoff mass", Quantity.MASS)
        self.tw = Metric("Liftoff thrust / weight", unit="", digits=3)
        layout.addWidget(metric_row(self.total, self.liftoff, self.tw))
        self.breakdown = ResultTable(
            ["Stage", "Delta-v  [m/s]", "Burn time  [s]", "T/W at ignition", "Mass ratio"]
        )
        self.breakdown.setMaximumHeight(150)
        layout.addWidget(self.breakdown)
        self.plot = PlotCanvas(self._render, None, "")
        layout.addWidget(self.plot, 1)
        layout.addWidget(
            hint("Tsiolkovsky delta-v with vacuum Isp; gravity and drag losses are not included.")
        )
        self.payload.changed_si.connect(lambda _v: self._compute())
        self._vehicle = None
        self._load_preset()

    def _add_row(self, values) -> None:
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value) if not isinstance(value, float) else f"{value:g}")
            if col:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self._compute()

    def _load_preset(self) -> None:
        kind = self.preset.value()
        if kind == "small":
            vehicle = create_custom_vehicle(
                [
                    {
                        "name": "Stage 1",
                        "dry_mass": 2000.0,
                        "propellant_mass": 20_000.0,
                        "thrust_sl_per_engine": 300e3,
                        "thrust_vac_per_engine": 330e3,
                        "isp_sl": 270.0,
                        "isp_vac": 295.0,
                    },
                    {
                        "name": "Stage 2",
                        "dry_mass": 500.0,
                        "propellant_mass": 5000.0,
                        "thrust_vac_per_engine": 50e3,
                        "isp_vac": 320.0,
                    },
                ],
                payload_mass=250.0,
            )
        else:
            vehicle = (
                create_medium_lift_reference()
                if kind == "medium"
                else create_heavy_lift_reference()
            )
            vehicle.payload_mass = 10_000.0 if kind == "medium" else 40_000.0
        self.table.setRowCount(0)
        for stage in vehicle.stages:
            e = stage.engine
            self._add_row(
                [
                    stage.name,
                    stage.dry_mass / 1000.0,
                    stage.propellant_mass / 1000.0,
                    e.num_engines,
                    e.thrust_vac_per_engine / 1000.0,
                    e.isp_sl,
                    e.isp_vac,
                ]
            )
        self.payload.set_value_si(vehicle.payload_mass)
        self._compute()

    def _configs(self) -> list[dict]:
        configs = []
        for row in range(self.table.rowCount()):

            def cell(c: int, row: int = row) -> str:
                item = self.table.item(row, c)
                return item.text() if item else "0"

            isp_vac = float(cell(6))
            isp_sl = float(cell(5))
            thrust_vac = float(cell(4)) * 1000.0
            configs.append(
                {
                    "name": cell(0),
                    "dry_mass": float(cell(1)) * 1000.0,
                    "propellant_mass": float(cell(2)) * 1000.0,
                    "num_engines": int(float(cell(3))),
                    "thrust_vac_per_engine": thrust_vac,
                    "thrust_sl_per_engine": thrust_vac * isp_sl / isp_vac if isp_vac else 0.0,
                    "isp_sl": isp_sl,
                    "isp_vac": isp_vac,
                }
            )
        return configs

    def _compute(self) -> None:
        try:
            vehicle = create_custom_vehicle(self._configs(), payload_mass=self.payload.value_si())
        except (ValueError, KeyError, ZeroDivisionError):
            self.total.set_value(None)
            return
        self._vehicle = vehicle
        breakdown = vehicle.get_stage_delta_v_breakdown()
        masses = []
        above = vehicle.payload_mass
        for stage in reversed(vehicle.stages):
            masses.append(above + stage.dry_mass + stage.propellant_mass)
            above = masses[-1]
        masses = masses[::-1]
        rows = []
        for i, (stage, (name, dv)) in enumerate(zip(vehicle.stages, breakdown, strict=True)):
            e = stage.engine
            thrust = e.total_thrust_sl if i == 0 and e.total_thrust_sl > 0 else e.total_thrust_vac
            burn = stage.propellant_mass / e.get_mass_flow_rate()
            ratio = masses[i] / (masses[i] - stage.propellant_mass)
            rows.append([name, dv, burn, thrust / (masses[i] * G0), ratio])
        self.breakdown.set_rows(rows)
        self.total.set_value(vehicle.get_total_delta_v())
        self.liftoff.set_value(masses[0])
        self.tw.set_value(rows[0][3] if rows else None)
        self._breakdown = breakdown
        self.plot.refresh()

    def _render(self, figure) -> None:
        p = theme.current()
        ax = figure.add_subplot(111)
        left = 0.0
        for i, (name, dv) in enumerate(self._breakdown):
            color = p.signal if i == 0 else p.series[min(i, len(p.series) - 1)]
            ax.barh([0], [dv], left=left, color=color, height=0.5)
            ax.text(
                left + dv / 2,
                0,
                f"{name}\n{dv:,.0f} m/s",
                ha="center",
                va="center",
                fontsize=8,
                color=p.bg,
            )
            left += dv
        for target, label in (
            (7800.0, "LEO orbital velocity"),
            (9400.0, "LEO incl. typical losses"),
        ):
            ax.axvline(target, color=p.line_strong, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.text(target, 0.38, f" {label}", fontsize=8, color=p.muted)
        ax.set_axisbelow(True)
        ax.set_yticks([])
        ax.set_xlabel("Cumulative ideal delta-v  [m/s]")
        ax.set_xlim(0, max(left, 9400.0) * 1.08)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
