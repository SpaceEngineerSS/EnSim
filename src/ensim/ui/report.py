"""Reports and machine-readable exports of the current workspace."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ensim import __version__

from .workspace import Workspace


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and type(value).__module__.startswith("ensim"):
        return value.value
    return value


def collect(workspace: Workspace) -> dict:
    """Inputs and every result computed in this session, as plain data."""
    out: dict = {
        "ensim_version": __version__,
        "written": datetime.now().isoformat(timespec="seconds"),
        "project": workspace.project.to_dict(),
    }
    if workspace.result is not None:
        out["engine"] = workspace.result.to_dict()
    flight = workspace.results.get("flight")
    if flight is not None:
        out["flight"] = {
            "apogee_m": flight.apogee_altitude,
            "apogee_time_s": flight.apogee_time,
            "burnout_time_s": flight.burnout_time,
            "max_velocity_m_s": flight.max_velocity,
            "max_mach": flight.max_mach,
            "max_acceleration_g": flight.max_acceleration,
            "max_dynamic_pressure_pa": flight.max_q,
            "flight_time_s": flight.flight_time,
            "landing_east_m": flight.landing_x,
            "landing_north_m": flight.landing_y,
        }
    dispersion = workspace.results.get("dispersion")
    if dispersion is not None:
        out["dispersion"] = {
            "runs": dispersion.num_simulations,
            "landed": dispersion.n_landed,
            "cep_m": dispersion.cep_radius,
            "ellipse_semi_major_m": dispersion.ellipse_semi_major,
            "ellipse_semi_minor_m": dispersion.ellipse_semi_minor,
            "apogee_mean_m": dispersion.apogee_mean,
            "apogee_std_m": dispersion.apogee_std,
        }
    cooling = workspace.results.get("cooling")
    if cooling:
        out["cooling"] = [_plain(asdict(station)) for station in cooling]
    return _plain(out)


def write_json(workspace: Workspace, path: Path) -> None:
    path.write_text(json.dumps(collect(workspace), indent=2) + "\n", encoding="utf-8")


def write_report(workspace: Workspace, path: Path) -> None:
    """Markdown summary for a design review or a lab notebook."""
    data = collect(workspace)
    r = workspace.result
    i = r.inputs
    lines = [
        f"# {workspace.project.project_name}",
        "",
        f"EnSim {__version__}, {data['written'].replace('T', ' ')}",
        "",
        "## Engine",
        "",
        "| Input | Value |",
        "|---|---|",
        f"| Propellants | {i.fuel} / {i.oxidizer} |",
        f"| O/F (mass) | {i.of_ratio:g} |",
        f"| Chamber pressure | {i.chamber_pressure / 1e5:g} bar |",
        f"| Area ratio | {i.expansion_ratio:g} |",
        f"| Throat area | {i.throat_area * 1e4:g} cm² |",
        f"| c* and Cf efficiency | {i.eta_cstar:g}, {i.eta_cf:g} |",
        f"| Divergence half-angle | {i.nozzle_half_angle_deg:g} deg |",
        "",
        "| Result | Vacuum | Sea level |",
        "|---|---|---|",
    ]
    sl = r.sea_level
    for label, getter, scale, unit in (
        ("Isp", lambda p: p.isp, 1.0, "s"),
        ("Thrust", lambda p: p.thrust, 1e-3, "kN"),
        ("Thrust coefficient", lambda p: p.c_f, 1.0, ""),
        ("Exit pressure", lambda p: p.exit_pressure, 1e-3, "kPa"),
    ):
        vacuum = f"{getter(r.vacuum) * scale:.4g} {unit}".strip()
        sea = f"{getter(sl) * scale:.4g} {unit}".strip() if sl else "n/a"
        lines.append(f"| {label} | {vacuum} | {sea} |")
    lines += [
        "",
        f"Chamber temperature {r.chamber_temperature:.1f} K, frozen gamma {r.gamma:.4f}, "
        f"molar mass {r.molecular_weight:.3f} g/mol, c* {r.vacuum.c_star:.1f} m/s, "
        f"mass flow {r.vacuum.mass_flow_rate:.4g} kg/s.",
        "",
        "Chamber composition (mole fraction):",
        "",
    ]
    lines += [
        f"- {name}: {fraction:.5f}" for name, fraction in r.composition.items() if fraction >= 1e-4
    ]
    if "flight" in data:
        f = data["flight"]
        lines += [
            "",
            "## Flight",
            "",
            f"Apogee {f['apogee_m']:.1f} m at {f['apogee_time_s']:.1f} s; burnout at {f['burnout_time_s']:.2f} s; "
            f"maximum speed {f['max_velocity_m_s']:.1f} m/s (Mach {f['max_mach']:.2f}); "
            f"maximum acceleration {f['max_acceleration_g']:.2f} g; maximum q {f['max_dynamic_pressure_pa'] / 1e3:.2f} kPa.",
        ]
        if f["landing_east_m"] is not None:
            lines.append(
                f"Landed {f['landing_east_m']:.1f} m east and {f['landing_north_m']:.1f} m north of the pad "
                f"after {f['flight_time_s']:.1f} s."
            )
    if "dispersion" in data:
        d = data["dispersion"]
        lines += [
            "",
            "## Dispersion",
            "",
            f"{d['runs']} runs, {d['landed']} landed. CEP {d['cep_m']:.1f} m; 99.7 % ellipse "
            f"{d['ellipse_semi_major_m']:.1f} x {d['ellipse_semi_minor_m']:.1f} m; apogee "
            f"{d['apogee_mean_m']:.1f} ± {d['apogee_std_m']:.1f} m.",
        ]
    lines += [
        "",
        "## Model",
        "",
        "Chamber equilibrium by Gibbs energy minimisation with NASA Glenn data; propellants enter "
        "in their storage state. The nozzle expands the chamber composition frozen, with "
        "temperature-dependent heat capacity. Flight uses a rigid-body 6-DOF model with Barrowman "
        "aerodynamics and the US Standard Atmosphere 1976. See the model limitations page of the "
        "documentation before using these numbers for design decisions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
