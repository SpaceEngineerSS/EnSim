"""Command-line interface.

ensim                     open the desktop application (needs ensim[gui])
ensim engine ...          one engine operating point
ensim run project.ensim   evaluate a saved project, optionally fly it
ensim check               installation self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ensim import __version__

GUI_HINT = 'The desktop application needs the GUI extra: pip install "ensim[gui]"'


def _rows(rows: Sequence[tuple[str, str]]) -> str:
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {label:<{width}}  {value}" for label, value in rows)


def _format_engine(result: Any) -> str:
    inputs = result.inputs
    lines = [
        f"{inputs.fuel}/{inputs.oxidizer}  O/F {inputs.of_ratio:g}  "
        f"Pc {inputs.chamber_pressure / 1e5:g} bar  eps {inputs.expansion_ratio:g}",
        "",
        "Chamber",
        _rows(
            [
                ("temperature", f"{result.chamber_temperature:8.1f} K"),
                ("gamma (frozen)", f"{result.gamma:8.4f}"),
                ("molecular weight", f"{result.molecular_weight:8.3f} g/mol"),
                ("c*", f"{result.vacuum.c_star:8.1f} m/s"),
            ]
        ),
        "",
        "Nozzle",
    ]
    rows = [("Isp vacuum", f"{result.vacuum.isp:8.1f} s")]
    if result.sea_level is not None:
        rows.append(("Isp sea level", f"{result.sea_level.isp:8.1f} s"))
    else:
        rows.append(("Isp sea level", "     n/a  (flow cannot stay choked at 1 atm)"))
    rows += [
        ("exit Mach", f"{result.vacuum.exit_mach:8.3f}"),
        ("exit pressure", f"{result.vacuum.exit_pressure / 1e3:8.2f} kPa"),
        ("Cf vacuum", f"{result.vacuum.c_f:8.4f}"),
    ]
    if inputs.throat_area is not None:
        rows += [
            ("thrust vacuum", f"{result.vacuum.thrust / 1e3:8.2f} kN"),
            ("mass flow", f"{result.vacuum.mass_flow_rate:8.3f} kg/s"),
        ]
    lines.append(_rows(rows))
    lines += ["", "Composition (mole fraction)"]
    lines.append(
        _rows(
            [
                (name, f"{fraction:8.5f}")
                for name, fraction in result.composition.items()
                if fraction >= 1e-3
            ]
        )
    )
    return "\n".join(lines)


def _write_json(data: dict[str, Any], target: str | None) -> None:
    text = json.dumps(data, indent=2)
    if target in (None, "-"):
        print(text)
    else:
        Path(target).write_text(text + "\n", encoding="utf-8")


def cmd_engine(args: argparse.Namespace) -> int:
    from ensim.core.engine import EngineInputs, analyze_engine

    inputs = EngineInputs(
        fuel=args.fuel,
        oxidizer=args.oxidizer,
        of_ratio=args.of,
        chamber_pressure=args.pc * 1e5,
        expansion_ratio=args.eps,
        throat_area=args.throat_area * 1e-4 if args.throat_area else None,
        ambient_pressure=args.pa * 1e5,
        eta_cstar=args.eta_cstar,
        eta_cf=args.eta_cf,
        nozzle_half_angle_deg=args.half_angle,
    )
    result = analyze_engine(inputs)
    if args.json is not None:
        _write_json(result.to_dict(), args.json)
    else:
        print(_format_engine(result))
    return 0


def _ambient_from_label(label: str) -> float:
    """Project files store the ambient setting as text, e.g. 'Sea Level (1.01 bar)'."""
    if "(" in label and "bar" in label:
        try:
            return float(label.split("(")[1].split("bar")[0]) * 1e5
        except ValueError:
            pass
    return 0.0


def cmd_run(args: argparse.Namespace) -> int:
    from ensim.core.engine import EngineInputs, analyze_engine
    from ensim.core.project_manager import ProjectManager

    project = ProjectManager()
    data = project.load(args.project)
    engine = data.engine
    inputs = EngineInputs(
        fuel=engine.fuel,
        oxidizer=engine.oxidizer,
        of_ratio=engine.of_ratio,
        chamber_pressure=engine.chamber_pressure_bar * 1e5,
        expansion_ratio=engine.expansion_ratio,
        throat_area=engine.throat_area_cm2 * 1e-4,
        ambient_pressure=_ambient_from_label(engine.ambient),
        eta_cstar=engine.eta_cstar,
        eta_cf=engine.eta_cf,
        nozzle_half_angle_deg=engine.alpha_deg,
    )
    result = analyze_engine(inputs)
    output: dict[str, Any] = {"project": str(args.project), "engine": result.to_dict()}
    text = [_format_engine(result)]

    if args.flight:
        from ensim.core.flight_6dof import simulate_flight_6dof

        rocket = project.build_rocket()
        recovery = data.recovery
        environment = data.environment
        mdot = result.vacuum.mass_flow_rate
        burn_time = rocket.engine.propellant_mass / mdot
        flight = simulate_flight_6dof(
            rocket,
            thrust_vac=result.vacuum.thrust,
            isp_vac=result.vacuum.isp,
            burn_time=burn_time,
            exit_area=inputs.throat_area * inputs.expansion_ratio,
            launch_angle_deg=environment.launch_angle,
            rail_length=environment.rail_length,
            wind_speed=environment.wind_speed,
            wind_direction_deg=environment.wind_direction,
            chute_diameter=recovery.main_diameter,
            chute_cd=recovery.main_cd,
            deploy_at_apogee=not recovery.dual_deploy,
            main_deploy_altitude_m=recovery.main_deploy_altitude,
            drogue_diameter=recovery.drogue_diameter if recovery.dual_deploy else 0.0,
            drogue_cd=recovery.drogue_cd,
            max_time=args.max_time,
            nozzle_exit_pressure=result.vacuum.exit_pressure,
        )
        summary = {
            "apogee_m": flight.apogee_altitude,
            "apogee_time_s": flight.apogee_time,
            "max_velocity_m_s": flight.max_velocity,
            "max_mach": flight.max_mach,
            "max_acceleration_g": flight.max_acceleration,
            "max_dynamic_pressure_pa": flight.max_q,
            "burnout_time_s": flight.burnout_time,
            "flight_time_s": flight.flight_time,
            "landed": flight.ground_impact,
            "landing_east_m": flight.landing_x,
            "landing_north_m": flight.landing_y,
        }
        output["flight"] = summary
        rows = [
            ("apogee", f"{flight.apogee_altitude:10.1f} m at {flight.apogee_time:.1f} s"),
            ("max velocity", f"{flight.max_velocity:10.1f} m/s (Mach {flight.max_mach:.2f})"),
            ("max acceleration", f"{flight.max_acceleration:10.2f} g"),
            ("max q", f"{flight.max_q / 1e3:10.2f} kPa"),
            ("burnout", f"{flight.burnout_time:10.2f} s"),
        ]
        if flight.ground_impact:
            rows.append(
                (
                    "landing",
                    f"{flight.landing_x:10.1f} m E, {flight.landing_y:.1f} m N "
                    f"after {flight.flight_time:.1f} s",
                )
            )
        else:
            rows.append(("landing", f"   still airborne at {flight.flight_time:.0f} s"))
        text += ["", f"Flight ({rocket.name}, {rocket.wet_mass:.2f} kg)", _rows(rows)]

    if args.json is not None:
        _write_json(output, args.json)
    else:
        print("\n".join(text))
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    """Short end-to-end check of an installation."""
    from ensim.core.engine import EngineInputs, analyze_engine
    from ensim.core.thermodynamics import calculate_cp
    from ensim.utils.nasa_parser import load_default_database

    database = load_default_database()
    cp = calculate_cp(1000.0, database["H2O"])
    if not 40.0 <= cp <= 43.0:
        raise RuntimeError(
            f"H2O heat capacity at 1000 K is {cp:.4g} J/(mol K), expected about 41.3"
        )
    result = analyze_engine(
        EngineInputs("H2", "O2", of_ratio=7.9367, chamber_pressure=1.01325e6, expansion_ratio=40.0)
    )
    if not 3000.0 < result.chamber_temperature < 3700.0 or result.vacuum.isp <= 0.0:
        raise RuntimeError("Equilibrium and nozzle results are outside physical bounds")
    print(f"EnSim {__version__}")
    print(
        _rows(
            [
                ("species", str(len(database))),
                ("H2O Cp(1000 K)", f"{cp:.3f} J/(mol K)"),
                ("H2/O2 Tc, 10 bar", f"{result.chamber_temperature:.1f} K"),
                ("Isp vacuum, eps 40", f"{result.vacuum.isp:.1f} s"),
            ]
        )
    )
    print("OK")
    return 0


def cmd_gui(_: argparse.Namespace | None = None) -> int:
    try:
        from ensim.ui.app import run
    except ImportError as error:
        print(f"{GUI_HINT}\n({error})", file=sys.stderr)
        return 2
    return run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ensim",
        description="Liquid rocket engine analysis and 6-DOF flight simulation.",
    )
    parser.add_argument("--version", action="version", version=f"EnSim {__version__}")
    parser.add_argument("--test", action="store_true", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", metavar="command")

    commands.add_parser("gui", help="open the desktop application")

    engine = commands.add_parser("engine", help="analyse one engine operating point")
    engine.add_argument("--fuel", required=True, help="fuel species, e.g. H2, CH4, RP1")
    engine.add_argument("--oxidizer", default="O2", help="oxidizer species (default O2)")
    engine.add_argument("--of", type=float, required=True, help="oxidizer/fuel mass ratio")
    engine.add_argument("--pc", type=float, required=True, help="chamber pressure, bar")
    engine.add_argument("--eps", type=float, required=True, help="nozzle area ratio Ae/At")
    engine.add_argument("--pa", type=float, default=0.0, help="ambient pressure, bar (default 0)")
    engine.add_argument("--throat-area", type=float, help="throat area, cm² (enables thrust)")
    engine.add_argument("--eta-cstar", type=float, default=1.0, help="c* efficiency")
    engine.add_argument("--eta-cf", type=float, default=1.0, help="thrust coefficient efficiency")
    engine.add_argument("--half-angle", type=float, default=15.0, help="nozzle half-angle, deg")
    engine.add_argument(
        "--json", nargs="?", const="-", metavar="FILE", help="write JSON (to stdout without FILE)"
    )

    run = commands.add_parser("run", help="evaluate a saved .ensim project")
    run.add_argument("project", type=Path)
    run.add_argument("--flight", action="store_true", help="also fly the project's rocket")
    run.add_argument("--max-time", type=float, default=600.0, help="flight time limit, s")
    run.add_argument("--json", nargs="?", const="-", metavar="FILE", help="write JSON")

    commands.add_parser("check", help="run a short installation self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"engine": cmd_engine, "run": cmd_run, "check": cmd_check, "gui": cmd_gui}
    if args.test:
        args.command = "check"
    if args.command is None:
        return cmd_gui()
    try:
        return handlers[args.command](args)
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"ensim {args.command}: {message}", file=sys.stderr)
        return 1
    except Exception as error:  # ProjectFileError, ConvergenceError and friends
        if type(error).__module__.startswith("ensim."):
            print(f"ensim {args.command}: {error}", file=sys.stderr)
            return 1
        raise


def gui_main() -> int:
    """Entry point for the ``ensim-gui`` launcher."""
    return cmd_gui()


if __name__ == "__main__":
    raise SystemExit(main())
