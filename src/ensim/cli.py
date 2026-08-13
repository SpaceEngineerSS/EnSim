"""Command-line entry points for EnSim."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

from ensim import __version__


def run_gui() -> NoReturn:
    """Launch the desktop application."""
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

    try:
        import numpy as np
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QIcon
        from PyQt6.QtWidgets import QApplication

        from ensim.core.propulsion import calculate_c_star
        from ensim.core.thermodynamics import cp_over_r
        from ensim.ui.splash_screen import EnSimSplashScreen
        from ensim.ui.windows.main_window import MainWindow
    except ImportError as exc:
        raise SystemExit(
            "The desktop dependencies are unavailable. Install EnSim with its GUI "
            f"dependencies and retry. Original import error: {exc}"
        ) from exc

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("EnSim")
    app.setOrganizationName("EnSim")
    app.setApplicationVersion(__version__)

    icon_path = Path(__file__).parent / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    splash = EnSimSplashScreen()
    splash.show()
    app.processEvents()

    splash.showMessage("Loading thermodynamic models...")
    cp_over_r(1000.0, np.array([1.0, 1e-5, 1e-8, 1e-11, 1e-14, 1e4, 1.0]))
    splash.showMessage("Loading propulsion models...")
    calculate_c_star(1.2, 500.0, 3000.0)
    splash.showMessage("Starting EnSim...")
    app.processEvents()

    window = MainWindow()
    window.show()
    splash.finish(window)
    raise SystemExit(app.exec())


def run_validation_test() -> None:
    """Run a deterministic installation and physics smoke test."""
    from ensim.core.chemistry import CombustionProblem
    from ensim.core.propulsion import NozzleConditions, calculate_performance
    from ensim.core.thermodynamics import calculate_cp
    from ensim.utils.nasa_parser import load_default_database

    database = load_default_database()
    h2o_cp = calculate_cp(1000.0, database["H2O"])
    if not 40.0 <= h2o_cp <= 43.0:
        raise RuntimeError(f"H2O heat-capacity check failed: {h2o_cp:.6g} J/(mol K)")

    problem = CombustionProblem(database)
    problem.add_fuel("H2", moles=2.0, temperature=298.15)
    problem.add_oxidizer("O2", moles=1.0, temperature=298.15)
    equilibrium = problem.solve(pressure=1_013_250.0)
    if not equilibrium.converged:
        raise RuntimeError("Chemical-equilibrium solver did not converge")

    nozzle = NozzleConditions(
        area_ratio=40.0,
        chamber_pressure=1_013_250.0,
        ambient_pressure=0.0,
    )
    performance = calculate_performance(
        T_chamber=equilibrium.temperature,
        P_chamber=nozzle.chamber_pressure,
        gamma=equilibrium.gamma,
        mean_molecular_weight=equilibrium.mean_molecular_weight,
        nozzle=nozzle,
    )
    if not 2500.0 < equilibrium.temperature < 4500.0 or performance.isp <= 0.0:
        raise RuntimeError("The coupled equilibrium/nozzle result is outside physical bounds")

    print(f"EnSim {__version__} validation smoke test passed")
    print(f"  Packaged species: {len(database)}")
    print(f"  H2O Cp at 1000 K: {h2o_cp:.3f} J/(mol K)")
    print(f"  H2/O2 chamber temperature: {equilibrium.temperature:.1f} K")
    print(f"  Frozen-flow vacuum Isp (area ratio 40): {performance.isp:.1f} s")
    print("Run `python -m pytest` for the complete verification suite.")


def main() -> None:
    """Parse command-line arguments and dispatch the requested mode."""
    parser = argparse.ArgumentParser(prog="ensim", description="EnSim engineering simulator")
    parser.add_argument(
        "--test",
        action="store_true",
        help="run a deterministic installation and physics smoke test",
    )
    parser.add_argument("--version", action="version", version=f"EnSim {__version__}")
    args = parser.parse_args()
    run_validation_test() if args.test else run_gui()


if __name__ == "__main__":
    main()
