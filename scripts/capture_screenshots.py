"""Capture the desktop interface for the documentation.

    python scripts/capture_screenshots.py [--theme console|paper] [--scale 1.5]

Opens the bundled examples in an offscreen window with private settings,
runs each view and writes docs/assets/screens/<name>.png. Tasks run
synchronously, so every capture shows a finished result.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "screens"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theme", choices=("console", "paper"), default="console")
    parser.add_argument("--scale", type=float, default=1.5, help="device pixel ratio")
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_SCALE_FACTOR"] = f"{args.scale:g}"
    os.environ["ENSIM_DISABLE_3D"] = "1"

    from PyQt6.QtWidgets import QApplication, QTabWidget

    from ensim.ui import theme
    from ensim.ui.app import _use_c_locale
    from ensim.ui.settings import Settings
    from ensim.ui.tasks import TaskRunner
    from ensim.ui.windows.main_window import MainWindow, examples_dir

    app = QApplication(sys.argv[:1])
    _use_c_locale()
    theme.load_fonts()
    theme.apply(app, args.theme)

    settings_dir = tempfile.TemporaryDirectory()
    window = MainWindow(
        Settings(Path(settings_dir.name) / "settings.ini"), TaskRunner(synchronous=True)
    )
    window.set_theme(args.theme)
    window.resize(1366, 800)
    window.show()

    def open_example(name: str) -> None:
        if not window.open_path(examples_dir() / name, remember=False, confirm=False):
            raise RuntimeError(f"could not open the example {name}")

    open_example("kerolox_sounding_rocket.ensim")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def settle(seconds: float = 0.0) -> None:
        end = time.perf_counter() + seconds
        while True:
            app.processEvents()
            if time.perf_counter() >= end:
                break

    def tab(section, title: str) -> None:
        for widget in section.findChildren(QTabWidget):
            for index in range(widget.count()):
                if widget.tabText(index) == title:
                    widget.setCurrentIndex(index)
                    settle()
                    return
        raise LookupError(f"no tab {title!r} in {section.title}")

    def capture(name: str) -> None:
        settle(0.6)  # let the number and fade animations finish
        path = OUTPUT / f"{name}.png"
        if not window.grab().save(str(path), "PNG"):
            raise RuntimeError(f"could not write {path}")
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KiB")

    engine, vehicle, flight, analysis = window.sections[:4]

    window.show_section(0)
    window.run_current()
    tab(engine, "Performance")
    capture("engine_performance")

    window.show_section(1)
    tab(vehicle, "Design")
    capture("vehicle_design")
    tab(vehicle, "Stages")
    capture("vehicle_stages")

    window.show_section(2)
    window.run_current()
    tab(flight, "Trajectory")
    capture("flight_trajectory")
    tab(flight, "Dispersion")
    flight.dispersion.runs.setValue(200)
    flight.dispersion.workers.setValue(1)
    flight.dispersion.run()
    capture("flight_dispersion")

    window.show_section(3)
    tab(analysis, "Sweep")
    analysis.sweep.run()
    capture("analysis_sweep")
    tab(analysis, "MOC contour")
    analysis.moc._from_engine()
    analysis.moc.run()
    capture("analysis_moc")

    # Regenerative cooling only makes sense for a larger engine than the
    # sounding rocket's 0.4 kN: use the 100 bar methalox example.
    open_example("methalox_sealevel.ensim")
    window.show_section(0)
    window.run_current()
    tab(engine, "Cooling")
    cooling = engine.cooling.widget
    cooling.width.set_value_si(1.0e-3)
    cooling.height.set_value_si(2.5e-3)
    cooling.inlet_p_ratio.set_value_si(2.0)  # keeps the methane above its critical pressure
    cooling.engine_changed()
    cooling.run()
    capture("engine_cooling")

    window.workspace.project._modified = False
    window.close()
    settings_dir.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
