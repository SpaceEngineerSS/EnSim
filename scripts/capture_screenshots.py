"""Capture the current desktop interface for the documentation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PyQt6.QtWidgets import QApplication, QScrollArea, QTabWidget  # noqa: E402

from ensim.ui.windows.main_window import MainWindow  # noqa: E402
from ensim.ui.workers import CalculationWorker, SimulationParams  # noqa: E402


def process_events(app: QApplication, count: int = 4) -> None:
    for _ in range(count):
        app.processEvents()


def capture(window: MainWindow, app: QApplication, name: str) -> None:
    process_events(app)
    path = PROJECT_ROOT / "docs" / f"{name}.png"
    pixmap = window.grab()
    if not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"Could not save {path}")
    print(f"{path.name}: {path.stat().st_size / 1024:.1f} KiB")


def populate_engine_result(window: MainWindow) -> None:
    params = SimulationParams(
        fuel="H2",
        oxidizer="O2",
        of_ratio_mass=6.0,
        chamber_pressure_bar=68.0,
        expansion_ratio=40.0,
        ambient_pressure_bar=0.0,
        throat_area_cm2=100.0,
    )
    results = []
    errors = []
    worker = CalculationWorker(params)
    worker.finished.connect(results.append)
    worker.error.connect(errors.append)
    worker.run()
    if errors or not results:
        raise RuntimeError(errors[0] if errors else "Engine calculation returned no result")
    window._last_params = params
    window._on_simulation_complete(results[0])


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("EnSim")
    window = MainWindow()
    window._first_run = False
    window.resize(1440, 920)
    window.show()
    process_events(app)
    populate_engine_result(window)

    window.tabs.setCurrentIndex(0)
    capture(window, app, "screenshot_main")

    window.tabs.setCurrentIndex(1)
    results_tabs = window.tabs.widget(1).findChild(QTabWidget, "subTabs")
    if results_tabs is None:
        raise RuntimeError("Results sub-tabs were not found")
    results_tabs.setCurrentIndex(0)
    capture(window, app, "screenshot_graphs")
    results_tabs.setCurrentIndex(1)
    if window.view3d_widget._plotter is not None:
        window.view3d_widget._plotter.render()
        path = PROJECT_ROOT / "docs" / "screenshot_3d.png"
        window.view3d_widget._plotter.screenshot(str(path))
        print(f"{path.name}: {path.stat().st_size / 1024:.1f} KiB")

    window.tabs.setCurrentIndex(2)
    window.engine_tabs.setCurrentIndex(0)
    window.cooling_widget._run_analysis()
    if not window.cooling_widget._worker.wait(60_000):
        raise RuntimeError("Cooling calculation timed out")
    process_events(app)
    capture(window, app, "screenshot_engine")

    window.tabs.setCurrentIndex(3)
    vehicle_tabs = window.tabs.widget(3).findChild(QTabWidget, "subTabs")
    if vehicle_tabs is None:
        raise RuntimeError("Vehicle sub-tabs were not found")
    vehicle_tabs.setCurrentIndex(1)
    vehicle_scroll = window.vehicle_widget.findChild(QScrollArea)
    if vehicle_scroll is not None:
        vehicle_scroll.ensureWidgetVisible(window.vehicle_widget.axial_cd_spin)
    window.vehicle_widget._update_diagram()
    capture(window, app, "screenshot_vehicle")

    window.tabs.setCurrentIndex(4)
    window.advanced_widget.moc_tab._generate()
    capture(window, app, "screenshot_advanced")

    window.close()
    process_events(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
