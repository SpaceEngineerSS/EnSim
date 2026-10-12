"""End-to-end checks for the desktop engine-analysis data path."""

import pytest
from PyQt6.QtWidgets import QApplication

from ensim.core.presets import ENGINE_PRESETS
from ensim.ui.windows.main_window import MainWindow
from ensim.ui.workers import CalculationWorker, SimulationParams


def test_engine_worker_uses_mass_ratio_and_selected_operating_pressure():
    results = []
    errors = []
    worker = CalculationWorker(
        SimulationParams(
            fuel="H2",
            oxidizer="O2",
            fuel_moles=1.0,
            oxidizer_moles=1.0,
            of_ratio_mass=6.0,
            chamber_pressure_bar=68.0,
            expansion_ratio=20.0,
            ambient_pressure_bar=1.01325,
            throat_area_cm2=100.0,
        )
    )
    worker.finished.connect(results.append)
    worker.error.connect(errors.append)
    worker.run()

    assert not errors
    result = results[0]
    assert result.converged
    assert result.operating_ambient_pressure == pytest.approx(101_325.0)
    assert result.isp_operating == pytest.approx(result.isp_sea_level)
    assert result.thrust_operating < result.thrust_vacuum
    assert 3000.0 < result.temperature < 4500.0
    assert 300.0 < result.isp_vacuum < 500.0


def test_engine_worker_rejects_unknown_species_before_solving():
    errors = []
    worker = CalculationWorker(
        SimulationParams(fuel="NOT_A_SPECIES", oxidizer="O2", of_ratio_mass=2.0)
    )
    worker.error.connect(errors.append)
    worker.run()
    assert errors == ["Fuel species is unavailable: NOT_A_SPECIES"]


def test_completed_engine_run_selects_results_workspace():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    results = []
    worker = CalculationWorker(SimulationParams(fuel="H2", oxidizer="O2", of_ratio_mass=6.0))
    worker.finished.connect(results.append)
    worker.run()
    window._last_params = worker.params
    window._on_simulation_complete(results[0])
    try:
        assert window.tabs.currentIndex() == 1
    finally:
        window.close()
        app.processEvents()


def test_every_reference_case_maps_to_runnable_input_species():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        for preset in ENGINE_PRESETS.values():
            window._load_preset(preset)
            assert window.input_panel.fuel_combo.currentData() == preset.fuel
            assert window.input_panel.oxidizer_combo.currentData() == preset.oxidizer
            assert window.input_panel.of_ratio_spin.value() == pytest.approx(preset.of_ratio)
    finally:
        window.close()
        app.processEvents()
