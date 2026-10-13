"""End-to-end checks of the desktop application, headless."""

import json

import numpy as np
import pytest
from PyQt6.QtWidgets import QTabWidget


def test_every_section_and_tab_builds(example, qapp):
    win = example
    assert [s.title for s in win.sections] == ["Engine", "Vehicle", "Flight", "Analysis", "Log"]
    for index, section in enumerate(win.sections):
        win.show_section(index)
        qapp.processEvents()
        assert section.built
        for tabs in section.findChildren(QTabWidget):
            for tab in range(tabs.count()):
                tabs.setCurrentIndex(tab)
                qapp.processEvents()


def test_engine_run_fills_metrics_and_header(example, qapp):
    win = example
    win.show_section(0)
    win.run_current()
    qapp.processEvents()
    result = win.workspace.result
    assert result is not None
    # NASA-CEA-validated frozen expansion for the example: about 257 s in vacuum
    assert result.vacuum.isp == pytest.approx(257.3, abs=0.5)
    assert "Converged" in win.header._state.text()
    assert len(win.workspace.run.profile["mach"]) == len(win.workspace.run.contour.x)


def test_flight_runs_the_engine_first_when_needed(example, qapp):
    win = example
    win.show_section(2)
    win.run_current()
    qapp.processEvents()
    flight = win.sections[2]._result
    assert flight is not None and flight.ground_impact
    assert 4000.0 < flight.apogee_altitude < 9000.0


def test_editing_an_input_marks_the_project_modified(example, qapp):
    win = example
    win.show_section(0)
    engine = win.sections[0]
    engine.pc.set_value_si(30e5)
    engine.pc.changed_si.emit(30e5)
    assert win.workspace.data.engine.chamber_pressure_bar == pytest.approx(30.0)
    assert win.workspace.project.is_modified
    assert win.windowTitle().startswith("kerolox_sounding_rocket *")


def test_units_switch_keeps_si_values(example, qapp):
    from ensim.utils.units import UnitSystem

    win = example
    win.show_section(0)
    field = win.sections[0].pc
    before = field.value_si()
    win.set_units(UnitSystem.IMPERIAL)
    assert field.suffix().strip() == "psi"
    assert field.value() == pytest.approx(before / 6894.757, rel=1e-3)
    assert field.value_si() == pytest.approx(before)
    win.set_units(UnitSystem.SI)


def test_theme_switch_and_analysis_views(example, qapp):
    win = example
    win.show_section(0)
    win.run_current()
    win.show_section(3)
    analysis = win.sections[3]
    analysis.sweep.run()
    assert analysis.sweep._result is not None
    assert np.all(np.isfinite(analysis.sweep._result.isp_vacuum))
    analysis.moc._from_engine()
    analysis.moc.run()
    assert analysis.moc._contour is not None
    analysis.compare.pin()
    analysis.compare.pin()
    assert analysis.compare.table.rowCount() == 2
    win.set_theme("paper")
    qapp.processEvents()
    win.set_theme("console")


def test_report_and_json_export(example, qapp, tmp_path):
    from ensim.ui.report import write_json, write_report

    win = example
    win.show_section(2)
    win.run_current()
    report = tmp_path / "report.md"
    data = tmp_path / "results.json"
    write_report(win.workspace, report)
    write_json(win.workspace, data)
    text = report.read_text(encoding="utf-8")
    assert "## Engine" in text and "## Flight" in text
    parsed = json.loads(data.read_text(encoding="utf-8"))
    assert parsed["engine"]["vacuum"]["isp"] == pytest.approx(win.workspace.result.vacuum.isp)
    assert parsed["flight"]["apogee_m"] > 0.0


def test_bad_project_file_is_reported_not_raised(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: shown.append(args[2]))
    bad = tmp_path / "bad.ensim"
    bad.write_text("{", encoding="utf-8")
    assert not window.open_path(bad, confirm=False)
    assert shown and "not a valid project" in shown[0]


def test_failed_cooling_run_clears_the_previous_result(example, qapp):
    win = example
    win.show_section(0)
    win.run_current()
    engine = win.sections[0]
    engine.views.setCurrentIndex(4)  # Cooling
    qapp.processEvents()
    cooling = engine.cooling.widget
    cooling.run()
    assert cooling._results is not None
    # Channels so small that the pressure drop exceeds the inlet pressure
    cooling.width.set_value_si(2e-4)
    cooling.height.set_value_si(2e-4)
    cooling.inlet_p_ratio.set_value_si(1.05)
    cooling.run()
    assert cooling._results is None
    assert cooling.m_wall._si is None
    assert "pressure drop exceeds the inlet pressure" in cooling.note.text()
