from pathlib import Path

import pytest

from ensim.core.project_manager import ProjectManager

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    ("filename", "fuel", "oxidizer"),
    [
        ("h2_o2_vacuum.ensim", "H2", "O2"),
        ("methalox_sealevel.ensim", "CH4", "O2"),
        ("rp1_lox_reference.ensim", "RP1", "O2"),
    ],
)
def test_packaged_examples_load_as_v3_projects(filename, fuel, oxidizer):
    project = ProjectManager()
    data = project.load(EXAMPLES / filename)

    assert data is not None
    assert data.version == "3.0"
    assert data.engine.fuel == fuel
    assert data.engine.oxidizer == oxidizer
    assert data.engine.chamber_pressure_bar > 0.0
    assert data.engine.expansion_ratio > 1.0


def test_project_round_trip_preserves_nested_inputs(tmp_path):
    path = tmp_path / "round_trip.ensim"
    source = ProjectManager()
    source.new_project()
    source.update_inputs("CH4", "O2", 3.45, 120.0, 85.0, 42.0, "Vacuum (0 bar)")
    source.data.rocket.name = "Reference Vehicle"
    source.data.last_isp_vacuum = 355.2

    assert source.save(path)

    restored = ProjectManager()
    data = restored.load(path)
    assert data is not None
    assert data.engine == source.data.engine
    assert data.rocket == source.data.rocket
    assert data.last_isp_vacuum == pytest.approx(355.2)
    assert not restored.is_modified


def test_invalid_project_returns_none_without_mutating_path(tmp_path):
    path = tmp_path / "invalid.ensim"
    path.write_text("not json", encoding="utf-8")
    project = ProjectManager()

    assert project.load(path) is None
    assert project.current_path is None
