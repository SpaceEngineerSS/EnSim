import json
from pathlib import Path

import pytest

from ensim.core.project_manager import FORMAT_VERSION, ProjectFileError, ProjectManager
from ensim.core.rocket import NoseShape, create_default_rocket

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    ("filename", "fuel", "oxidizer"),
    [
        ("h2_o2_vacuum.ensim", "LH2", "LOX"),
        ("methalox_sealevel.ensim", "LCH4", "LOX"),
        ("rp1_lox_reference.ensim", "RP-1", "LOX"),
    ],
)
def test_packaged_examples_load(filename, fuel, oxidizer):
    data = ProjectManager().load(EXAMPLES / filename)
    assert data.engine.fuel == fuel
    assert data.engine.oxidizer == oxidizer
    assert data.engine.chamber_pressure_bar > 0.0
    assert data.engine.expansion_ratio > 1.0


def test_round_trip_preserves_every_section(tmp_path):
    source = ProjectManager()
    source.new_project()
    source.update_inputs("LCH4", "LOX", 3.45, 120.0, 85.0, 42.0, 0.0)
    source.data.rocket.name = "Reference Vehicle"
    source.data.rocket.engine_mass_dry = 3.3
    source.data.recovery.dual_deploy = True
    source.data.environment.wind_speed = 4.0
    source.data.last_isp_vacuum = 355.2

    written = source.save(tmp_path / "round_trip")
    assert written.suffix == ".ensim"

    restored = ProjectManager()
    data = restored.load(written)
    assert data.version == FORMAT_VERSION
    assert data.engine == source.data.engine
    assert data.rocket == source.data.rocket
    assert data.recovery == source.data.recovery
    assert data.environment == source.data.environment
    assert data.last_isp_vacuum == pytest.approx(355.2)
    assert not restored.is_modified


def test_saved_dry_mass_and_tank_length_reach_the_rocket(tmp_path):
    project = ProjectManager()
    project.new_project()
    project.data.rocket.engine_mass_dry = 4.2
    project.data.rocket.tank_length = 0.7
    rocket = project.build_rocket()
    assert rocket.engine.engine_mass_dry == pytest.approx(4.2)
    assert rocket.engine.tank_length == pytest.approx(0.7)


def test_default_project_rocket_matches_default_rocket():
    built = ProjectManager().build_rocket()
    default = create_default_rocket()
    assert built.wet_mass == pytest.approx(default.wet_mass)
    assert built.total_length == pytest.approx(default.total_length)


def test_haack_nose_survives_and_unknown_shape_is_reported():
    project = ProjectManager()
    project.data.rocket.nose_shape = "haack"
    assert project.build_rocket().nose.shape is NoseShape.HAACK
    project.data.rocket.nose_shape = "bullet"
    with pytest.raises(ProjectFileError, match="bullet"):
        project.build_rocket()


def test_v1_flat_file_and_unknown_fields_load(tmp_path):
    flat = tmp_path / "old.ensim"
    flat.write_text(
        json.dumps({"fuel": "CH4", "oxidizer": "O2", "of_ratio": 3.2}), encoding="utf-8"
    )
    migrated = ProjectManager().load(flat).engine
    assert migrated.of_ratio == pytest.approx(3.2)
    assert (migrated.fuel, migrated.oxidizer) == ("LCH4", "LOX")

    newer = tmp_path / "newer.ensim"
    newer.write_text(
        json.dumps({"meta": {"version": "9.0"}, "engine": {"fuel": "H2", "future_field": 1}}),
        encoding="utf-8",
    )
    assert ProjectManager().load(newer).engine.fuel == "H2"


def test_invalid_file_raises_and_keeps_previous_path(tmp_path):
    path = tmp_path / "invalid.ensim"
    path.write_text("not json", encoding="utf-8")
    project = ProjectManager()
    with pytest.raises(ProjectFileError, match="not a valid project"):
        project.load(path)
    assert project.current_path is None
    with pytest.raises(ProjectFileError):
        project.load(tmp_path / "missing.ensim")


def test_save_without_path_is_an_error():
    with pytest.raises(ProjectFileError):
        ProjectManager().save()


def test_v3_ambient_label_becomes_a_pressure(tmp_path):
    path = tmp_path / "v3.ensim"
    path.write_text(
        json.dumps({"meta": {"version": "3.0"}, "engine": {"ambient": "Sea Level (1.01 bar)"}}),
        encoding="utf-8",
    )
    assert ProjectManager().load(path).engine.ambient_pressure_bar == pytest.approx(1.01)
