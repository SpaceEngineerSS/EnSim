import json
from pathlib import Path

import pytest

from ensim.cli import main

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_check_passes(capsys):
    assert main(["check"]) == 0
    assert "OK" in capsys.readouterr().out


def test_engine_text_and_json(capsys):
    args = ["engine", "--fuel", "LCH4", "--of", "3.4", "--pc", "100", "--eps", "35"]
    assert main(args) == 0
    assert "Isp vacuum" in capsys.readouterr().out
    assert main([*args, "--throat-area", "50", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["vacuum"]["thrust"] > 0.0
    assert data["chamber"]["temperature"] > 3000.0


def test_run_project_with_flight_writes_json(tmp_path):
    target = tmp_path / "out.json"
    assert (
        main(
            [
                "run",
                str(EXAMPLES / "kerolox_sounding_rocket.ensim"),
                "--flight",
                "--json",
                str(target),
            ]
        )
        == 0
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["flight"]["landed"] is True
    assert 3000.0 < data["flight"]["apogee_m"] < 12000.0
    assert data["flight"]["max_acceleration_g"] < 10.0


@pytest.mark.parametrize(
    "argv",
    [
        ["engine", "--fuel", "Xx", "--of", "2", "--pc", "10", "--eps", "5"],
        ["engine", "--fuel", "LH2", "--of", "-1", "--pc", "10", "--eps", "5"],
        ["run", "does-not-exist.ensim"],
    ],
)
def test_errors_exit_with_status_one(argv, capsys):
    assert main(argv) == 1
    assert capsys.readouterr().err.startswith("ensim ")
