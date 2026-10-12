"""Liquid-propellant rocket cases against NASA CEA 3.3.2.

References come from scripts/generate_cea_references.py: reactants in their
storage state (LH2 at 20.27 K, LOX at 90.17 K, LCH4 at 111.6 K, storables at
298.15 K) and composition frozen at the chamber.

Both codes expand the frozen chamber mixture with temperature-dependent cp,
so chamber state, c* and vacuum Isp all agree to better than 0.1 %.
"""

import csv
from pathlib import Path

import pytest

from ensim.core.engine import EngineInputs, analyze_engine

REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "cea_liquid_rockets.csv"
CEA_TO_ENSIM = {
    "H2(L)": "LH2",
    "O2(L)": "LOX",
    "CH4(L)": "LCH4",
    "RP-1": "RP-1",
    "N2H4(L)": "N2H4",
    "N2O4(L)": "NTO",
    "CH6N2(L)": "MMH",
}


def _cases():
    with REFERENCE.open(encoding="utf-8") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        return [pytest.param(row, id=row["case"]) for row in rows]


@pytest.mark.validation
@pytest.mark.parametrize("row", _cases())
def test_liquid_rocket_matches_cea(row):
    result = analyze_engine(
        EngineInputs(
            CEA_TO_ENSIM[row["fuel"]],
            CEA_TO_ENSIM[row["oxidizer"]],
            of_ratio=float(row["of_ratio"]),
            chamber_pressure=float(row["chamber_pressure_bar"]) * 1e5,
            expansion_ratio=float(row["area_ratio"]),
            nozzle_half_angle_deg=0.0,  # CEA reports the loss-free nozzle
        )
    )
    assert result.chamber_temperature == pytest.approx(float(row["chamber_temperature"]), rel=5e-4)
    assert result.molecular_weight == pytest.approx(float(row["molecular_weight"]), rel=2e-3)
    assert result.gamma == pytest.approx(float(row["gamma_frozen"]), rel=1e-3)
    assert result.vacuum.c_star == pytest.approx(float(row["c_star"]), rel=1e-3)
    assert result.vacuum.isp == pytest.approx(float(row["isp_vacuum_frozen"]), rel=1e-3)


def test_liquid_reactants_run_cooler_than_gaseous_ones():
    liquid = analyze_engine(EngineInputs("LH2", "LOX", 6.0, 68.9476e5, 40.0))
    gas = analyze_engine(EngineInputs("H2", "O2", 6.0, 68.9476e5, 40.0))
    assert gas.chamber_temperature - liquid.chamber_temperature > 50.0
