import json

import pytest

from ensim.core.constants import P_ATM
from ensim.core.engine import EngineInputs, analyze_engine


@pytest.fixture(scope="module")
def kerolox():
    return analyze_engine(
        EngineInputs(
            "RP-1",
            "LOX",
            of_ratio=2.6,
            chamber_pressure=7e6,
            expansion_ratio=25.0,
            throat_area=0.01,
            ambient_pressure=P_ATM,
        )
    )


def test_kerolox_operating_point_is_physical(kerolox):
    assert 3400.0 < kerolox.chamber_temperature < 3800.0
    assert 1.1 < kerolox.gamma < 1.3
    assert kerolox.vacuum.isp > kerolox.sea_level.isp > 0.0
    assert kerolox.operating.isp == pytest.approx(kerolox.sea_level.isp)
    assert kerolox.vacuum.thrust > kerolox.sea_level.thrust
    assert kerolox.vacuum.mass_flow_rate == pytest.approx(kerolox.sea_level.mass_flow_rate)
    assert sum(kerolox.composition.values()) == pytest.approx(1.0, abs=1e-4)
    assert {"H2O", "CO2", "CO"} <= set(kerolox.composition)


def test_result_serialises_to_json(kerolox):
    text = json.dumps(kerolox.to_dict())
    data = json.loads(text)
    assert data["chamber"]["temperature"] == pytest.approx(kerolox.chamber_temperature)
    assert data["vacuum"]["isp"] == pytest.approx(kerolox.vacuum.isp)


def test_low_pressure_vacuum_engine_has_no_sea_level_point():
    result = analyze_engine(
        EngineInputs("H2", "O2", of_ratio=6.0, chamber_pressure=1.5e5, expansion_ratio=80.0)
    )
    assert result.sea_level is None
    assert result.vacuum.isp > 400.0


def test_unknown_propellant_is_reported():
    with pytest.raises(KeyError, match="Unobtainium"):
        analyze_engine(EngineInputs("Unobtainium", "LOX", 2.0, 5e6, 20.0))


@pytest.mark.parametrize(
    "kwargs",
    [{"of_ratio": 0.0}, {"chamber_pressure": -1.0}, {"expansion_ratio": 1.0}, {"throat_area": 0.0}],
)
def test_invalid_inputs_are_rejected(kwargs):
    values = {"of_ratio": 2.0, "chamber_pressure": 5e6, "expansion_ratio": 20.0}
    values.update(kwargs)
    with pytest.raises(ValueError):
        EngineInputs("LCH4", "LOX", **values)
