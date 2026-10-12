import pytest

from ensim.core.engine import EngineInputs, analyze_engine
from ensim.core.frozen_flow import frozen_nozzle
from ensim.utils.nasa_parser import load_default_database


@pytest.fixture(scope="module")
def chamber():
    return analyze_engine(EngineInputs("LCH4", "LOX", 3.4, 100e5, 35.0)).equilibrium


@pytest.mark.parametrize("area_ratio", [2.0, 10.0, 80.0])
def test_throat_is_sonic_and_area_ratio_is_honoured(chamber, area_ratio):
    nozzle = frozen_nozzle(chamber, load_default_database(), area_ratio)
    assert nozzle.throat.mach == pytest.approx(1.0, abs=1e-4)
    assert nozzle.throat.mass_flux / nozzle.exit.mass_flux == pytest.approx(area_ratio, rel=1e-8)
    assert nozzle.exit.mach > 1.0
    assert nozzle.exit.temperature < nozzle.throat.temperature < chamber.temperature


def test_c_star_is_independent_of_area_ratio_and_isp_grows(chamber):
    database = load_default_database()
    short = frozen_nozzle(chamber, database, 5.0)
    long = frozen_nozzle(chamber, database, 50.0)
    assert short.c_star == pytest.approx(long.c_star, rel=1e-6)
    assert long.isp() > short.isp()
    assert short.isp(101325.0) < short.isp()


def test_area_ratio_must_exceed_one(chamber):
    with pytest.raises(ValueError):
        frozen_nozzle(chamber, load_default_database(), 1.0)


def test_profile_matches_nozzle_at_throat_and_exit(chamber):
    import numpy as np

    from ensim.core.frozen_flow import frozen_profile

    database = load_default_database()
    nozzle = frozen_nozzle(chamber, database, 35.0)
    ratios = np.array([4.0, 1.5, 1.0, 1.5, 35.0])
    branch = np.array([False, False, True, True, True])
    profile = frozen_profile(chamber, database, ratios, branch)
    assert profile["mach"][0] < profile["mach"][1] < 1.0 < profile["mach"][3]
    assert profile["mach"][2] == pytest.approx(1.0, abs=1e-4)
    assert profile["pressure"][-1] == pytest.approx(nozzle.exit.pressure, rel=1e-8)
    assert np.all(np.diff(profile["temperature"]) < 0.0)


def test_performance_at_altitude_interpolates_between_sea_level_and_vacuum():
    from ensim.core.engine import EngineInputs, analyze_engine

    result = analyze_engine(EngineInputs("RP-1", "LOX", 2.6, 70e5, 16.0, throat_area=0.01))
    mid = result.performance_at(0.5 * 101325.0)
    assert result.sea_level.isp < mid.isp < result.vacuum.isp
    assert result.performance_at(50 * 101325.0) is None  # above the throat pressure
