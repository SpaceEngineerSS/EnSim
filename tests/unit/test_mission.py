import numpy as np
import pytest

from ensim.core.mission import ATMOSPHERE_MODEL_TOP, analyze_altitude_sweep, get_atmosphere


@pytest.mark.parametrize(
    ("geometric_altitude", "temperature", "pressure", "density"),
    [
        # U.S. Standard Atmosphere 1976, Table I (geometric altitude)
        (0.0, 288.150, 101_325.0, 1.2250),
        (11_000.0, 216.774, 22_699.9, 0.36480),
        (20_000.0, 216.650, 5_529.3, 0.088910),
        (50_000.0, 270.650, 79.779, 1.0269e-3),
        (80_000.0, 198.639, 1.0524, 1.8458e-5),
    ],
)
def test_standard_atmosphere_table(geometric_altitude, temperature, pressure, density):
    state = get_atmosphere(geometric_altitude)
    assert state.temperature == pytest.approx(temperature, abs=2e-3)
    assert state.pressure == pytest.approx(pressure, rel=2e-4)
    assert state.density == pytest.approx(density, rel=3e-4)
    assert state.within_model


def test_altitude_sweep_is_steady_performance_grid():
    profile = analyze_altitude_sweep(
        T_chamber=3500.0,
        P_chamber=7.0e6,
        gamma=1.2,
        mean_mw=22.0,
        expansion_ratio=20.0,
        throat_area=0.01,
        max_altitude=20_000.0,
        step_size=2_000.0,
    )

    assert np.array_equal(profile.altitudes, np.arange(0.0, 20_001.0, 2_000.0))
    assert np.all(np.diff(profile.thrust) > 0.0)
    assert np.all(np.diff(profile.isp) > 0.0)
    assert profile.optimal_altitude in profile.altitudes
    assert len(profile.pressure_ratio) == len(profile.altitudes)


def test_atmosphere_rejects_negative_altitude():
    with pytest.raises(ValueError):
        get_atmosphere(-1.0)


def test_atmosphere_is_continuous_above_model_top():
    top = get_atmosphere(ATMOSPHERE_MODEL_TOP)
    above = get_atmosphere(ATMOSPHERE_MODEL_TOP + 1.0)
    far = get_atmosphere(150_000.0)
    assert above.pressure == pytest.approx(top.pressure, rel=2e-4)
    assert above.speed_of_sound == pytest.approx(top.speed_of_sound)
    assert 0.0 < far.pressure < top.pressure
    assert not far.within_model
