import numpy as np
import pytest

from ensim.core.mission import analyze_altitude_sweep, get_atmosphere


def test_standard_atmosphere_reference_states():
    sea_level = get_atmosphere(0.0)
    tropopause = get_atmosphere(11_000.0)

    assert sea_level.temperature == pytest.approx(288.15)
    assert sea_level.pressure == pytest.approx(101_325.0)
    assert sea_level.density == pytest.approx(1.2250, rel=2e-4)
    assert tropopause.temperature == pytest.approx(216.65)
    assert tropopause.pressure == pytest.approx(22_632.1, rel=2e-4)


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


def test_atmosphere_uses_explicit_vacuum_above_model_top():
    atmosphere = get_atmosphere(90_000.0)
    assert atmosphere.pressure == 0.0
    assert atmosphere.density == 0.0
    assert atmosphere.speed_of_sound == 0.0
