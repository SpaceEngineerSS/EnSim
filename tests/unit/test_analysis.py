import numpy as np
import pytest

from ensim.core.analysis import SweepConfig, run_sweep


def _config(parameter: str, start: float, end: float) -> SweepConfig:
    return SweepConfig(
        parameter=parameter,
        start=start,
        end=end,
        steps=5,
        base_chamber_pressure_bar=50.0,
        base_expansion_ratio=20.0,
        base_throat_area_cm2=10.0,
    )


def test_pressure_sweep_holds_frozen_gas_properties_and_increases_thrust():
    result = run_sweep(
        _config("chamber_pressure", 20.0, 100.0),
        gamma=1.22,
        mean_mw=22.0,
        temperature=3400.0,
    )

    assert np.all(np.diff(result.thrust_vacuum) > 0.0)
    assert np.ptp(result.isp_vacuum) == pytest.approx(0.0, abs=1e-10)
    assert np.all(result.isp_sea_level < result.isp_vacuum)


def test_expansion_sweep_rejects_area_ratio_below_unity():
    with pytest.raises(ValueError, match="expansion ratio"):
        run_sweep(
            _config("expansion_ratio", 0.5, 20.0),
            gamma=1.22,
            mean_mw=22.0,
            temperature=3400.0,
        )
