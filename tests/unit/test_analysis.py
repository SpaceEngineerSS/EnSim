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


def test_engine_sweep_resolves_equilibrium_at_every_point():
    import numpy as np

    from ensim.core.analysis import sweep_engine
    from ensim.core.engine import EngineInputs

    base = EngineInputs("LCH4", "LOX", 3.4, 100e5, 35.0, throat_area=0.01)
    sweep = sweep_engine(base, "of_ratio", np.linspace(2.4, 4.4, 6))
    # Chamber temperature peaks inside the range, Isp peaks fuel-rich of it
    assert 0 < np.argmax(sweep.chamber_temperature) < 5
    assert np.argmax(sweep.isp_vacuum) <= np.argmax(sweep.chamber_temperature)
    assert np.all(np.isfinite(sweep.isp_vacuum))
