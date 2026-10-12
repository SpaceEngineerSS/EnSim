"""End-to-end checks of the dispersion runner."""

import numpy as np
import pytest

from ensim.core.flight_6dof import SimulationCancelled, simulate_flight_6dof
from ensim.core.monte_carlo import DispersionConfig, run_monte_carlo
from ensim.core.rocket import create_default_rocket

NOMINAL = {"thrust_vac": 1500.0, "isp_vac": 220.0, "burn_time": 3.0}


def _quiet_config(**overrides):
    values = {
        "num_simulations": 4,
        "thrust_sigma": 0.0,
        "isp_sigma": 0.0,
        "burn_time_sigma": 0.0,
        "cd_sigma": 0.0,
        "fin_misalignment_max": 0.0,
        "launch_angle_sigma": 0.0,
        "wind_speed_mean": 0.0,
        "wind_speed_sigma": 0.0,
        "seed": 7,
        "n_workers": 1,
    }
    values.update(overrides)
    return DispersionConfig(**values)


def test_runs_use_the_vehicle_drag_coefficient():
    low_drag = create_default_rocket()
    low_drag.axial_drag_coefficient = 0.2
    high_drag = create_default_rocket()
    high_drag.axial_drag_coefficient = 0.9

    kwargs = {**NOMINAL, "launch_angle_deg": 90.0, "max_time": 200.0, "config": _quiet_config()}
    low = run_monte_carlo(low_drag, **kwargs)
    high = run_monte_carlo(high_drag, **kwargs)
    assert low.apogee_mean > high.apogee_mean

    direct = simulate_flight_6dof(
        high_drag, **NOMINAL, launch_angle_deg=90.0, max_time=200.0, dt=0.02
    )
    assert high.apogee_mean == pytest.approx(direct.apogee_altitude, rel=1e-9)


def test_only_ground_impacts_enter_landing_statistics():
    short = run_monte_carlo(
        create_default_rocket(), **NOMINAL, max_time=5.0, config=_quiet_config()
    )
    assert short.n_successful == 4
    assert short.n_landed == 0
    assert short.landing_points.shape == (0, 2)
    assert np.isnan(short.cep_radius)

    full = run_monte_carlo(
        create_default_rocket(),
        **NOMINAL,
        max_time=200.0,
        config=_quiet_config(wind_speed_mean=4.0, wind_speed_sigma=1.0),
    )
    assert full.n_landed == 4
    assert np.all(np.isfinite(full.landing_points))


def test_progress_and_cancel_callbacks():
    seen = []
    run_monte_carlo(
        create_default_rocket(),
        **NOMINAL,
        max_time=5.0,
        config=_quiet_config(),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (4, 4)

    with pytest.raises(SimulationCancelled):
        run_monte_carlo(
            create_default_rocket(),
            **NOMINAL,
            max_time=5.0,
            config=_quiet_config(),
            cancel=lambda: True,
        )


def test_launch_elevation_must_be_above_horizon():
    with pytest.raises(ValueError):
        run_monte_carlo(create_default_rocket(), **NOMINAL, launch_angle_deg=120.0)
