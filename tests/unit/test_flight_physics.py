"""Behavioural checks of the full-state 6-DOF flight model."""

import copy

import numpy as np
import pytest

from ensim.core.flight_6dof import (
    SimulationCancelled,
    calculate_aerodynamic_moment,
    derivatives_6dof,
    simulate_flight_6dof,
)
from ensim.core.math_utils import q_conjugate, q_derivative, q_normalize, q_rotate_vector
from ensim.core.rocket import create_default_rocket


def _flight(**overrides):
    kwargs = {
        "thrust_vac": 1500.0,
        "isp_vac": 220.0,
        "burn_time": 3.0,
        "max_time": 60.0,
        "launch_angle_deg": 90.0,
        "rail_length": 2.0,
    }
    kwargs.update(overrides)
    return simulate_flight_6dof(create_default_rocket(), **kwargs)


def test_restoring_moment_turns_nose_into_relative_wind():
    """CP aft of CG must reduce the crossflow, not amplify it."""
    velocity = np.array([50.0, 0.0, 5.0])
    normal_force = np.array([0.0, 0.0, -velocity[2] / np.linalg.norm(velocity)])
    moment = calculate_aerodynamic_moment(
        cp_position=1.0, cg_position=0.5, force_aero_body=normal_force
    )

    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_next = q_normalize(q + 0.01 * q_derivative(q, moment))
    velocity_body = q_rotate_vector(q_conjugate(q_next), velocity)
    assert abs(velocity_body[2]) < abs(velocity[2])


def test_stable_rocket_weathercocks_upwind():
    """With wind from the north, a stable rocket tilts north and lands upwind of a ballistic body."""
    result = _flight(wind_speed=6.0, wind_direction_deg=0.0, max_time=25.0)
    assert result.stability_margin[0] > 1.0
    burnout = np.searchsorted(result.time, result.burnout_time)
    assert result.position_y[burnout] > 0.0
    # Weathercocking has damped the rail-exit angle of attack by burnout
    assert abs(result.angle_of_attack[burnout]) < 10.0


def test_max_acceleration_is_sensed_specific_force():
    result = _flight()
    burning = (result.thrust > 0.0) & (result.time > result.liftoff_time)
    expected = np.max(
        np.sqrt((result.thrust[burning] - result.drag[burning]) ** 2)
        / result.mass[burning]
        / 9.80665
    )
    assert result.max_acceleration == pytest.approx(expected, rel=0.02)
    # Weight is not part of the sensed acceleration in flight
    assert result.max_acceleration < np.max(result.thrust / result.mass / 9.80665) + 1e-9


def test_rocket_is_not_modified():
    rocket = create_default_rocket()
    snapshot = copy.deepcopy(rocket)
    simulate_flight_6dof(rocket, thrust_vac=1500.0, isp_vac=220.0, burn_time=3.0, max_time=5.0)
    assert rocket == snapshot


def test_cancel_callback_stops_run():
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 10

    with pytest.raises(SimulationCancelled):
        _flight(cancel=cancel)


def test_landing_point_reported_only_after_ground_impact():
    landed = _flight(max_time=120.0)
    assert landed.ground_impact
    assert landed.landing_x is not None and landed.landing_y is not None
    assert landed.position_z[-1] == pytest.approx(0.0, abs=1e-9)

    airborne = _flight(max_time=5.0)
    assert not airborne.ground_impact
    assert airborne.landing_x is None


def test_parachute_slows_descent():
    ballistic = _flight(max_time=200.0)
    chute = _flight(max_time=200.0, chute_diameter=1.0)
    assert chute.parachute_deploy_time == pytest.approx(chute.apogee_time, abs=0.05)
    assert abs(chute.velocity_z[-1]) < 0.5 * abs(ballistic.velocity_z[-1])
    assert chute.flight_time > ballistic.flight_time


def test_delayed_deployment_waits_for_altitude():
    result = _flight(
        max_time=200.0, chute_diameter=1.0, deploy_at_apogee=False, main_deploy_altitude_m=100.0
    )
    index = np.searchsorted(result.time, result.parachute_deploy_time)
    assert result.position_z[index] <= 101.0


def test_adaptive_run_records_requested_output_grid():
    result = _flight(use_adaptive=True, max_time=10.0, output_dt=0.05)
    spacing = np.diff(result.time)
    assert np.allclose(spacing, 0.05, atol=1e-9)
    assert result.forced_min_steps == 0


def test_derivative_has_full_state_length_for_zero_mass():
    state = np.zeros(14)
    state[6] = 1.0
    zero = np.zeros(3)
    deriv = derivatives_6dof(state, 0.0, np.ones(3), zero, zero, zero, zero, 0.0)
    assert deriv.shape == (14,)
    assert np.all(deriv == 0.0)


def test_fin_misalignment_produces_trim_moment_not_initial_tilt():
    aligned = _flight(max_time=4.0)
    canted = _flight(max_time=4.0, fin_misalignment_deg=1.0)
    assert aligned.quaternion_w[0] == pytest.approx(canted.quaternion_w[0])
    lateral = np.hypot(canted.position_x[-1], canted.position_y[-1])
    assert lateral > np.hypot(aligned.position_x[-1], aligned.position_y[-1]) + 0.01
