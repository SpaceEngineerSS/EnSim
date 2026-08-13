"""Physics checks for trajectory and mass optimization."""

import numpy as np
import pytest

from ensim.core.optimization import (
    TrajectoryConstraints,
    _evaluate_gravity_turn,
    optimize_gravity_turn,
    optimize_propellant_load,
)


def test_vertical_powered_ascent_respects_rocket_equation_upper_bound():
    evaluation = _evaluate_gravity_turn(
        vehicle_mass=1000.0,
        thrust=20_000.0,
        isp=300.0,
        propellant_mass=500.0,
        kickoff_altitude=1e9,
        kickoff_angle_deg=0.0,
        pitch_rate_deg_s=0.0,
        reference_area=0.0,
        drag_coefficient=0.0,
        integration_step=0.05,
    )
    ideal_delta_v = 300.0 * 9.80665 * np.log(2.0)
    assert 0.0 < evaluation.final_vertical_velocity < ideal_delta_v
    assert evaluation.final_horizontal_velocity == pytest.approx(0.0, abs=1e-12)
    assert evaluation.burn_time == pytest.approx(500.0 / (20_000.0 / (300.0 * 9.80665)))


def test_ascent_integration_converges_under_step_refinement():
    common = {
        "vehicle_mass": 1000.0,
        "thrust": 20_000.0,
        "isp": 300.0,
        "propellant_mass": 500.0,
        "kickoff_altitude": 500.0,
        "kickoff_angle_deg": 5.0,
        "pitch_rate_deg_s": 0.5,
        "reference_area": 1.0,
        "drag_coefficient": 0.3,
    }
    coarse = _evaluate_gravity_turn(**common, integration_step=0.25)
    fine = _evaluate_gravity_turn(**common, integration_step=0.125)
    assert coarse.final_altitude == pytest.approx(fine.final_altitude, rel=2e-3)
    assert coarse.final_velocity == pytest.approx(fine.final_velocity, rel=2e-3)
    assert coarse.max_dynamic_pressure == pytest.approx(fine.max_dynamic_pressure, rel=3e-3)


def test_gravity_turn_optimizer_reports_simulated_path_metrics():
    result = optimize_gravity_turn(
        vehicle_mass=1000.0,
        thrust=20_000.0,
        isp=300.0,
        propellant_mass=500.0,
        constraints=TrajectoryConstraints(
            max_dynamic_pressure=100_000.0,
            max_acceleration=5.0,
            target_altitude=20_000.0,
            target_velocity=700.0,
            target_flight_path_angle=np.radians(50.0),
        ),
        reference_area=1.0,
        drag_coefficient=0.3,
        integration_step=0.5,
    )
    assert result.optimal_params["model"] == "planar_3dof_powered_ascent"
    assert result.optimal_params["final_altitude"] > 0.0
    assert result.optimal_params["max_dynamic_pressure"] > 0.0
    assert result.optimal_params["constraints_satisfied"] is True


def test_propellant_load_closed_form_and_infeasible_case():
    feasible = optimize_propellant_load(
        dry_mass=100.0,
        tank_volume=1.0,
        propellant_density=1000.0,
        target_delta_v=1000.0,
        isp=300.0,
    )
    assert feasible.success
    infeasible = optimize_propellant_load(
        dry_mass=100.0,
        tank_volume=0.01,
        propellant_density=1000.0,
        target_delta_v=5000.0,
        isp=300.0,
    )
    assert not infeasible.success
    assert infeasible.optimal_params["delta_v_shortfall"] > 0.0
