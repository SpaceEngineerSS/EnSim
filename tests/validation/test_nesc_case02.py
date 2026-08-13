"""Cross-comparison with NASA NESC Atmospheric Check-Case 02."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from ensim.core.ballistics import propagate_dragless_wgs84
from ensim.core.flight_6dof import rk4_step_6dof
from ensim.core.geodesy import ecef_to_enu_matrix, eci_to_ecef, eci_to_ecef_matrix
from ensim.core.math_utils import (
    q_from_rotation_matrix,
    q_to_euler,
    q_to_rotation_matrix,
)

REFERENCE_FILE = Path(__file__).parents[1] / "reference" / "nesc_atmospheric_case_02_sim_04.csv"
NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def test_nesc_atmospheric_case_02_tumbling_brick_history():
    reference = np.genfromtxt(REFERENCE_FILE, delimiter=",", names=True)
    trajectory = propagate_dragless_wgs84(
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=30_000.0 * 0.3048,
        duration=30.0,
        output_step=5.0,
    )
    body_to_ecef = ecef_to_enu_matrix(0.0, 0.0).T @ NED_TO_ENU
    initial_quaternion = q_from_rotation_matrix(body_to_ecef)
    inertia = np.array([0.001894220, 0.006211019, 0.007194665])
    state = np.concatenate(
        (
            np.zeros(6),
            initial_quaternion,
            np.radians([10.0, 20.0, 30.0]),
            [0.0],
        )
    )
    zero = np.zeros(3)
    euler_history = np.empty((len(reference), 3))
    rate_history = np.empty((len(reference), 3))
    reference_index = 0

    for step in range(3001):
        elapsed_time = step * 0.01
        if abs(elapsed_time - reference["time_s"][reference_index]) < 1e-12:
            position_ecef = eci_to_ecef(
                trajectory.eci_position[reference_index],
                elapsed_time,
            )
            latitude = np.arctan2(
                position_ecef[2],
                np.hypot(position_ecef[0], position_ecef[1]),
            )
            longitude = np.arctan2(position_ecef[1], position_ecef[0])
            ned_to_ecef = ecef_to_enu_matrix(latitude, longitude).T @ NED_TO_ENU
            body_to_ned = (
                ned_to_ecef.T @ eci_to_ecef_matrix(elapsed_time) @ q_to_rotation_matrix(state[6:10])
            )
            euler_history[reference_index] = np.degrees(
                q_to_euler(q_from_rotation_matrix(body_to_ned))
            )
            rate_history[reference_index] = np.degrees(state[10:13])
            reference_index += 1
            if reference_index == len(reference):
                break
        state = rk4_step_6dof(
            state,
            0.01,
            1.0,
            inertia,
            zero,
            zero,
            zero,
            zero,
            0.0,
        )

    expected_euler = np.column_stack(
        (reference["roll_deg"], reference["pitch_deg"], reference["yaw_deg"])
    )
    expected_rates = np.column_stack(
        (
            reference["roll_rate_deg_s"],
            reference["pitch_rate_deg_s"],
            reference["yaw_rate_deg_s"],
        )
    )
    assert_allclose(euler_history, expected_euler, rtol=0.0, atol=2e-8)
    assert_allclose(rate_history, expected_rates, rtol=0.0, atol=2e-8)
