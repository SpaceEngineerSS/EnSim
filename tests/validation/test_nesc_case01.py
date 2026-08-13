"""Cross-comparison with NASA NESC Atmospheric Check-Case 01."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from ensim.core.ballistics import propagate_dragless_wgs84

FEET_PER_METRE = 1.0 / 0.3048
REFERENCE_FILE = Path(__file__).parents[1] / "reference" / "nesc_atmospheric_case_01_sim_06.csv"


def test_nesc_atmospheric_case_01_translational_history():
    reference = np.genfromtxt(REFERENCE_FILE, delimiter=",", names=True)
    trajectory = propagate_dragless_wgs84(
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=30_000.0 * 0.3048,
        duration=30.0,
        output_step=5.0,
    )

    assert_allclose(trajectory.time, reference["time_s"], atol=1e-12)
    assert_allclose(
        trajectory.eci_position[:, 0] * FEET_PER_METRE,
        reference["eci_x_ft"],
        rtol=0.0,
        atol=2e-3,
    )
    assert_allclose(
        trajectory.eci_position[:, 1] * FEET_PER_METRE,
        reference["eci_y_ft"],
        rtol=0.0,
        atol=2e-3,
    )
    assert_allclose(
        trajectory.eci_velocity[:, 0] * FEET_PER_METRE,
        reference["eci_vx_ft_s"],
        rtol=0.0,
        atol=2e-4,
    )
    assert_allclose(
        trajectory.eci_velocity[:, 1] * FEET_PER_METRE,
        reference["eci_vy_ft_s"],
        rtol=0.0,
        atol=2e-4,
    )
    assert_allclose(
        trajectory.ned_velocity_m_s[:, 2] * FEET_PER_METRE,
        reference["ned_down_velocity_ft_s"],
        rtol=0.0,
        atol=2e-4,
    )
    assert_allclose(
        trajectory.ellipsoid_altitude_m * FEET_PER_METRE,
        reference["altitude_msl_ft"],
        rtol=0.0,
        atol=2e-3,
    )
    assert_allclose(
        trajectory.longitude_deg,
        reference["longitude_deg"],
        rtol=0.0,
        atol=2e-10,
    )
    assert_allclose(
        trajectory.latitude_deg,
        reference["latitude_deg"],
        rtol=0.0,
        atol=1e-12,
    )
