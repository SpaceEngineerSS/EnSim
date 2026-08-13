"""Verification tests for WGS-84 coordinates and Earth-fixed dynamics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ensim.core.geodesy import (
    EARTH_ROTATION_RATE,
    WGS84_SEMI_MAJOR_AXIS,
    WGS84_SEMI_MINOR_AXIS,
    earth_fixed_acceleration,
    ecef_to_eci,
    ecef_to_enu_matrix,
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
    j2_gravity_ecef,
)


def test_wgs84_equator_and_pole_positions():
    assert_allclose(
        geodetic_to_ecef(0.0, 0.0, 0.0),
        [WGS84_SEMI_MAJOR_AXIS, 0.0, 0.0],
        atol=1e-9,
    )
    assert_allclose(
        geodetic_to_ecef(np.pi / 2.0, 0.0, 0.0),
        [0.0, 0.0, WGS84_SEMI_MINOR_AXIS],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("latitude", "longitude", "altitude"),
    [
        (0.0, 0.0, 0.0),
        (np.radians(41.0082), np.radians(28.9784), 125.0),
        (np.radians(-33.8688), np.radians(151.2093), 15_000.0),
        (np.radians(89.999), np.radians(-45.0), 400_000.0),
    ],
)
def test_geodetic_ecef_round_trip(latitude, longitude, altitude):
    recovered = ecef_to_geodetic(geodetic_to_ecef(latitude, longitude, altitude))
    assert recovered[0] == pytest.approx(latitude, abs=2e-12)
    assert recovered[1] == pytest.approx(longitude, abs=2e-12)
    assert recovered[2] == pytest.approx(altitude, abs=2e-5)


def test_enu_matrix_is_orthonormal_and_has_expected_equatorial_axes():
    rotation = ecef_to_enu_matrix(0.0, 0.0)
    assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)
    assert_allclose(rotation, [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-15)


def test_eci_ecef_rotation_is_reversible():
    vector = np.array([6.8e6, -1.2e6, 3.4e6])
    elapsed = 12_345.0
    assert_allclose(eci_to_ecef(ecef_to_eci(vector, elapsed), elapsed), vector, atol=1e-9)


def test_j2_gravity_points_inward_on_equator_and_pole():
    equator = j2_gravity_ecef(np.array([WGS84_SEMI_MAJOR_AXIS, 0.0, 0.0]))
    pole = j2_gravity_ecef(np.array([0.0, 0.0, WGS84_SEMI_MINOR_AXIS]))
    assert equator[0] < 0.0
    assert pole[2] < 0.0
    assert_allclose(equator[1:], 0.0, atol=0.0)
    assert_allclose(pole[:2], 0.0, atol=0.0)


def test_stationary_surface_effective_gravity_matches_wgs84_scale():
    equator_position = geodetic_to_ecef(0.0, 0.0, 0.0)
    pole_position = geodetic_to_ecef(np.pi / 2.0, 0.0, 0.0)
    equator = np.linalg.norm(earth_fixed_acceleration(equator_position, np.zeros(3)))
    pole = np.linalg.norm(earth_fixed_acceleration(pole_position, np.zeros(3)))
    assert 9.77 < equator < 9.79
    assert 9.82 < pole < 9.84
    assert pole > equator


def test_coriolis_term_for_upward_equatorial_motion_is_westward():
    position = geodetic_to_ecef(0.0, 0.0, 0.0)
    stationary = earth_fixed_acceleration(position, np.zeros(3))
    upward_velocity_ecef = np.array([100.0, 0.0, 0.0])
    moving = earth_fixed_acceleration(position, upward_velocity_ecef)
    coriolis = moving - stationary
    assert coriolis[1] == pytest.approx(-2.0 * EARTH_ROTATION_RATE * 100.0)
    assert_allclose(coriolis[[0, 2]], 0.0, atol=1e-15)
