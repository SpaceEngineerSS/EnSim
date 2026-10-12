"""WGS-84 coordinate transforms and Earth-fixed gravitation.

The equations follow the WGS-84 ellipsoid and the J2 acceleration model used
by the NASA NESC 6-DOF verification check-case family. Angles are radians and
SI units are used throughout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ensim.utils.numba_support import jit

from .constants import (
    EARTH_GRAVITATIONAL_PARAMETER,
    EARTH_J2,
    EARTH_ROTATION_RATE,
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_AXIS,
)

WGS84_FIRST_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
WGS84_SEMI_MINOR_AXIS = WGS84_SEMI_MAJOR_AXIS * (1.0 - WGS84_FLATTENING)

Vector = NDArray[np.float64]


def geodetic_to_ecef(latitude: float, longitude: float, altitude: float) -> Vector:
    """Convert WGS-84 geodetic coordinates to ECEF position in metres."""
    if not np.isfinite([latitude, longitude, altitude]).all():
        raise ValueError("Geodetic coordinates must be finite")
    if not -np.pi / 2.0 <= latitude <= np.pi / 2.0:
        raise ValueError("Latitude must be between -pi/2 and pi/2")

    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    radius_prime_vertical = WGS84_SEMI_MAJOR_AXIS / np.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_latitude**2
    )
    radial = (radius_prime_vertical + altitude) * cos_latitude
    return np.array(
        [
            radial * np.cos(longitude),
            radial * np.sin(longitude),
            (radius_prime_vertical * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED) + altitude)
            * sin_latitude,
        ],
        dtype=np.float64,
    )


@jit(nopython=True, cache=True)
def geodetic_from_ecef(position: Vector) -> tuple[float, float, float]:
    """Latitude, longitude (rad) and ellipsoidal height (m) of an ECEF point.

    Fixed-point iteration on latitude; converges to 1e-13 rad in a few
    passes for any point outside Earth's core. No input validation.
    """
    x = position[0]
    y = position[1]
    z = position[2]
    horizontal_radius = np.sqrt(x * x + y * y)
    longitude = np.arctan2(y, x)
    if horizontal_radius < 1e-9:
        latitude = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        return latitude, longitude, abs(z) - WGS84_SEMI_MINOR_AXIS

    latitude = np.arctan2(z, horizontal_radius * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED))
    for _ in range(10):
        sin_latitude = np.sin(latitude)
        prime_vertical = WGS84_SEMI_MAJOR_AXIS / np.sqrt(
            1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude
        )
        altitude = horizontal_radius / np.cos(latitude) - prime_vertical
        updated = np.arctan2(
            z,
            horizontal_radius
            * (
                1.0
                - WGS84_FIRST_ECCENTRICITY_SQUARED * prime_vertical / (prime_vertical + altitude)
            ),
        )
        converged = abs(updated - latitude) < 1e-13
        latitude = updated
        if converged:
            break

    sin_latitude = np.sin(latitude)
    prime_vertical = WGS84_SEMI_MAJOR_AXIS / np.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude
    )
    return latitude, longitude, horizontal_radius / np.cos(latitude) - prime_vertical


def ecef_to_geodetic(position: Vector) -> tuple[float, float, float]:
    """Convert an ECEF position to WGS-84 latitude, longitude and altitude."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("ECEF position must be a finite three-vector")
    if float(np.hypot(position[0], position[1])) < 1e-9 and abs(position[2]) < 1e-9:
        raise ValueError("Geodetic coordinates are undefined at Earth's centre")
    latitude, longitude, altitude = geodetic_from_ecef(position)
    return float(latitude), float(longitude), float(altitude)


@jit(nopython=True, cache=True)
def ecef_to_enu_matrix(latitude: float, longitude: float) -> NDArray[np.float64]:
    """Return the direction-cosine matrix that rotates ECEF vectors to ENU."""
    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    sin_longitude = np.sin(longitude)
    cos_longitude = np.cos(longitude)
    matrix = np.empty((3, 3))
    matrix[0, 0] = -sin_longitude
    matrix[0, 1] = cos_longitude
    matrix[0, 2] = 0.0
    matrix[1, 0] = -sin_latitude * cos_longitude
    matrix[1, 1] = -sin_latitude * sin_longitude
    matrix[1, 2] = cos_latitude
    matrix[2, 0] = cos_latitude * cos_longitude
    matrix[2, 1] = cos_latitude * sin_longitude
    matrix[2, 2] = sin_latitude
    return matrix


def enu_to_ecef(vector: Vector, latitude: float, longitude: float) -> Vector:
    """Rotate a local ENU vector into ECEF coordinates."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("ENU vector must have three components")
    return ecef_to_enu_matrix(latitude, longitude).T @ vector


@jit(nopython=True, cache=True)
def j2_acceleration(position: Vector) -> Vector:
    """Point-mass plus J2 gravitational acceleration, m/s^2.

    The field is axisymmetric about the z axis, so the same expression holds in
    ECEF and ECI. No input checks: callers inside integrators guarantee a
    finite, non-zero position. Use :func:`j2_gravity_ecef` from Python code.
    """
    x = position[0]
    y = position[1]
    z = position[2]
    radius_squared = x * x + y * y + z * z
    radius = np.sqrt(radius_squared)
    z_ratio_squared = z * z / radius_squared
    correction = 1.5 * EARTH_J2 * (WGS84_SEMI_MAJOR_AXIS / radius) ** 2
    common = -EARTH_GRAVITATIONAL_PARAMETER / (radius_squared * radius)
    out = np.empty(3)
    out[0] = common * x * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared))
    out[1] = common * y * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared))
    out[2] = common * z * (1.0 + correction * (3.0 - 5.0 * z_ratio_squared))
    return out


def j2_gravity_ecef(position: Vector) -> Vector:
    """Return gravitational acceleration in ECEF axes, excluding rotation."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("ECEF position must be a finite three-vector")
    if float(position @ position) <= 0.0:
        raise ValueError("Gravity is undefined at Earth's centre")
    return j2_acceleration(position)


def earth_fixed_acceleration(
    position: Vector,
    velocity: Vector,
    *,
    include_j2: bool = True,
) -> Vector:
    """Return ECEF acceleration from gravity, Coriolis and centrifugal terms."""
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("Position and velocity must be three-vectors")

    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("Acceleration is undefined at Earth's centre")
    gravity = (
        j2_gravity_ecef(position)
        if include_j2
        else -EARTH_GRAVITATIONAL_PARAMETER * position / radius**3
    )
    earth_rate = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    coriolis = -2.0 * np.cross(earth_rate, velocity)
    centrifugal = -np.cross(earth_rate, np.cross(earth_rate, position))
    return gravity + coriolis + centrifugal


def ecef_to_eci(vector: Vector, elapsed_time: float) -> Vector:
    """Rotate an ECEF vector to the simplified equinox-aligned ECI frame."""
    return ecef_to_eci_matrix(elapsed_time) @ np.asarray(vector, dtype=np.float64)


@jit(nopython=True, cache=True)
def ecef_to_eci_matrix(elapsed_time: float) -> NDArray[np.float64]:
    """Return the direction-cosine matrix from ECEF to equinox-aligned ECI."""
    angle = EARTH_ROTATION_RATE * elapsed_time
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix = np.zeros((3, 3))
    matrix[0, 0] = cosine
    matrix[0, 1] = -sine
    matrix[1, 0] = sine
    matrix[1, 1] = cosine
    matrix[2, 2] = 1.0
    return matrix


def eci_to_ecef(vector: Vector, elapsed_time: float) -> Vector:
    """Rotate an ECI vector to the simplified equinox-aligned ECEF frame."""
    return eci_to_ecef_matrix(elapsed_time) @ np.asarray(vector, dtype=np.float64)


@jit(nopython=True, cache=True)
def eci_to_ecef_matrix(elapsed_time: float) -> NDArray[np.float64]:
    """Return the direction-cosine matrix from equinox-aligned ECI to ECEF."""
    return ecef_to_eci_matrix(-elapsed_time)
