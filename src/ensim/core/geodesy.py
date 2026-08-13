"""WGS-84 coordinate transforms and Earth-fixed gravitation.

The equations follow the WGS-84 ellipsoid and the J2 acceleration model used
by the NASA NESC 6-DOF verification check-case family. Angles are radians and
SI units are used throughout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

WGS84_SEMI_MAJOR_AXIS = 6_378_137.0
WGS84_FLATTENING = 1.0 / 298.257_223_563
WGS84_FIRST_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
WGS84_SEMI_MINOR_AXIS = WGS84_SEMI_MAJOR_AXIS * (1.0 - WGS84_FLATTENING)
EARTH_GRAVITATIONAL_PARAMETER = 3.986_004_418e14
EARTH_J2 = 1.082_626_68e-3
EARTH_ROTATION_RATE = 7.292_115e-5

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


def ecef_to_geodetic(position: Vector) -> tuple[float, float, float]:
    """Convert an ECEF position to WGS-84 latitude, longitude and altitude."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("ECEF position must be a finite three-vector")

    x, y, z = position
    horizontal_radius = float(np.hypot(x, y))
    longitude = float(np.arctan2(y, x))
    if horizontal_radius < 1e-9:
        if abs(z) < 1e-9:
            raise ValueError("Geodetic coordinates are undefined at Earth's centre")
        latitude = float(np.copysign(np.pi / 2.0, z))
        return latitude, longitude, float(abs(z) - WGS84_SEMI_MINOR_AXIS)

    latitude = float(
        np.arctan2(
            z,
            horizontal_radius * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED),
        )
    )
    altitude = 0.0
    for _ in range(10):
        sin_latitude = np.sin(latitude)
        radius_prime_vertical = WGS84_SEMI_MAJOR_AXIS / np.sqrt(
            1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_latitude**2
        )
        cos_latitude = np.cos(latitude)
        if abs(cos_latitude) > 1e-10:
            altitude = horizontal_radius / cos_latitude - radius_prime_vertical
        else:
            altitude = abs(z) - WGS84_SEMI_MINOR_AXIS
        updated = float(
            np.arctan2(
                z,
                horizontal_radius
                * (
                    1.0
                    - WGS84_FIRST_ECCENTRICITY_SQUARED
                    * radius_prime_vertical
                    / (radius_prime_vertical + altitude)
                ),
            )
        )
        if abs(updated - latitude) < 1e-13:
            latitude = updated
            break
        latitude = updated

    sin_latitude = np.sin(latitude)
    radius_prime_vertical = WGS84_SEMI_MAJOR_AXIS / np.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_latitude**2
    )
    altitude = horizontal_radius / np.cos(latitude) - radius_prime_vertical
    return latitude, longitude, float(altitude)


def ecef_to_enu_matrix(latitude: float, longitude: float) -> NDArray[np.float64]:
    """Return the direction-cosine matrix that rotates ECEF vectors to ENU."""
    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    sin_longitude = np.sin(longitude)
    cos_longitude = np.cos(longitude)
    return np.array(
        [
            [-sin_longitude, cos_longitude, 0.0],
            [
                -sin_latitude * cos_longitude,
                -sin_latitude * sin_longitude,
                cos_latitude,
            ],
            [
                cos_latitude * cos_longitude,
                cos_latitude * sin_longitude,
                sin_latitude,
            ],
        ],
        dtype=np.float64,
    )


def enu_to_ecef(vector: Vector, latitude: float, longitude: float) -> Vector:
    """Rotate a local ENU vector into ECEF coordinates."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("ENU vector must have three components")
    return ecef_to_enu_matrix(latitude, longitude).T @ vector


def j2_gravity_ecef(position: Vector) -> Vector:
    """Return gravitational acceleration in ECEF axes, excluding rotation."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("ECEF position must be a finite three-vector")
    x, y, z = position
    radius_squared = float(position @ position)
    if radius_squared <= 0.0:
        raise ValueError("Gravity is undefined at Earth's centre")
    radius = np.sqrt(radius_squared)
    z_ratio_squared = z * z / radius_squared
    correction = 1.5 * EARTH_J2 * (WGS84_SEMI_MAJOR_AXIS / radius) ** 2
    common = -EARTH_GRAVITATIONAL_PARAMETER / radius**3
    return common * np.array(
        [
            x * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared)),
            y * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared)),
            z * (1.0 + correction * (3.0 - 5.0 * z_ratio_squared)),
        ],
        dtype=np.float64,
    )


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


def ecef_to_eci_matrix(elapsed_time: float) -> NDArray[np.float64]:
    """Return the direction-cosine matrix from ECEF to equinox-aligned ECI."""
    angle = EARTH_ROTATION_RATE * elapsed_time
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(vector: Vector, elapsed_time: float) -> Vector:
    """Rotate an ECI vector to the simplified equinox-aligned ECEF frame."""
    return eci_to_ecef_matrix(elapsed_time) @ np.asarray(vector, dtype=np.float64)


def eci_to_ecef_matrix(elapsed_time: float) -> NDArray[np.float64]:
    """Return the direction-cosine matrix from equinox-aligned ECI to ECEF."""
    return ecef_to_eci_matrix(-elapsed_time)
