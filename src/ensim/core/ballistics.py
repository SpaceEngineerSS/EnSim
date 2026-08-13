"""Earth-centred propagation for dragless ballistic verification cases."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .geodesy import (
    EARTH_GRAVITATIONAL_PARAMETER,
    EARTH_J2,
    EARTH_ROTATION_RATE,
    WGS84_SEMI_MAJOR_AXIS,
    ecef_to_enu_matrix,
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
)

VectorHistory = NDArray[np.float64]


@dataclass(frozen=True)
class BallisticTrajectory:
    """Time history returned by :func:`propagate_dragless_wgs84`."""

    time: NDArray[np.float64]
    eci_position: VectorHistory
    eci_velocity: VectorHistory
    latitude_deg: NDArray[np.float64]
    longitude_deg: NDArray[np.float64]
    ellipsoid_altitude_m: NDArray[np.float64]
    ned_velocity_m_s: VectorHistory


def _j2_acceleration_eci(position: NDArray[np.float64]) -> NDArray[np.float64]:
    radius_squared = float(position @ position)
    radius = np.sqrt(radius_squared)
    z_ratio_squared = position[2] ** 2 / radius_squared
    correction = 1.5 * EARTH_J2 * (WGS84_SEMI_MAJOR_AXIS / radius) ** 2
    common = -EARTH_GRAVITATIONAL_PARAMETER / radius**3
    return common * np.array(
        [
            position[0] * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared)),
            position[1] * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared)),
            position[2] * (1.0 + correction * (3.0 - 5.0 * z_ratio_squared)),
        ]
    )


def propagate_dragless_wgs84(
    *,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    ned_velocity_m_s: NDArray[np.float64] | None = None,
    duration: float,
    output_step: float = 0.1,
    rtol: float = 1e-11,
    atol: float = 1e-9,
) -> BallisticTrajectory:
    """Propagate a point mass in ECI under the axisymmetric WGS-84 J2 field.

    The initial velocity is relative to the rotating Earth and expressed in
    local North-East-Down axes. Atmospheric drag and all applied forces are
    intentionally excluded so the function can reproduce elemental NESC
    translational-equation check cases.
    """
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("Longitude must be between -180 and 180 degrees")
    if duration <= 0.0 or output_step <= 0.0:
        raise ValueError("Duration and output step must be positive")

    initial_ned_velocity = (
        np.zeros(3)
        if ned_velocity_m_s is None
        else np.asarray(
            ned_velocity_m_s,
            dtype=np.float64,
        )
    )
    if initial_ned_velocity.shape != (3,) or not np.isfinite(initial_ned_velocity).all():
        raise ValueError("NED velocity must be a finite three-vector")

    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    initial_position = geodetic_to_ecef(latitude, longitude, altitude_m)
    ecef_to_enu = ecef_to_enu_matrix(latitude, longitude)
    initial_enu_velocity = np.array(
        [initial_ned_velocity[1], initial_ned_velocity[0], -initial_ned_velocity[2]]
    )
    initial_ecef_velocity = ecef_to_enu.T @ initial_enu_velocity
    earth_rate = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    initial_eci_velocity = initial_ecef_velocity + np.cross(
        earth_rate,
        initial_position,
    )
    initial_state = np.concatenate((initial_position, initial_eci_velocity))

    def derivative(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate((state[3:6], _j2_acceleration_eci(state[0:3])))

    sample_count = int(np.floor(duration / output_step + 1e-12))
    output_times = output_step * np.arange(sample_count + 1, dtype=np.float64)
    if output_times[-1] < duration - 1e-12:
        output_times = np.append(output_times, duration)
    else:
        output_times[-1] = duration

    solution = solve_ivp(
        derivative,
        (0.0, duration),
        initial_state,
        method="DOP853",
        t_eval=output_times,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Ballistic propagation failed: {solution.message}")

    eci_position = solution.y[0:3].T
    eci_velocity = solution.y[3:6].T
    latitude_history = np.empty(len(output_times))
    longitude_history = np.empty(len(output_times))
    altitude_history = np.empty(len(output_times))
    ned_velocity_history = np.empty((len(output_times), 3))
    for index, elapsed_time in enumerate(output_times):
        position_ecef = eci_to_ecef(eci_position[index], elapsed_time)
        velocity_ecef = eci_to_ecef(
            eci_velocity[index] - np.cross(earth_rate, eci_position[index]),
            elapsed_time,
        )
        current_latitude, current_longitude, current_altitude = ecef_to_geodetic(position_ecef)
        velocity_enu = (
            ecef_to_enu_matrix(
                current_latitude,
                current_longitude,
            )
            @ velocity_ecef
        )
        latitude_history[index] = np.degrees(current_latitude)
        longitude_history[index] = np.degrees(current_longitude)
        altitude_history[index] = current_altitude
        ned_velocity_history[index] = [
            velocity_enu[1],
            velocity_enu[0],
            -velocity_enu[2],
        ]

    return BallisticTrajectory(
        time=output_times,
        eci_position=eci_position,
        eci_velocity=eci_velocity,
        latitude_deg=latitude_history,
        longitude_deg=longitude_history,
        ellipsoid_altitude_m=altitude_history,
        ned_velocity_m_s=ned_velocity_history,
    )
