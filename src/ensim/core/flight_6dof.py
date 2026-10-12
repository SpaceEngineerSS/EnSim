"""
Six-degree-of-freedom rigid-body flight simulation.

State vector (14 elements)::

    [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m_prop]

    position        x, y, z     m      inertial frame (local ENU, or ECI in WGS-84 mode)
    velocity        vx, vy, vz  m/s    inertial frame
    attitude        q0..q3             unit quaternion, body -> inertial, scalar first
    angular rate    wx, wy, wz  rad/s  body frame
    propellant      m_prop      kg

Equations of motion::

    dr/dt = v
    dv/dt = R(q) (F_thrust + F_aero) / m + g(r)
    dq/dt = 1/2 q (x) [0, w]
    dw/dt = I^-1 (M - w x I w)
    dm/dt = -mdot while burning

Every Runge-Kutta stage re-evaluates thrust, aerodynamics, mass, inertia and
gravity from that stage's state, so the integrators keep their nominal order.

Body axes: x forward through the nose, y and z complete a right-handed set.
Positions along the airframe (CG, CP) are measured aft from the nose tip.

References:
    Stevens, Lewis & Johnson, "Aircraft Control and Simulation", 3rd ed.
    Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics", 3rd ed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ensim.core.aero import calculate_total_cp
from ensim.core.constants import EARTH_RADIUS_MEAN, EARTH_ROTATION_RATE, G0
from ensim.core.geodesy import (
    ecef_to_eci,
    ecef_to_eci_matrix,
    ecef_to_enu_matrix,
    eci_to_ecef,
    eci_to_ecef_matrix,
    geodetic_from_ecef,
    geodetic_to_ecef,
    j2_acceleration,
)
from ensim.core.integrators import (
    DP5_A,
    DP5_B5,
    DP5_E,
    compute_new_step_size,
    rk45_error_norm,
)
from ensim.core.math_utils import (
    cross_product,
    q_conjugate,
    q_derivative,
    q_from_dcm,
    q_from_euler,
    q_normalize,
    q_rotate_vector,
    q_to_euler,
    q_to_rotation_matrix,
)
from ensim.core.mission import atmosphere_properties
from ensim.core.rocket import Rocket
from ensim.utils.numba_support import jit

R_EARTH = EARTH_RADIUS_MEAN

__all__ = [
    "G0",
    "R_EARTH",
    "FlightResult6DOF",
    "PhysicsViolationError",
    "SimulationCancelled",
    "calculate_aero_angles",
    "calculate_aerodynamic_moment",
    "calculate_gravity_vector",
    "calculate_inertia_principal",
    "calculate_inertia_tensor_cylindrical",
    "derivatives_6dof",
    "rk4_step_6dof",
    "rk45_step_rocket",
    "simulate_flight_6dof",
]


class PhysicsViolationError(Exception):
    """Raised when the inputs or the integrated state are not physical."""


class SimulationCancelled(Exception):
    """Raised when the caller's cancel callback asks the simulation to stop."""


@dataclass
class FlightResult6DOF:
    """Time histories and summary values of one 6-DOF run.

    Positions and velocities are reported in the local ENU frame of the launch
    site in both integration modes.
    """

    time: np.ndarray  # s

    position_x: np.ndarray  # m, east
    position_y: np.ndarray  # m, north
    position_z: np.ndarray  # m, up (height above the launch site)

    velocity_x: np.ndarray  # m/s
    velocity_y: np.ndarray  # m/s
    velocity_z: np.ndarray  # m/s
    velocity_magnitude: np.ndarray  # m/s

    quaternion_w: np.ndarray  # body -> local ENU
    quaternion_x: np.ndarray
    quaternion_y: np.ndarray
    quaternion_z: np.ndarray

    roll: np.ndarray  # deg
    pitch: np.ndarray  # deg
    yaw: np.ndarray  # deg

    omega_x: np.ndarray  # rad/s, body frame
    omega_y: np.ndarray  # rad/s
    omega_z: np.ndarray  # rad/s

    thrust: np.ndarray  # N
    drag: np.ndarray  # N, airframe plus parachute
    gravity_force: np.ndarray  # N, magnitude of m*g

    mach: np.ndarray  # -
    dynamic_pressure: np.ndarray  # Pa

    mass: np.ndarray  # kg, total
    propellant_mass: np.ndarray  # kg, state[13]

    isp: np.ndarray  # s, delivered at local ambient pressure
    thrust_loss_factor: np.ndarray  # -, 1.0 while burning: separation losses are not modelled
    flow_regime: (
        np.ndarray
    )  # -1 off/unknown, 0 matched, 1 over-, 2 strongly over-, 3 under-expanded

    angle_of_attack: np.ndarray  # deg
    sideslip_angle: np.ndarray  # deg
    stability_margin: np.ndarray  # calibers

    geodetic_latitude: np.ndarray  # deg, NaN in local mode
    geodetic_longitude: np.ndarray  # deg, NaN in local mode
    geodetic_altitude: np.ndarray  # m above the ellipsoid, NaN in local mode

    liftoff_time: float = 0.0  # s
    burnout_time: float = 0.0  # s
    burnout_altitude: float = 0.0  # m
    burnout_velocity: float = 0.0  # m/s
    apogee_time: float = 0.0  # s
    apogee_altitude: float = 0.0  # m
    max_velocity: float = 0.0  # m/s
    max_mach: float = 0.0  # -
    max_acceleration: float = 0.0  # g, sensed (non-gravitational) acceleration
    max_q: float = 0.0  # Pa
    max_alpha: float = 0.0  # deg, total angle of attack while powered or coasting up
    flight_time: float = 0.0  # s

    success: bool = True
    abort_reason: str | None = None
    apogee_reached: bool = False
    ground_impact: bool = False
    termination_reason: str = "maximum_time"
    reference_frame: str = "local_enu"
    launch_latitude_deg: float | None = None
    launch_longitude_deg: float | None = None
    launch_altitude_m: float = 0.0
    parachute_deploy_time: float | None = None  # s, main parachute
    drogue_deploy_time: float | None = None  # s
    landing_x: float | None = None  # m east of the pad, None without ground impact
    landing_y: float | None = None  # m north of the pad
    forced_min_steps: int = 0  # adaptive steps accepted at h_min above tolerance


# Mass properties, gravity and aerodynamic primitives


@jit(nopython=True, cache=True)
def calculate_inertia_tensor_cylindrical(mass: float, length: float, radius: float) -> np.ndarray:
    """Inertia tensor of a uniform solid cylinder whose axis is body x, kg m^2."""
    inertia = np.zeros((3, 3))
    inertia[0, 0] = 0.5 * mass * radius * radius
    inertia[1, 1] = mass * (3.0 * radius * radius + length * length) / 12.0
    inertia[2, 2] = inertia[1, 1]
    return inertia


@jit(nopython=True, cache=True)
def calculate_inertia_principal(mass: float, length: float, radius: float) -> np.ndarray:
    """Principal moments [Ixx, Iyy, Izz] of a uniform solid cylinder, kg m^2."""
    ixx = 0.5 * mass * radius * radius
    iyy = mass * (3.0 * radius * radius + length * length) / 12.0
    return np.array([ixx, iyy, iyy])


@jit(nopython=True, cache=True)
def calculate_gravity_vector(altitude: float) -> np.ndarray:
    """Inverse-square gravity in local ENU axes, m/s^2.

    g(h) = g0 (Re / (Re + h))^2, pointing down. Negative altitudes are treated
    as sea level.
    """
    g_magnitude = G0 * (EARTH_RADIUS_MEAN / (EARTH_RADIUS_MEAN + max(0.0, altitude))) ** 2
    return np.array([0.0, 0.0, -g_magnitude])


@jit(nopython=True, cache=True)
def calculate_aero_angles(velocity_body: np.ndarray) -> tuple[float, float]:
    """Angle of attack and sideslip (rad) from the air-relative body velocity."""
    vx = velocity_body[0]
    vy = velocity_body[1]
    vz = velocity_body[2]
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-6:
        return 0.0, 0.0
    alpha = np.arctan2(-vz, vx)
    sin_beta = min(1.0, max(-1.0, vy / speed))
    return alpha, np.arcsin(sin_beta)


@jit(nopython=True, cache=True)
def calculate_aerodynamic_moment(
    cp_position: float, cg_position: float, force_aero_body: np.ndarray
) -> np.ndarray:
    """Moment about the CG of a force acting at the CP, N m, body frame.

    Both positions are measured aft from the nose tip while body x points
    forward, so the lever arm from CG to CP is (cg - cp) along x. With the CP
    aft of the CG the moment turns the nose into the relative wind.
    """
    r_arm = np.array([cg_position - cp_position, 0.0, 0.0])
    return cross_product(r_arm, force_aero_body)


# Constant-load primitives. These integrate one step with the applied forces
# held fixed and are kept for verification cases (NESC tumbling brick) and for
# users who drive their own force model. simulate_flight_6dof does not use them.


@jit(nopython=True, cache=True)
def derivatives_6dof(
    state: np.ndarray,
    mass: float,
    I_principal: np.ndarray,
    F_thrust_body: np.ndarray,
    F_aero_body: np.ndarray,
    M_aero: np.ndarray,
    M_thrust: np.ndarray,
    altitude: float,
    use_earth_model: bool = False,
) -> np.ndarray:
    """State derivative for given body-frame loads.

    ``use_earth_model`` switches gravity from the flat-Earth inverse-square law
    to point mass plus J2 evaluated at ``state[0:3]`` in ECI. The propellant
    rate (element 13) is left at zero for the caller to set.
    """
    deriv = np.zeros(14)
    if mass <= 0.0:
        return deriv

    q = state[6:10].copy()
    omega = state[10:13].copy()

    g = j2_acceleration(state[0:3]) if use_earth_model else calculate_gravity_vector(altitude)

    force_inertial = q_rotate_vector(q, F_thrust_body + F_aero_body)
    q_dot = q_derivative(q, omega)

    I_omega = I_principal * omega
    gyro = cross_product(omega, I_omega)
    moment = M_aero + M_thrust

    deriv[0:3] = state[3:6]
    deriv[3:6] = force_inertial / mass + g
    deriv[6:10] = q_dot
    for axis in range(3):
        if I_principal[axis] > 1e-12:
            deriv[10 + axis] = (moment[axis] - gyro[axis]) / I_principal[axis]
    return deriv


@jit(nopython=True, cache=True)
def _stage_mass(state: np.ndarray, dry_mass: float) -> float:
    return max(dry_mass + max(state[13], 0.0), 1e-12)


@jit(nopython=True, cache=True)
def rk4_step_6dof(
    state: np.ndarray,
    dt: float,
    mass: float,
    I_principal: np.ndarray,
    F_thrust_body: np.ndarray,
    F_aero_body: np.ndarray,
    M_aero: np.ndarray,
    M_thrust: np.ndarray,
    altitude: float,
    mdot: float = 0.0,
    use_earth_model: bool = False,
) -> np.ndarray:
    """One classical RK4 step with constant loads and per-stage mass."""
    dry_mass = mass - max(state[13], 0.0)

    k1 = derivatives_6dof(
        state,
        mass,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        altitude,
        use_earth_model,
    )
    k1[13] = -mdot
    s2 = state + 0.5 * dt * k1
    k2 = derivatives_6dof(
        s2,
        _stage_mass(s2, dry_mass),
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        max(0.0, s2[2]),
        use_earth_model,
    )
    k2[13] = -mdot
    s3 = state + 0.5 * dt * k2
    k3 = derivatives_6dof(
        s3,
        _stage_mass(s3, dry_mass),
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        max(0.0, s3[2]),
        use_earth_model,
    )
    k3[13] = -mdot
    s4 = state + dt * k3
    k4 = derivatives_6dof(
        s4,
        _stage_mass(s4, dry_mass),
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        max(0.0, s4[2]),
        use_earth_model,
    )
    k4[13] = -mdot

    out = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out[6:10] = q_normalize(out[6:10].copy())
    if out[13] < 0.0:
        out[13] = 0.0
    return out


@jit(nopython=True, cache=True)
def rk45_step_rocket(
    state: np.ndarray,
    h: float,
    mass: float,
    I_principal: np.ndarray,
    F_thrust_body: np.ndarray,
    F_aero_body: np.ndarray,
    M_aero: np.ndarray,
    M_thrust: np.ndarray,
    altitude: float,
    mdot: float = 0.0,
    use_earth_model: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Dormand-Prince 5(4) step with constant loads.

    Returns the fifth-order state, the embedded error estimate and the
    derivative at the new state.
    """
    k = np.zeros((7, 14))
    dry_mass = mass - max(state[13], 0.0)
    k[0] = derivatives_6dof(
        state,
        mass,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        altitude,
        use_earth_model,
    )
    k[0, 13] = -mdot
    for stage in range(1, 6):
        stage_state = state.copy()
        for j in range(stage):
            stage_state += h * DP5_A[stage, j] * k[j]
        k[stage] = derivatives_6dof(
            stage_state,
            _stage_mass(stage_state, dry_mass),
            I_principal,
            F_thrust_body,
            F_aero_body,
            M_aero,
            M_thrust,
            max(0.0, stage_state[2]),
            use_earth_model,
        )
        k[stage, 13] = -mdot

    y_new = state.copy()
    for j in range(6):
        y_new += h * DP5_B5[j] * k[j]
    k[6] = derivatives_6dof(
        y_new,
        _stage_mass(y_new, dry_mass),
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        max(0.0, y_new[2]),
        use_earth_model,
    )
    k[6, 13] = -mdot

    y_error = np.zeros(14)
    for j in range(7):
        y_error += h * DP5_E[j] * k[j]

    y_new[6:10] = q_normalize(y_new[6:10].copy())
    if y_new[13] < 0.0:
        y_new[13] = 0.0
    return y_new, y_error, k[6]


# Full-state flight kernel.
#
# All vehicle and environment parameters travel in one float vector so the
# Numba-compiled right-hand side has a fixed signature. Indices:

_P_THRUST_VAC = 0  # throttled vacuum thrust, N
_P_EXIT_AREA = 1  # m^2
_P_BURN_TIME = 2  # s
_P_MDOT = 3  # kg/s
_P_CD = 4  # axial drag coefficient incl. multiplier
_P_AREF = 5  # m^2
_P_CN_ALPHA = 6  # 1/rad, whole vehicle
_P_CP = 7  # m from nose
_P_DRY_MASS = 8  # kg
_P_DRY_MOMENT = 9  # kg m, first moment of dry mass about the nose tip
_P_X_PROP = 10  # m, propellant CG from nose
_P_LENGTH = 11  # m
_P_RADIUS = 12  # m
_P_WIND_SPEED = 13  # m/s at the reference height
_P_WIND_TO = 14  # rad, direction the wind blows toward, clockwise from north
_P_WIND_TOP = 15  # m, height above which the wind no longer grows
_P_CHUTE_CDA = 16  # m^2, zero until deployment
_P_EARTH = 17  # 1.0 for WGS-84/ECI mode
_P_LAUNCH_ALT = 18  # m above the ellipsoid
_P_CN_FINS = 19  # 1/rad, fin set
_P_CP_FINS = 20  # m from nose
_P_FIN_DELTA = 21  # rad, fin incidence error in the body x-z plane
_P_EXIT_PRESSURE = 22  # Pa, <= 0 when unknown
_P_REF_DIAMETER = 23  # m
_P_SIZE = 24

_WIND_REFERENCE_HEIGHT = 10.0  # m, anemometer height for the surface wind
_WIND_SHEAR_EXPONENT = 1.0 / 7.0  # neutral-stability power law

# Diagnostic vector layout, one entry per output sample
_D_THRUST = 0
_D_DRAG = 1
_D_GRAVITY = 2
_D_MACH = 3
_D_QBAR = 4
_D_MASS = 5
_D_ISP = 6
_D_REGIME = 7
_D_ALPHA = 8
_D_BETA = 9
_D_MARGIN = 10
_D_ACCEL = 11
_D_HEIGHT = 12
_D_LAT = 13
_D_LON = 14
_D_GEOD_ALT = 15
_D_POS = 16  # 3 entries, ENU
_D_VEL = 19  # 3 entries, ENU
_D_QUAT = 22  # 4 entries, body -> ENU
_D_SIZE = 26


@jit(nopython=True, cache=True)
def _flight_evaluate(
    t: float,
    y: np.ndarray,
    p: np.ndarray,
    launch_ecef: np.ndarray,
    launch_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-hand side and diagnostics of the full flight model at (t, y)."""
    deriv = np.zeros(14)
    diag = np.zeros(_D_SIZE)
    earth_mode = p[_P_EARTH] > 0.5

    r = y[0:3]
    v = y[3:6]
    q = y[6:10].copy()
    omega = y[10:13].copy()
    m_prop = max(y[13], 0.0)

    # Local frame, height and air-relative velocity
    if earth_mode:
        c_i2e = eci_to_ecef_matrix(t)
        r_ecef = c_i2e @ r
        lat, lon, h_geod = geodetic_from_ecef(r_ecef)
        c_enu = ecef_to_enu_matrix(lat, lon)
        height = h_geod - p[_P_LAUNCH_ALT]
        air_altitude = max(0.0, h_geod)
        earth_rate = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
        v_ecef = c_i2e @ (v - cross_product(earth_rate, r))
        pos_enu = launch_c @ (r_ecef - launch_ecef)
        vel_enu = c_enu @ v_ecef
        g = j2_acceleration(r)
    else:
        lat = 0.0
        lon = 0.0
        h_geod = 0.0
        height = r[2]
        air_altitude = max(0.0, height)
        pos_enu = r.copy()
        vel_enu = v.copy()
        g = calculate_gravity_vector(air_altitude)

    wind_height = min(max(height, 0.0), p[_P_WIND_TOP])
    if wind_height > 0.0:
        wind_speed = (
            p[_P_WIND_SPEED] * (wind_height / _WIND_REFERENCE_HEIGHT) ** _WIND_SHEAR_EXPONENT
        )
    else:
        wind_speed = 0.0
    wind_enu = np.array(
        [wind_speed * np.sin(p[_P_WIND_TO]), wind_speed * np.cos(p[_P_WIND_TO]), 0.0]
    )
    if earth_mode:
        wind_inertial = ecef_to_eci_matrix(t) @ (c_enu.T @ wind_enu)
        v_rel = v - cross_product(np.array([0.0, 0.0, EARTH_ROTATION_RATE]), r) - wind_inertial
    else:
        v_rel = v - wind_enu
    speed_rel = np.sqrt(v_rel[0] ** 2 + v_rel[1] ** 2 + v_rel[2] ** 2)

    _, pressure, density, sound_speed = atmosphere_properties(air_altitude)
    qbar = 0.5 * density * speed_rel * speed_rel

    # Mass properties
    mass = p[_P_DRY_MASS] + m_prop
    cg = (p[_P_DRY_MOMENT] + m_prop * p[_P_X_PROP]) / mass
    inertia = calculate_inertia_principal(mass, p[_P_LENGTH], p[_P_RADIUS])

    # Propulsion
    burning = t < p[_P_BURN_TIME] and y[13] > 0.0
    thrust = 0.0
    regime = -1.0
    if burning:
        thrust = max(0.0, p[_P_THRUST_VAC] - pressure * p[_P_EXIT_AREA])
        if p[_P_EXIT_PRESSURE] > 0.0 and pressure > 0.0:
            ratio = p[_P_EXIT_PRESSURE] / pressure
            if ratio < 0.4:
                regime = 2.0
            elif ratio < 1.0:
                regime = 1.0
            elif ratio > 1.05:
                regime = 3.0
            else:
                regime = 0.0
    force_body = np.array([thrust, 0.0, 0.0])

    # Aerodynamics
    v_body = q_rotate_vector(q_conjugate(q), v_rel)
    alpha, beta = calculate_aero_angles(v_body)
    drag = 0.0
    moment_body = np.zeros(3)
    if speed_rel > 1e-6:
        unit = v_body / speed_rel
        drag = qbar * p[_P_AREF] * (p[_P_CD] + p[_P_CHUTE_CDA] / p[_P_AREF])
        force_body -= drag * unit
        # Normal force from the crossflow: |F| = qbar A CN_alpha sin(total alpha)
        normal = np.array([0.0, -unit[1], -unit[2]]) * (qbar * p[_P_AREF] * p[_P_CN_ALPHA])
        force_body += normal
        moment_body += calculate_aerodynamic_moment(p[_P_CP], cg, normal)
        if p[_P_FIN_DELTA] != 0.0:
            fin_force = np.array([0.0, 0.0, -qbar * p[_P_AREF] * p[_P_CN_FINS] * p[_P_FIN_DELTA]])
            force_body += fin_force
            moment_body += calculate_aerodynamic_moment(p[_P_CP_FINS], cg, fin_force)

    # Derivatives
    deriv[0:3] = v
    deriv[3:6] = q_rotate_vector(q, force_body) / mass + g
    deriv[6:10] = q_derivative(q, omega)
    gyro = cross_product(omega, inertia * omega)
    for axis in range(3):
        deriv[10 + axis] = (moment_body[axis] - gyro[axis]) / inertia[axis]
    deriv[13] = -p[_P_MDOT] if burning else 0.0

    # Diagnostics
    diag[_D_THRUST] = thrust
    diag[_D_DRAG] = drag
    diag[_D_GRAVITY] = mass * np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    diag[_D_MACH] = speed_rel / sound_speed
    diag[_D_QBAR] = qbar
    diag[_D_MASS] = mass
    diag[_D_ISP] = thrust / (p[_P_MDOT] * G0) if burning and p[_P_MDOT] > 0.0 else 0.0
    diag[_D_REGIME] = regime
    diag[_D_ALPHA] = np.degrees(alpha)
    diag[_D_BETA] = np.degrees(beta)
    diag[_D_MARGIN] = (p[_P_CP] - cg) / p[_P_REF_DIAMETER]
    diag[_D_ACCEL] = np.sqrt(force_body[0] ** 2 + force_body[1] ** 2 + force_body[2] ** 2) / (
        mass * G0
    )
    diag[_D_HEIGHT] = height
    diag[_D_LAT] = np.degrees(lat)
    diag[_D_LON] = np.degrees(lon)
    diag[_D_GEOD_ALT] = h_geod
    diag[_D_POS : _D_POS + 3] = pos_enu
    diag[_D_VEL : _D_VEL + 3] = vel_enu
    if earth_mode:
        body_to_enu = c_enu @ eci_to_ecef_matrix(t) @ q_to_rotation_matrix(q)
        diag[_D_QUAT : _D_QUAT + 4] = q_from_dcm(body_to_enu)
    else:
        diag[_D_QUAT : _D_QUAT + 4] = q
    return deriv, diag


@jit(nopython=True, cache=True)
def _flight_rhs(
    t: float, y: np.ndarray, p: np.ndarray, launch_ecef: np.ndarray, launch_c: np.ndarray
) -> np.ndarray:
    deriv, _ = _flight_evaluate(t, y, p, launch_ecef, launch_c)
    return deriv


@jit(nopython=True, cache=True)
def _finish_state(y: np.ndarray) -> np.ndarray:
    y[6:10] = q_normalize(y[6:10].copy())
    if y[13] < 0.0:
        y[13] = 0.0
    return y


@jit(nopython=True, cache=True)
def _flight_rk4(
    t: float, y: np.ndarray, h: float, p: np.ndarray, launch_ecef: np.ndarray, launch_c: np.ndarray
) -> np.ndarray:
    k1 = _flight_rhs(t, y, p, launch_ecef, launch_c)
    k2 = _flight_rhs(t + 0.5 * h, y + 0.5 * h * k1, p, launch_ecef, launch_c)
    k3 = _flight_rhs(t + 0.5 * h, y + 0.5 * h * k2, p, launch_ecef, launch_c)
    k4 = _flight_rhs(t + h, y + h * k3, p, launch_ecef, launch_c)
    return _finish_state(y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


_DP5_C = np.array([0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0])


@jit(nopython=True, cache=True)
def _flight_dp45(
    t: float,
    y: np.ndarray,
    k_first: np.ndarray,
    h: float,
    p: np.ndarray,
    launch_ecef: np.ndarray,
    launch_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dormand-Prince 5(4) step reusing the FSAL derivative ``k_first``."""
    k = np.zeros((7, 14))
    k[0] = k_first
    for stage in range(1, 6):
        stage_state = y.copy()
        for j in range(stage):
            stage_state += h * DP5_A[stage, j] * k[j]
        k[stage] = _flight_rhs(t + _DP5_C[stage] * h, stage_state, p, launch_ecef, launch_c)
    y_new = y.copy()
    for j in range(6):
        y_new += h * DP5_B5[j] * k[j]
    k[6] = _flight_rhs(t + h, y_new, p, launch_ecef, launch_c)
    y_error = np.zeros(14)
    for j in range(7):
        y_error += h * DP5_E[j] * k[j]
    return _finish_state(y_new), y_error, k[6]


@jit(nopython=True, cache=True)
def _hermite(
    y0: np.ndarray, f0: np.ndarray, y1: np.ndarray, f1: np.ndarray, h: float, s: float
) -> np.ndarray:
    """Cubic Hermite interpolation at fraction s of a step of length h."""
    s2 = s * s
    s3 = s2 * s
    out = (
        (2.0 * s3 - 3.0 * s2 + 1.0) * y0
        + (s3 - 2.0 * s2 + s) * h * f0
        + (-2.0 * s3 + 3.0 * s2) * y1
        + (s3 - s2) * h * f1
    )
    return _finish_state(out)


class _History:
    """Growable column store for output samples."""

    def __init__(self, capacity: int):
        self.n = 0
        self.t = np.zeros(capacity)
        self.y = np.zeros((capacity, 14))
        self.d = np.zeros((capacity, _D_SIZE))

    def append(self, t: float, y: np.ndarray, d: np.ndarray) -> None:
        if self.n == self.t.size:
            grow = max(16, self.t.size // 2)
            self.t = np.concatenate((self.t, np.zeros(grow)))
            self.y = np.vstack((self.y, np.zeros((grow, 14))))
            self.d = np.vstack((self.d, np.zeros((grow, _D_SIZE))))
        self.t[self.n] = t
        self.y[self.n] = y
        self.d[self.n] = d
        self.n += 1


def _pack_parameters(
    rocket: Rocket,
    *,
    thrust_vac: float,
    throttle: float,
    exit_area: float,
    burn_time: float,
    mdot: float,
    cd_factor: float,
    wind_speed: float,
    wind_direction_deg: float,
    wind_profile_top_m: float,
    use_wgs84: bool,
    launch_altitude_m: float,
    fin_misalignment_deg: float,
    nozzle_exit_pressure: float | None,
) -> np.ndarray:
    cp_total, cn_total, components = calculate_total_cp(rocket)
    p = np.zeros(_P_SIZE)
    p[_P_THRUST_VAC] = thrust_vac * throttle
    p[_P_EXIT_AREA] = exit_area
    p[_P_BURN_TIME] = burn_time
    p[_P_MDOT] = mdot
    p[_P_CD] = rocket.axial_drag_coefficient * cd_factor
    p[_P_AREF] = rocket.reference_area
    p[_P_CN_ALPHA] = cn_total
    p[_P_CP] = cp_total
    p[_P_DRY_MASS] = rocket.dry_mass
    p[_P_DRY_MOMENT] = rocket.dry_mass * rocket.get_cg_from_propellant(0.0)
    p[_P_X_PROP] = rocket.engine.position
    p[_P_LENGTH] = rocket.total_length
    p[_P_RADIUS] = 0.5 * rocket.reference_diameter
    p[_P_WIND_SPEED] = wind_speed
    p[_P_WIND_TO] = np.radians(wind_direction_deg + 180.0)
    p[_P_WIND_TOP] = wind_profile_top_m
    p[_P_EARTH] = 1.0 if use_wgs84 else 0.0
    p[_P_LAUNCH_ALT] = launch_altitude_m if use_wgs84 else 0.0
    p[_P_CN_FINS] = components["cn_fins"]
    p[_P_CP_FINS] = components["cp_fins"]
    p[_P_FIN_DELTA] = np.radians(fin_misalignment_deg)
    p[_P_EXIT_PRESSURE] = nozzle_exit_pressure if nozzle_exit_pressure else 0.0
    p[_P_REF_DIAMETER] = rocket.reference_diameter
    return p


def simulate_flight_6dof(
    rocket: Rocket,
    thrust_vac: float,
    isp_vac: float,
    burn_time: float,
    exit_area: float = 0.01,
    dt: float = 0.01,
    max_time: float = 300.0,
    launch_angle_deg: float = 85.0,
    launch_azimuth_deg: float = 0.0,
    rail_length: float = 3.0,
    wind_speed: float = 0.0,
    wind_direction_deg: float = 0.0,
    use_adaptive: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    h_min: float = 1e-5,
    h_max: float = 1.0,
    output_dt: float = 0.01,
    throttle: float = 1.0,
    cd_factor: float = 1.0,
    fin_misalignment_deg: float = 0.0,
    chute_diameter: float = 0.0,
    deploy_at_apogee: bool = True,
    nozzle_exit_pressure: float | None = None,
    use_wgs84: bool = False,
    launch_latitude_deg: float = 0.0,
    launch_longitude_deg: float = 0.0,
    launch_altitude_m: float = 0.0,
    chute_cd: float = 1.5,
    main_deploy_altitude_m: float = 300.0,
    wind_profile_top_m: float = 1000.0,
    cancel: Callable[[], bool] | None = None,
    drogue_diameter: float = 0.0,
    drogue_cd: float = 1.2,
) -> FlightResult6DOF:
    """Simulate a single-stage flight with 6-DOF rigid-body dynamics.

    Args:
        rocket: Vehicle definition. It is not modified.
        thrust_vac: Vacuum thrust, N.
        isp_vac: Vacuum specific impulse, s.
        burn_time: Burn duration, s. The engine also stops when the
            propellant is exhausted.
        exit_area: Nozzle exit area, m^2, for the ambient-pressure thrust loss.
        dt: Step for the fixed-step RK4 integrator, s.
        max_time: Simulation end time, s.
        launch_angle_deg: Rail elevation above the horizon (90 = vertical).
        launch_azimuth_deg: Rail heading, clockwise from north.
        rail_length: Rail length, m. Attitude is held until the vehicle
            clears it.
        wind_speed: Surface wind at 10 m, m/s.
        wind_direction_deg: Direction the wind blows from, clockwise from north.
        use_adaptive: Use Dormand-Prince 5(4) with error control instead of RK4.
        rtol, atol: Error tolerances for the adaptive integrator.
        h_min, h_max: Step bounds for the adaptive integrator, s.
        output_dt: Spacing of output samples, s. Adaptive steps are not limited
            by it; samples are interpolated from the step polynomial.
        throttle: Thrust and mass-flow multiplier in [0, 1].
        cd_factor: Multiplier on the vehicle drag coefficient.
        fin_misalignment_deg: Fin incidence error in the pitch plane. Produces a
            normal force and a trim moment proportional to dynamic pressure.
        chute_diameter: Recovery parachute diameter, m (0 = no parachute).
        deploy_at_apogee: Deploy at apogee; otherwise deploy after apogee once
            below ``main_deploy_altitude_m``.
        nozzle_exit_pressure: Nozzle exit pressure, Pa, for the
            expansion-regime label. Does not change thrust.
        use_wgs84: Integrate in ECI with WGS-84 J2 gravity and Earth rotation.
        launch_latitude_deg, launch_longitude_deg, launch_altitude_m:
            Launch site for WGS-84 mode.
        chute_cd: Main parachute drag coefficient on its nominal area.
        drogue_diameter: Drogue diameter, m (0 = none). With a drogue the
            drogue opens at apogee and the main at ``main_deploy_altitude_m``;
            both then stay attached.
        drogue_cd: Drogue drag coefficient.
        main_deploy_altitude_m: Height above the pad for a delayed deployment.
        wind_profile_top_m: The 1/7 power-law wind profile grows up to this
            height and stays constant above it.
        cancel: Optional callable polled once per step; return True to stop.

    Raises:
        PhysicsViolationError: Invalid inputs or a non-finite state.
        SimulationCancelled: ``cancel`` returned True.
    """
    if thrust_vac < 0.0:
        raise PhysicsViolationError(f"Negative thrust: {thrust_vac} N")
    if isp_vac <= 0.0:
        raise PhysicsViolationError(f"Invalid Isp: {isp_vac} s")
    if rocket.wet_mass <= 0.0 or rocket.dry_mass <= 0.0:
        raise PhysicsViolationError(f"Invalid mass: {rocket.wet_mass} kg")
    if burn_time < 0.0 or dt <= 0.0 or max_time <= 0.0 or output_dt <= 0.0:
        raise PhysicsViolationError("Burn time must be nonnegative and time steps positive")
    if exit_area < 0.0 or rail_length < 0.0 or wind_speed < 0.0 or chute_diameter < 0.0:
        raise PhysicsViolationError("Geometry and environmental magnitudes cannot be negative")
    if not 0.0 <= throttle <= 1.0:
        raise PhysicsViolationError("Throttle must be between zero and one")
    if drogue_diameter < 0.0 or drogue_cd <= 0.0:
        raise PhysicsViolationError("Drogue diameter and drag coefficient are invalid")
    if rocket.axial_drag_coefficient < 0.0 or cd_factor <= 0.0 or chute_cd <= 0.0:
        raise PhysicsViolationError("Drag coefficients must be positive")
    if use_adaptive and not 0.0 < h_min <= h_max:
        raise PhysicsViolationError("Adaptive step bounds must satisfy 0 < h_min <= h_max")
    if use_wgs84 and not -90.0 <= launch_latitude_deg <= 90.0:
        raise PhysicsViolationError("Launch latitude must be between -90 and 90 degrees")
    if use_wgs84 and not -180.0 <= launch_longitude_deg <= 180.0:
        raise PhysicsViolationError("Launch longitude must be between -180 and 180 degrees")
    if use_wgs84 and not np.isfinite(launch_altitude_m):
        raise PhysicsViolationError("Launch altitude must be finite")

    mdot = thrust_vac * throttle / (isp_vac * G0)
    p = _pack_parameters(
        rocket,
        thrust_vac=thrust_vac,
        throttle=throttle,
        exit_area=exit_area,
        burn_time=burn_time,
        mdot=mdot,
        cd_factor=cd_factor,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction_deg,
        wind_profile_top_m=wind_profile_top_m,
        use_wgs84=use_wgs84,
        launch_altitude_m=launch_altitude_m,
        fin_misalignment_deg=fin_misalignment_deg,
        nozzle_exit_pressure=nozzle_exit_pressure,
    )
    chute_cda = chute_cd * np.pi * (0.5 * chute_diameter) ** 2
    drogue_cda = drogue_cd * np.pi * (0.5 * drogue_diameter) ** 2

    # Launch site and initial attitude
    elevation = np.radians(launch_angle_deg)
    azimuth = np.radians(launch_azimuth_deg)
    q_rail_local = q_from_euler(0.0, -elevation, np.pi / 2.0 - azimuth)
    rail_dir = np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ]
    )
    lat0 = np.radians(launch_latitude_deg) if use_wgs84 else 0.0
    lon0 = np.radians(launch_longitude_deg) if use_wgs84 else 0.0
    launch_ecef = geodetic_to_ecef(lat0, lon0, launch_altitude_m if use_wgs84 else 0.0)
    launch_c = ecef_to_enu_matrix(lat0, lon0)
    earth_rate = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    rail_body_to_enu = q_to_rotation_matrix(q_rail_local)

    def held_state(t: float, y: np.ndarray, along_rail: float) -> np.ndarray:
        """Pin the vehicle to the rail: attitude fixed, velocity along the rail."""
        y = y.copy()
        if use_wgs84:
            c_e2i = ecef_to_eci_matrix(t)
            body_to_eci = c_e2i @ launch_c.T @ rail_body_to_enu
            y[0:3] = c_e2i @ (launch_ecef + launch_c.T @ (along_rail * rail_dir))
            relative = eci_to_ecef(y[3:6] - np.cross(earth_rate, y[0:3]), t)
            speed = max(0.0, float(relative @ (launch_c.T @ rail_dir)))
            y[3:6] = np.cross(earth_rate, y[0:3]) + ecef_to_eci(launch_c.T @ (speed * rail_dir), t)
            y[6:10] = q_from_dcm(body_to_eci)
            y[10:13] = body_to_eci.T @ earth_rate
        else:
            speed = max(0.0, float(y[3:6] @ rail_dir))
            y[0:3] = along_rail * rail_dir
            y[3:6] = speed * rail_dir
            y[6:10] = q_rail_local
            y[10:13] = 0.0
        return y

    def rail_distance(t: float, y: np.ndarray) -> float:
        if use_wgs84:
            return float((launch_c @ (eci_to_ecef(y[0:3], t) - launch_ecef)) @ rail_dir)
        return float(y[0:3] @ rail_dir)

    y = np.zeros(14)
    y[13] = rocket.initial_propellant_mass
    y = held_state(0.0, y, 0.0)

    t = 0.0
    history = _History(int(min(max_time, 600.0) / output_dt) + 8)
    deriv, diag = _flight_evaluate(t, y, p, launch_ecef, launch_c)
    history.append(t, y, diag)
    next_output = output_dt

    has_lifted = False
    liftoff_time = 0.0
    on_rail = rail_length > 0.0
    has_apogee = False
    apogee_time = 0.0
    apogee_alt = 0.0
    chute_deployed = False
    drogue_deployed = False
    deploy_time: float | None = None
    drogue_time: float | None = None
    burnout_time = burn_time
    burnout_alt = 0.0
    burnout_vel = 0.0
    has_burnout = burn_time <= 0.0 or y[13] <= 0.0
    ground_impact = False
    cleared_ground = False
    landing: tuple[float, float] | None = None
    adaptive_h = min(dt, h_max) if use_adaptive else dt
    prev_err = 1.0
    forced_min_steps = 0
    termination = "maximum_time"

    while t < max_time - 1e-12:
        if cancel is not None and cancel():
            raise SimulationCancelled(f"Cancelled at t={t:.3f} s")

        # Hold on the pad until thrust exceeds weight along the rail
        if not has_lifted:
            weight_along_rail = diag[_D_MASS] * G0 * np.sin(elevation)
            if diag[_D_THRUST] > weight_along_rail:
                has_lifted = True
                liftoff_time = t
            else:
                step = min(dt, max_time - t)
                if t < burn_time < t + step:
                    step = burn_time - t
                if diag[_D_THRUST] > 0.0:
                    y[13] = max(0.0, y[13] - mdot * step)
                t += step
                y = held_state(t, y, 0.0)
                deriv, diag = _flight_evaluate(t, y, p, launch_ecef, launch_c)
                if t >= next_output - 1e-12 or t >= max_time - 1e-12:
                    history.append(t, y, diag)
                    next_output = t + output_dt
                if burn_time <= t and not has_burnout:
                    has_burnout = True
                    burnout_time = t
                continue

        # Step size, landing exactly on burnout
        h_cap = max_time - t
        if t < burn_time:
            h_cap = min(h_cap, burn_time - t)
        if on_rail:
            # The rail constraint is applied between steps, so keep them short there.
            h_cap = min(h_cap, dt)
        h = min(adaptive_h if use_adaptive else dt, h_cap)

        if use_adaptive:
            while True:
                y_new, y_err, k_new = _flight_dp45(t, y, deriv, h, p, launch_ecef, launch_c)
                if use_wgs84:
                    offset_old = ecef_to_eci(launch_ecef, t)
                    offset_new = ecef_to_eci(launch_ecef, t + h)
                    scaled_old = y.copy()
                    scaled_new = y_new.copy()
                    scaled_old[0:3] -= offset_old
                    scaled_new[0:3] -= offset_new
                    scaled_old[3:6] -= np.cross(earth_rate, offset_old)
                    scaled_new[3:6] -= np.cross(earth_rate, offset_new)
                    err = rk45_error_norm(scaled_new, scaled_old, y_err, atol, rtol)
                else:
                    err = rk45_error_norm(y_new, y, y_err, atol, rtol)
                if err <= 1.0 or h <= h_min * (1.0 + 1e-9):
                    if err > 1.0:
                        forced_min_steps += 1
                    adaptive_h = compute_new_step_size(h, err, prev_err, h_min, h_max)
                    prev_err = max(err, 1e-10)
                    break
                h = min(compute_new_step_size(h, err, prev_err, h_min, h_max), h_cap)
            k_new_state = k_new
        else:
            y_new = _flight_rk4(t, y, h, p, launch_ecef, launch_c)
            k_new_state = None

        if not np.all(np.isfinite(y_new)):
            raise PhysicsViolationError(f"Non-finite state at t={t + h:.3f} s")

        t_new = t + h
        if on_rail:
            along = rail_distance(t_new, y_new)
            if along < rail_length:
                y_new = held_state(t_new, y_new, max(0.0, along))
                k_new_state = None
            else:
                on_rail = False

        deriv_new, diag_new = _flight_evaluate(t_new, y_new, p, launch_ecef, launch_c)
        if k_new_state is None:
            k_new_state = deriv_new

        # Burnout (by time or by propellant exhaustion)
        if not has_burnout and (t_new >= burn_time - 1e-12 or y_new[13] <= 0.0):
            has_burnout = True
            burnout_time = t_new
            burnout_alt = diag_new[_D_HEIGHT]
            burnout_vel = float(np.linalg.norm(diag_new[_D_VEL : _D_VEL + 3]))

        vz_old = diag[_D_VEL + 2]
        vz_new = diag_new[_D_VEL + 2]
        z_old = diag[_D_HEIGHT]
        z_new = diag_new[_D_HEIGHT]

        # Apogee: vertical velocity changes sign, refined with the cubic in time
        if not has_apogee and t > liftoff_time and vz_old > 0.0 >= vz_new:
            s = float(np.clip(vz_old / (vz_old - vz_new), 0.0, 1.0))
            s2 = s * s
            s3 = s2 * s
            apogee_time = t + s * h
            apogee_alt = (
                (2.0 * s3 - 3.0 * s2 + 1.0) * z_old
                + (s3 - 2.0 * s2 + s) * h * vz_old
                + (-2.0 * s3 + 3.0 * s2) * z_new
                + (s3 - s2) * h * vz_new
            )
            has_apogee = True

        if z_new > 1.0:
            cleared_ground = True
        impact = cleared_ground and z_new <= 0.0

        # Output samples inside (t, t_new]
        end = t_new
        impact_fraction = 1.0
        if impact:
            impact_fraction = (
                float(np.clip(z_old / (z_old - z_new), 0.0, 1.0)) if z_old > 0 else 1.0
            )
            end = t + impact_fraction * h
        while next_output < end - 1e-12:
            s = (next_output - t) / h
            y_s = _hermite(y, deriv, y_new, k_new_state, h, s)
            _, diag_s = _flight_evaluate(next_output, y_s, p, launch_ecef, launch_c)
            history.append(next_output, y_s, diag_s)
            next_output += output_dt

        if impact:
            y_hit = _hermite(y, deriv, y_new, k_new_state, h, impact_fraction)
            _, diag_hit = _flight_evaluate(end, y_hit, p, launch_ecef, launch_c)
            diag_hit[_D_POS + 2] = 0.0
            diag_hit[_D_HEIGHT] = 0.0
            history.append(end, y_hit, diag_hit)
            landing = (float(diag_hit[_D_POS]), float(diag_hit[_D_POS + 1]))
            ground_impact = True
            termination = "ground_impact"
            t = end
            break

        t, y, deriv, diag = t_new, y_new, deriv_new, diag_new
        if t >= next_output - 1e-12 or t >= max_time - 1e-12:
            history.append(t, y, diag)
            next_output = t + output_dt

        # Parachute deployment is an event: change the model between steps
        model_changed = False
        if has_apogee and drogue_cda > 0.0 and not drogue_deployed:
            drogue_deployed = True
            drogue_time = t
            model_changed = True
        main_altitude_reached = z_new <= main_deploy_altitude_m
        deploy_now = main_altitude_reached or (deploy_at_apogee and drogue_cda == 0.0)
        if has_apogee and chute_cda > 0.0 and not chute_deployed and deploy_now:
            chute_deployed = True
            deploy_time = t
            model_changed = True
        if model_changed:
            p[_P_CHUTE_CDA] = (drogue_cda if drogue_deployed else 0.0) + (
                chute_cda if chute_deployed else 0.0
            )
            deriv, diag = _flight_evaluate(t, y, p, launch_ecef, launch_c)

    n = history.n
    times = history.t[:n]
    states = history.y[:n]
    diags = history.d[:n]
    if n and not has_burnout:
        burnout_time = float(times[-1])
    if burnout_alt == 0.0 and has_burnout and n:
        index = int(np.searchsorted(times, burnout_time))
        index = min(index, n - 1)
        burnout_alt = float(diags[index, _D_HEIGHT])
        burnout_vel = float(np.linalg.norm(diags[index, _D_VEL : _D_VEL + 3]))

    euler = np.array([q_to_euler(diags[i, _D_QUAT : _D_QUAT + 4]) for i in range(n)]).reshape(n, 3)
    speed = np.linalg.norm(diags[:, _D_VEL : _D_VEL + 3], axis=1)
    ascending = times <= (apogee_time if has_apogee else times[-1] if n else 0.0)
    nan = np.full(n, np.nan)
    burning = diags[:, _D_THRUST] > 0.0

    return FlightResult6DOF(
        time=times.copy(),
        position_x=diags[:, _D_POS].copy(),
        position_y=diags[:, _D_POS + 1].copy(),
        position_z=np.maximum(diags[:, _D_HEIGHT], 0.0),
        velocity_x=diags[:, _D_VEL].copy(),
        velocity_y=diags[:, _D_VEL + 1].copy(),
        velocity_z=diags[:, _D_VEL + 2].copy(),
        velocity_magnitude=speed,
        quaternion_w=diags[:, _D_QUAT].copy(),
        quaternion_x=diags[:, _D_QUAT + 1].copy(),
        quaternion_y=diags[:, _D_QUAT + 2].copy(),
        quaternion_z=diags[:, _D_QUAT + 3].copy(),
        roll=np.degrees(euler[:, 0]),
        pitch=np.degrees(euler[:, 1]),
        yaw=np.degrees(euler[:, 2]),
        omega_x=states[:, 10].copy(),
        omega_y=states[:, 11].copy(),
        omega_z=states[:, 12].copy(),
        thrust=diags[:, _D_THRUST].copy(),
        drag=diags[:, _D_DRAG].copy(),
        gravity_force=diags[:, _D_GRAVITY].copy(),
        mach=diags[:, _D_MACH].copy(),
        dynamic_pressure=diags[:, _D_QBAR].copy(),
        mass=diags[:, _D_MASS].copy(),
        propellant_mass=np.maximum(states[:, 13], 0.0),
        isp=diags[:, _D_ISP].copy(),
        thrust_loss_factor=np.where(burning, 1.0, 0.0),
        flow_regime=diags[:, _D_REGIME].copy(),
        angle_of_attack=diags[:, _D_ALPHA].copy(),
        sideslip_angle=diags[:, _D_BETA].copy(),
        stability_margin=diags[:, _D_MARGIN].copy(),
        geodetic_latitude=diags[:, _D_LAT].copy() if use_wgs84 else nan,
        geodetic_longitude=diags[:, _D_LON].copy() if use_wgs84 else nan.copy(),
        geodetic_altitude=diags[:, _D_GEOD_ALT].copy() if use_wgs84 else nan.copy(),
        liftoff_time=liftoff_time,
        burnout_time=burnout_time,
        burnout_altitude=burnout_alt,
        burnout_velocity=burnout_vel,
        apogee_time=apogee_time,
        apogee_altitude=apogee_alt,
        max_velocity=float(speed.max()) if n else 0.0,
        max_mach=float(diags[:, _D_MACH].max()) if n else 0.0,
        max_acceleration=float(diags[times >= liftoff_time, _D_ACCEL].max()) if has_lifted else 0.0,
        max_q=float(diags[:, _D_QBAR].max()) if n else 0.0,
        max_alpha=float(
            np.abs(diags[ascending & (times > liftoff_time), _D_ALPHA]).max(initial=0.0)
        ),
        flight_time=float(times[-1]) if n else 0.0,
        success=has_lifted,
        abort_reason=None if has_lifted else "Vehicle did not lift off",
        apogee_reached=has_apogee,
        ground_impact=ground_impact,
        termination_reason=termination,
        reference_frame="wgs84_eci" if use_wgs84 else "local_enu",
        launch_latitude_deg=launch_latitude_deg if use_wgs84 else None,
        launch_longitude_deg=launch_longitude_deg if use_wgs84 else None,
        launch_altitude_m=launch_altitude_m if use_wgs84 else 0.0,
        parachute_deploy_time=deploy_time,
        drogue_deploy_time=drogue_time,
        landing_x=landing[0] if landing else None,
        landing_y=landing[1] if landing else None,
        forced_min_steps=forced_min_steps,
    )
