"""
6-DOF Rigid Body Flight Simulator.

Simulates rocket flight using full 6 Degrees of Freedom physics:
- 3 translational (x, y, z position)
- 3 rotational (roll, pitch, yaw via quaternion)

State Vector (13 elements):
    [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]

    Position:  [x, y, z]      (m) - Inertial frame (ENU: East-North-Up)
    Velocity:  [vx, vy, vz]   (m/s) - Inertial frame
    Quaternion: [q0, q1, q2, q3] (w, x, y, z) - Body to inertial rotation
    Angular Velocity: [wx, wy, wz] (rad/s) - Body frame

Physics:
    Translational: m(dv/dt) = F_gravity + F_thrust + F_aero
    Rotational: I(dω/dt) + ω × (Iω) = τ_aero + τ_thrust
    Quaternion: dq/dt = ½ ω ⊗ q

References:
    - Stevens & Lewis, "Aircraft Control and Simulation", 3rd ed.
    - Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics", 3rd ed.
"""

from dataclasses import dataclass

import numpy as np

from ensim.core.aero import calculate_total_cp
from ensim.core.geodesy import (
    EARTH_GRAVITATIONAL_PARAMETER,
    EARTH_J2,
    EARTH_ROTATION_RATE,
    WGS84_SEMI_MAJOR_AXIS,
    ecef_to_eci,
    ecef_to_eci_matrix,
    ecef_to_enu_matrix,
    ecef_to_geodetic,
    eci_to_ecef,
    eci_to_ecef_matrix,
    enu_to_ecef,
    geodetic_to_ecef,
    j2_gravity_ecef,
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
    q_from_euler,
    q_from_rotation_matrix,
    q_mult,
    q_normalize,
    q_rotate_vector,
    q_to_euler,
    q_to_rotation_matrix,
)
from ensim.core.mission import get_atmosphere
from ensim.core.rocket import Rocket
from ensim.utils.numba_support import jit

# =============================================================================
# Physical Constants
# =============================================================================

G0 = 9.80665  # m/s² - Standard gravity at sea level
R_EARTH = 6371000.0  # m - Mean Earth radius
MU_EARTH = 3.986004418e14  # m³/s² - Earth gravitational parameter


# =============================================================================
# Custom Exceptions
# =============================================================================


class PhysicsViolationError(Exception):
    """
    Raised when physically impossible state is detected.

    Examples:
        - Negative mass
        - NaN in state vector
        - Negative chamber pressure
    """

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FlightResult6DOF:
    """Complete 6-DOF flight simulation results."""

    # Time history
    time: np.ndarray  # [s]

    # Position (inertial frame)
    position_x: np.ndarray  # [m] East
    position_y: np.ndarray  # [m] North
    position_z: np.ndarray  # [m] Up (altitude)

    # Velocity (inertial frame)
    velocity_x: np.ndarray  # [m/s]
    velocity_y: np.ndarray  # [m/s]
    velocity_z: np.ndarray  # [m/s]
    velocity_magnitude: np.ndarray  # [m/s]

    # Orientation (quaternion history)
    quaternion_w: np.ndarray  # Scalar part
    quaternion_x: np.ndarray  # Vector x
    quaternion_y: np.ndarray  # Vector y
    quaternion_z: np.ndarray  # Vector z

    # Euler angles (derived from quaternion)
    roll: np.ndarray  # [deg] - φ
    pitch: np.ndarray  # [deg] - θ
    yaw: np.ndarray  # [deg] - ψ

    # Angular velocity (body frame)
    omega_x: np.ndarray  # [rad/s] - Roll rate
    omega_y: np.ndarray  # [rad/s] - Pitch rate
    omega_z: np.ndarray  # [rad/s] - Yaw rate

    # Forces (inertial frame magnitude)
    thrust: np.ndarray  # [N]
    drag: np.ndarray  # [N]
    gravity_force: np.ndarray  # [N]

    # Atmospheric data
    mach: np.ndarray  # [-]
    dynamic_pressure: np.ndarray  # [Pa]

    # Mass and inertia
    mass: np.ndarray  # [kg] - Total mass
    propellant_mass: np.ndarray  # [kg] - Remaining propellant (state[13])

    # Engine performance (from RocketEngine)
    isp: np.ndarray  # [s] - Specific impulse
    thrust_loss_factor: np.ndarray  # [-] - Flow separation loss (0.5-1.0)
    flow_regime: np.ndarray  # [-] - 0=Attached, 1=Over, 2=Separated, 3=Under

    # Stability
    angle_of_attack: np.ndarray  # [deg] - α
    sideslip_angle: np.ndarray  # [deg] - β
    stability_margin: np.ndarray  # [calibers]

    # Geodetic coordinates (WGS-84 mode; NaN in local mode)
    geodetic_latitude: np.ndarray  # [deg]
    geodetic_longitude: np.ndarray  # [deg]
    geodetic_altitude: np.ndarray  # [m] above ellipsoid

    # Key events
    liftoff_time: float = 0.0  # [s]
    burnout_time: float = 0.0  # [s]
    burnout_altitude: float = 0.0  # [m]
    burnout_velocity: float = 0.0  # [m/s]
    apogee_time: float = 0.0  # [s]
    apogee_altitude: float = 0.0  # [m]
    max_velocity: float = 0.0  # [m/s]
    max_mach: float = 0.0  # [-]
    max_acceleration: float = 0.0  # [G]
    max_q: float = 0.0  # [Pa]
    max_alpha: float = 0.0  # [deg]
    flight_time: float = 0.0  # [s] - Total flight duration

    # Status
    success: bool = True
    abort_reason: str | None = None
    apogee_reached: bool = False
    ground_impact: bool = False
    termination_reason: str = "maximum_time"
    reference_frame: str = "local_enu"
    launch_latitude_deg: float | None = None
    launch_longitude_deg: float | None = None
    launch_altitude_m: float = 0.0


# =============================================================================
# Inertia Tensor Calculation (Cylindrical Approximation)
# =============================================================================


@jit(nopython=True, cache=True)
def calculate_inertia_tensor_cylindrical(mass: float, length: float, radius: float) -> np.ndarray:
    """
    Calculate moment of inertia tensor for a cylinder.

    Approximates rocket as a uniform solid cylinder aligned with X-axis.

    For a cylinder of mass m, length L, radius r:
        Ixx = ½ m r²           (roll - about longitudinal axis)
        Iyy = Izz = m(3r² + L²)/12   (pitch/yaw - transverse axes)

    Args:
        mass: Total mass (kg)
        length: Cylinder length (m)
        radius: Cylinder radius (m)

    Returns:
        3x3 inertia tensor [kg·m²]
    """
    # Moments of inertia (principal axes)
    Ixx = 0.5 * mass * radius * radius
    Iyy = mass * (3.0 * radius * radius + length * length) / 12.0
    Izz = Iyy  # Symmetric about longitudinal axis

    # Return diagonal inertia tensor (principal axes aligned with body)
    inertia = np.zeros((3, 3))
    inertia[0, 0] = Ixx
    inertia[1, 1] = Iyy
    inertia[2, 2] = Izz

    return inertia


@jit(nopython=True, cache=True)
def calculate_inertia_principal(mass: float, length: float, radius: float) -> np.ndarray:
    """
    Calculate principal moments of inertia [Ixx, Iyy, Izz].

    Faster than full tensor for diagonal case.

    Args:
        mass: Total mass (kg)
        length: Cylinder length (m)
        radius: Cylinder radius (m)

    Returns:
        Array [Ixx, Iyy, Izz] in kg·m²
    """
    Ixx = 0.5 * mass * radius * radius
    Iyy = mass * (3.0 * radius * radius + length * length) / 12.0
    Izz = Iyy

    return np.array([Ixx, Iyy, Izz])


# =============================================================================
# Gravity Model
# =============================================================================


@jit(nopython=True, cache=True)
def calculate_gravity_vector(altitude: float) -> np.ndarray:
    """
    Calculate gravitational acceleration vector at altitude.

    Uses inverse square law in local vertical direction (ENU frame):
        g(h) = g0 × (R_e / (R_e + h))² × [0, 0, -1]

    Args:
        altitude: Altitude above sea level (m)

    Returns:
        Gravity vector [gx, gy, gz] in m/s² (ENU frame)
    """
    g_magnitude = G0 * (R_EARTH / (R_EARTH + max(0.0, altitude))) ** 2
    return np.array([0.0, 0.0, -g_magnitude])


# =============================================================================
# Aerodynamic Forces and Moments
# =============================================================================


@jit(nopython=True, cache=True)
def calculate_aero_angles(velocity_body: np.ndarray) -> tuple[float, float]:
    """
    Calculate angle of attack (α) and sideslip (β) from body velocity.

    Body frame convention:
        X = forward (thrust direction)
        Y = right wing
        Z = down

    Args:
        velocity_body: Velocity vector in body frame [vx, vy, vz] (m/s)

    Returns:
        Tuple of (alpha, beta) in radians
    """
    vx = velocity_body[0]
    vy = velocity_body[1]
    vz = velocity_body[2]

    V = np.sqrt(vx * vx + vy * vy + vz * vz)

    if V < 1e-6:
        return 0.0, 0.0

    # Angle of attack: angle in vertical plane
    alpha = np.arctan2(-vz, vx)

    # Sideslip: angle in horizontal plane
    sinb = vy / V
    if sinb > 1.0:
        sinb = 1.0
    elif sinb < -1.0:
        sinb = -1.0
    beta = np.arcsin(sinb)

    return alpha, beta


@jit(nopython=True, cache=True)
def calculate_aerodynamic_moment(
    cp_position: float, cg_position: float, force_aero_body: np.ndarray
) -> np.ndarray:
    """
    Calculate aerodynamic moment about CG.

    The moment arm is the vector from CG to CP (in body frame):
        r_cp = [cp_position - cg_position, 0, 0]  (assuming CP on centerline)

    Moment = r_cp × F_aero

    Args:
        cp_position: CP distance from nose tip (m)
        cg_position: CG distance from nose tip (m)
        force_aero_body: Aerodynamic force in body frame [Fx, Fy, Fz] (N)

    Returns:
        Moment vector [Mx, My, Mz] in N·m (body frame)
    """
    # Moment arm from CG to CP (in body X direction)
    r_arm = np.array([cp_position - cg_position, 0.0, 0.0])

    # τ = r × F
    moment = cross_product(r_arm, force_aero_body)

    return moment


# =============================================================================
# 6-DOF State Derivatives (Core Physics)
# =============================================================================


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
    """
    Calculate state derivatives for 6-DOF rigid body dynamics.

    State: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]

    Physics:
        ṗ = v                                    (position rate)
        v̇ = (F_thrust + F_aero)/m + g            (velocity rate)
        q̇ = ½ ω ⊗ q                              (quaternion rate)
        ω̇ = I⁻¹ (τ - ω × Iω)                     (Euler's equations)

    Args:
        state: 13-element state vector
        mass: Current mass (kg)
        I_principal: Principal moments [Ixx, Iyy, Izz] (kg·m²)
        F_thrust_body: Thrust force in body frame (N)
        F_aero_body: Aerodynamic force in body frame (N)
        M_aero: Aerodynamic moment about CG (N·m)
        M_thrust: Thrust moment about CG (N·m) - for thrust vectoring
        altitude: Current altitude (m) - for gravity calculation

    Returns:
        13-element state derivative vector
    """
    # Validate mass
    if mass <= 0:
        # Return zero derivatives to halt simulation
        return np.zeros(13)

    # Extract state components
    # Position
    # x, y, z = state[0], state[1], state[2]

    # Velocity (inertial frame)
    vx, vy, vz = state[3], state[4], state[5]

    # Quaternion (body → inertial)
    q = state[6:10].copy()

    # Angular velocity (body frame)
    omega = state[10:13].copy()
    wx, wy, wz = omega[0], omega[1], omega[2]

    # =========================================================================
    # Position derivatives: ṗ = v
    # =========================================================================
    dx = vx
    dy = vy
    dz = vz

    # =========================================================================
    # Velocity derivatives: v̇ = F_total/m + g
    # =========================================================================

    # Transform forces from body to inertial frame
    F_thrust_inertial = q_rotate_vector(q, F_thrust_body)
    F_aero_inertial = q_rotate_vector(q, F_aero_body)

    if use_earth_model:
        position = state[0:3]
        radius_squared = position @ position
        radius = np.sqrt(radius_squared)
        z_ratio_squared = position[2] * position[2] / radius_squared
        correction = 1.5 * EARTH_J2 * (WGS84_SEMI_MAJOR_AXIS / radius) ** 2
        common = -EARTH_GRAVITATIONAL_PARAMETER / radius**3
        g = common * np.array(
            [
                position[0] * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared)),
                position[1] * (1.0 + correction * (1.0 - 5.0 * z_ratio_squared)),
                position[2] * (1.0 + correction * (3.0 - 5.0 * z_ratio_squared)),
            ]
        )
    else:
        g = calculate_gravity_vector(altitude)

    # Total acceleration
    dvx = (F_thrust_inertial[0] + F_aero_inertial[0]) / mass + g[0]
    dvy = (F_thrust_inertial[1] + F_aero_inertial[1]) / mass + g[1]
    dvz = (F_thrust_inertial[2] + F_aero_inertial[2]) / mass + g[2]

    # =========================================================================
    # Quaternion derivatives: q̇ = ½ ω ⊗ q
    # =========================================================================
    q_dot = q_derivative(q, omega)

    # =========================================================================
    # Angular velocity derivatives: Euler's equations
    # I·ω̇ + ω × (I·ω) = τ
    # ω̇ = I⁻¹·(τ - ω × I·ω)
    # =========================================================================

    Ixx, Iyy, Izz = I_principal[0], I_principal[1], I_principal[2]

    # I·ω
    I_omega = np.array([Ixx * wx, Iyy * wy, Izz * wz])

    # ω × I·ω (gyroscopic term)
    gyro = cross_product(omega, I_omega)

    # Total moment
    M_total = M_aero + M_thrust

    # ω̇ = I⁻¹·(τ - ω × I·ω)
    dwx = (M_total[0] - gyro[0]) / Ixx if Ixx > 1e-10 else 0.0
    dwy = (M_total[1] - gyro[1]) / Iyy if Iyy > 1e-10 else 0.0
    dwz = (M_total[2] - gyro[2]) / Izz if Izz > 1e-10 else 0.0

    # =========================================================================
    # Assemble derivative vector (14 elements for 14-state vector)
    # =========================================================================
    # Propellant depletion is assigned by the integration step.
    deriv = np.array(
        [
            dx,
            dy,
            dz,  # Position [0:3]
            dvx,
            dvy,
            dvz,  # Velocity [3:6]
            q_dot[0],
            q_dot[1],
            q_dot[2],
            q_dot[3],  # Quaternion [6:10]
            dwx,
            dwy,
            dwz,  # Angular velocity [10:13]
            0.0,
        ]
    )

    return deriv


# =============================================================================
# RK4 Integrator with Quaternion Normalization
# =============================================================================


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
    """
    Single RK4 integration step for 6-DOF dynamics with propellant mass.

    Uses 4th-order Runge-Kutta method with quaternion renormalization.
    Integrates propellant mass as state[13].

    Args:
        state: Current 14-element state vector
        dt: Time step (s)
        mass: Current mass (kg)
        I_principal: Principal moments of inertia (kg·m²)
        F_thrust_body: Thrust force in body frame (N)
        F_aero_body: Aerodynamic force in body frame (N)
        M_aero: Aerodynamic moment (N·m)
        M_thrust: Thrust moment (N·m)
        altitude: Current altitude (m)
        mdot: Mass flow rate (kg/s)

    Returns:
        New 14-element state vector
    """
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
    k1[13] = -mdot  # Propellant depletion

    # k2 (at t + dt/2)
    state2 = state + 0.5 * dt * k1
    alt2 = max(0.0, state2[2])
    mass2 = max(dry_mass + max(state2[13], 0.0), 1e-12)
    k2 = derivatives_6dof(
        state2,
        mass2,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt2,
        use_earth_model,
    )
    k2[13] = -mdot

    # k3 (at t + dt/2)
    state3 = state + 0.5 * dt * k2
    alt3 = max(0.0, state3[2])
    mass3 = max(dry_mass + max(state3[13], 0.0), 1e-12)
    k3 = derivatives_6dof(
        state3,
        mass3,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt3,
        use_earth_model,
    )
    k3[13] = -mdot

    # k4 (at t + dt)
    state4 = state + dt * k3
    alt4 = max(0.0, state4[2])
    mass4 = max(dry_mass + max(state4[13], 0.0), 1e-12)
    k4 = derivatives_6dof(
        state4,
        mass4,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt4,
        use_earth_model,
    )
    k4[13] = -mdot

    # RK4 update
    state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # =========================================================================
    # CRITICAL: Quaternion Normalization
    # =========================================================================
    q = state_new[6:10].copy()
    q_normalized = q_normalize(q)
    state_new[6] = q_normalized[0]
    state_new[7] = q_normalized[1]
    state_new[8] = q_normalized[2]
    state_new[9] = q_normalized[3]

    # Clamp propellant mass to non-negative
    if state_new[13] < 0.0:
        state_new[13] = 0.0

    return state_new


# =============================================================================
# RK45 Adaptive Integrator (Dormand-Prince 5(4))
# =============================================================================


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
    """
    Single Dormand-Prince 5(4) adaptive step for rocket dynamics.

    Uses 7-stage FSAL (First Same As Last) RK method for efficient
    error estimation and dense output.

    Args:
        state: Current 14-element state vector
        h: Step size
        mass: Current mass (kg)
        I_principal: Principal moments of inertia (kg·m²)
        F_thrust_body: Thrust force in body frame (N)
        F_aero_body: Aerodynamic force in body frame (N)
        M_aero: Aerodynamic moment (N·m)
        M_thrust: Thrust moment (N·m)
        altitude: Current altitude (m)
        mdot: Mass flow rate (kg/s)

    Returns:
        Tuple of:
            y_new: 5th-order accurate new state (14 elements)
            y_error: Error estimate (y5 - y4) for step-size control
            dy_new: Derivative at new point (for FSAL and interpolation)
    """
    n = 14  # State dimension

    # Allocate k stages
    k = np.zeros((7, n), dtype=np.float64)
    dry_mass = mass - max(state[13], 0.0)

    # Stage 1: k1 = f(t, y)
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

    # Stage 2: k2 = f(t + c2*h, y + h*a21*k1)
    y2 = state + h * (DP5_A[1, 0] * k[0])
    alt2 = max(0.0, y2[2])
    mass2 = max(dry_mass + max(y2[13], 0.0), 1e-12)
    k[1] = derivatives_6dof(
        y2,
        mass2,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt2,
        use_earth_model,
    )
    k[1, 13] = -mdot

    # Stage 3: k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
    y3 = state + h * (DP5_A[2, 0] * k[0] + DP5_A[2, 1] * k[1])
    alt3 = max(0.0, y3[2])
    mass3 = max(dry_mass + max(y3[13], 0.0), 1e-12)
    k[2] = derivatives_6dof(
        y3,
        mass3,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt3,
        use_earth_model,
    )
    k[2, 13] = -mdot

    # Stage 4
    y4 = state + h * (DP5_A[3, 0] * k[0] + DP5_A[3, 1] * k[1] + DP5_A[3, 2] * k[2])
    alt4 = max(0.0, y4[2])
    mass4 = max(dry_mass + max(y4[13], 0.0), 1e-12)
    k[3] = derivatives_6dof(
        y4,
        mass4,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt4,
        use_earth_model,
    )
    k[3, 13] = -mdot

    # Stage 5
    y5 = state + h * (
        DP5_A[4, 0] * k[0] + DP5_A[4, 1] * k[1] + DP5_A[4, 2] * k[2] + DP5_A[4, 3] * k[3]
    )
    alt5 = max(0.0, y5[2])
    mass5 = max(dry_mass + max(y5[13], 0.0), 1e-12)
    k[4] = derivatives_6dof(
        y5,
        mass5,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt5,
        use_earth_model,
    )
    k[4, 13] = -mdot

    # Stage 6
    y6 = state + h * (
        DP5_A[5, 0] * k[0]
        + DP5_A[5, 1] * k[1]
        + DP5_A[5, 2] * k[2]
        + DP5_A[5, 3] * k[3]
        + DP5_A[5, 4] * k[4]
    )
    alt6 = max(0.0, y6[2])
    mass6 = max(dry_mass + max(y6[13], 0.0), 1e-12)
    k[5] = derivatives_6dof(
        y6,
        mass6,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt6,
        use_earth_model,
    )
    k[5, 13] = -mdot

    # Compute 5th-order solution (y_new)
    y_new = state.copy()
    for i in range(n):
        for j in range(6):
            y_new[i] += h * DP5_B5[j] * k[j, i]

    # Stage 7 (FSAL): k7 = f(t + h, y_new) - this becomes k1 for next step
    alt_new = max(0.0, y_new[2])
    mass_new = max(dry_mass + max(y_new[13], 0.0), 1e-12)
    k[6] = derivatives_6dof(
        y_new,
        mass_new,
        I_principal,
        F_thrust_body,
        F_aero_body,
        M_aero,
        M_thrust,
        alt_new,
        use_earth_model,
    )
    k[6, 13] = -mdot

    # Compute error estimate: y_error = h * sum(E_i * k_i)
    y_error = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(7):
            y_error[i] += h * DP5_E[j] * k[j, i]

    # Normalize quaternion
    q = y_new[6:10].copy()
    q_norm = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if q_norm > 1e-10:
        y_new[6] = q[0] / q_norm
        y_new[7] = q[1] / q_norm
        y_new[8] = q[2] / q_norm
        y_new[9] = q[3] / q_norm

    # Clamp propellant mass
    if y_new[13] < 0.0:
        y_new[13] = 0.0

    return y_new, y_error, k[6]


# =============================================================================
# Main Simulation Function
# =============================================================================


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
    # Adaptive mode parameters
    use_adaptive: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    h_min: float = 1e-5,
    h_max: float = 1.0,
    output_dt: float = 0.01,  # Fixed output sampling rate (Dense Output)
    # Perturbation parameters (Monte Carlo)
    throttle: float = 1.0,
    cd_factor: float = 1.0,
    fin_misalignment_deg: float = 0.0,
    # Recovery parameters
    chute_diameter: float = 0.0,
    deploy_at_apogee: bool = True,
    nozzle_exit_pressure: float | None = None,
    use_wgs84: bool = False,
    launch_latitude_deg: float = 0.0,
    launch_longitude_deg: float = 0.0,
    launch_altitude_m: float = 0.0,
) -> FlightResult6DOF:
    """
    Simulate rocket flight using 6-DOF rigid body dynamics.

    Output coordinate system (ENU - East-North-Up):
        X: East
        Y: North
        Z: Up (altitude)

    Body Frame (aligned at launch):
        X: Forward (thrust direction)
        Y: Right
        Z: Down

    Args:
        rocket: Rocket vehicle definition
        thrust_vac: Vacuum thrust (N)
        isp_vac: Vacuum specific impulse (s)
        burn_time: Engine burn duration (s)
        exit_area: Nozzle exit area (m²)
        dt: Integration time step (s)
        max_time: Maximum simulation time (s)
        launch_angle_deg: Pitch angle from horizontal (90 = vertical)
        launch_azimuth_deg: Heading from North (0 = North, 90 = East)
        rail_length: Launch rail length (m)
        wind_speed: Wind speed (m/s)
        wind_direction_deg: Wind direction FROM (meteorological convention)
        throttle: Engine throttle (0.0-1.0), affects thrust and mass flow
        cd_factor: Drag coefficient multiplier for perturbation
        fin_misalignment_deg: Fin misalignment angle for quaternion perturbation
        use_wgs84: Integrate position and attitude in ECI with WGS-84/J2 gravity
        launch_latitude_deg: Geodetic launch latitude for WGS-84 mode
        launch_longitude_deg: Geodetic launch longitude for WGS-84 mode
        launch_altitude_m: Geodetic launch altitude above the WGS-84 ellipsoid

    Returns:
        FlightResult6DOF with complete trajectory history

    Raises:
        PhysicsViolationError: If non-physical state is detected
    """
    # Validate inputs
    if thrust_vac < 0:
        raise PhysicsViolationError(f"Negative thrust: {thrust_vac} N")
    if isp_vac <= 0:
        raise PhysicsViolationError(f"Invalid Isp: {isp_vac} s")
    if rocket.wet_mass <= 0:
        raise PhysicsViolationError(f"Invalid mass: {rocket.wet_mass} kg")
    if burn_time < 0.0 or dt <= 0.0 or max_time <= 0.0 or output_dt <= 0.0:
        raise PhysicsViolationError("Burn time must be nonnegative and time steps positive")
    if exit_area < 0.0 or rail_length < 0.0 or wind_speed < 0.0 or chute_diameter < 0.0:
        raise PhysicsViolationError("Geometry and environmental magnitudes cannot be negative")
    if not 0.0 <= throttle <= 1.0:
        raise PhysicsViolationError("Throttle must be between zero and one")
    if rocket.axial_drag_coefficient < 0.0 or cd_factor <= 0.0:
        raise PhysicsViolationError("Drag coefficient and its multiplier must be nonnegative")
    if use_wgs84 and not -90.0 <= launch_latitude_deg <= 90.0:
        raise PhysicsViolationError("Launch latitude must be between -90 and 90 degrees")
    if use_wgs84 and not -180.0 <= launch_longitude_deg <= 180.0:
        raise PhysicsViolationError("Launch longitude must be between -180 and 180 degrees")
    if use_wgs84 and not np.isfinite(launch_altitude_m):
        raise PhysicsViolationError("Launch altitude must be finite")

    # Setup engine parameters
    rocket.engine.thrust_vac = thrust_vac
    rocket.engine.isp_vac = isp_vac
    rocket.engine.burn_time = burn_time
    rocket.engine.mass_flow_rate = thrust_vac / (isp_vac * G0) if isp_vac > 0 else 0

    nominal_output_step = min(dt, output_dt) if use_adaptive else dt
    n_steps = int(np.ceil(max_time / nominal_output_step)) + 3

    # =========================================================================
    # Allocate output arrays
    # =========================================================================
    time_arr = np.zeros(n_steps)
    pos_x = np.zeros(n_steps)
    pos_y = np.zeros(n_steps)
    pos_z = np.zeros(n_steps)
    vel_x = np.zeros(n_steps)
    vel_y = np.zeros(n_steps)
    vel_z = np.zeros(n_steps)
    vel_mag = np.zeros(n_steps)
    quat_w = np.zeros(n_steps)
    quat_x = np.zeros(n_steps)
    quat_y = np.zeros(n_steps)
    quat_z = np.zeros(n_steps)
    roll_arr = np.zeros(n_steps)
    pitch_arr = np.zeros(n_steps)
    yaw_arr = np.zeros(n_steps)
    omega_x_arr = np.zeros(n_steps)
    omega_y_arr = np.zeros(n_steps)
    omega_z_arr = np.zeros(n_steps)
    thrust_arr = np.zeros(n_steps)
    drag_arr = np.zeros(n_steps)
    gravity_arr = np.zeros(n_steps)
    mach_arr = np.zeros(n_steps)
    q_dynamic = np.zeros(n_steps)
    mass_arr = np.zeros(n_steps)
    prop_mass_arr = np.zeros(n_steps)  # Propellant mass (state[13])
    isp_arr = np.zeros(n_steps)  # Specific impulse [s]
    thrust_loss_arr = np.zeros(n_steps)
    flow_regime_arr = np.full(n_steps, -1.0)
    alpha_arr = np.zeros(n_steps)
    beta_arr = np.zeros(n_steps)
    stability_arr = np.zeros(n_steps)
    latitude_arr = np.full(n_steps, np.nan)
    longitude_arr = np.full(n_steps, np.nan)
    geodetic_altitude_arr = np.full(n_steps, np.nan)

    # =========================================================================
    # Initial State
    # =========================================================================
    elevation = np.radians(launch_angle_deg)
    azimuth = np.radians(launch_azimuth_deg)
    q_init_local = q_from_euler(0.0, -elevation, np.pi / 2.0 - azimuth)

    # Apply fin misalignment perturbation (small rotation in pitch/yaw)
    if abs(fin_misalignment_deg) > 1e-6:
        # Convert misalignment to radians and apply as small pitch perturbation
        misalign_rad = np.radians(fin_misalignment_deg)
        q_perturb = q_from_euler(0.0, misalign_rad, misalign_rad * 0.5)
        q_init_local = q_mult(q_init_local, q_perturb)
        q_init_local = q_normalize(q_init_local)

    launch_latitude = np.radians(launch_latitude_deg) if use_wgs84 else 0.0
    launch_longitude = np.radians(launch_longitude_deg) if use_wgs84 else 0.0
    geodesy_launch_altitude = launch_altitude_m if use_wgs84 else 0.0
    launch_ecef = geodetic_to_ecef(
        launch_latitude,
        launch_longitude,
        geodesy_launch_altitude,
    )
    launch_ecef_to_enu = ecef_to_enu_matrix(launch_latitude, launch_longitude)
    earth_rate_vector = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    if use_wgs84:
        body_to_eci = launch_ecef_to_enu.T @ q_to_rotation_matrix(q_init_local)
        q_init = q_from_rotation_matrix(body_to_eci)
    else:
        q_init = q_init_local

    # Initial propellant mass
    m_prop_initial = rocket.initial_propellant_mass

    # Throttled mass flow rate (used throughout simulation)
    throttle_clamped = throttle
    base_mdot = thrust_vac / (isp_vac * G0) if isp_vac > 0 else 0
    throttled_mdot = base_mdot * throttle_clamped

    # Rail unit vector for initial state
    sin_la_init = np.sin(np.radians(launch_angle_deg))
    cos_la_init = np.cos(np.radians(launch_angle_deg))
    sin_az_init = np.sin(np.radians(launch_azimuth_deg))
    cos_az_init = np.cos(np.radians(launch_azimuth_deg))
    rail_dx_init = cos_la_init * sin_az_init
    rail_dy_init = cos_la_init * cos_az_init
    rail_dz_init = sin_la_init

    # Initial state vector: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m_prop]
    # 14 elements: 3 position + 3 velocity + 4 quaternion + 3 angular velocity + 1 propellant mass
    rail_direction_enu = np.array([rail_dx_init, rail_dy_init, rail_dz_init])
    if use_wgs84:
        initial_position = launch_ecef.copy()
        initial_velocity = np.cross(earth_rate_vector, initial_position) + enu_to_ecef(
            0.001 * rail_direction_enu, launch_latitude, launch_longitude
        )
    else:
        initial_position = np.zeros(3)
        initial_velocity = 0.001 * rail_direction_enu

    state = np.array(
        [
            initial_position[0],
            initial_position[1],
            initial_position[2],
            initial_velocity[0],
            initial_velocity[1],
            initial_velocity[2],
            q_init[0],
            q_init[1],
            q_init[2],
            q_init[3],  # Quaternion
            0.0,
            0.0,
            0.0,  # Angular velocity (stationary)
            m_prop_initial,  # Propellant mass (kg)
        ]
    )

    t = 0.0
    A_ref = rocket.reference_area
    rocket_length = rocket.total_length
    rocket_radius = rocket.reference_diameter / 2.0
    cp_position, cn_alpha, _ = calculate_total_cp(rocket)

    # Event tracking
    liftoff_time = 0.0
    burnout_time = burn_time
    burnout_alt = 0.0
    burnout_vel = 0.0
    apogee_time = 0.0
    apogee_alt = 0.0
    max_vel = 0.0
    max_mach_val = 0.0
    max_accel = 0.0
    max_q = 0.0
    max_alpha_val = 0.0

    has_lifted = False
    has_burnout = False
    has_apogee = False
    max_alt_reached = False
    prev_vz = 0.0
    prev_z = 0.0
    prev_time = 0.0
    chute_deployed = False
    adaptive_h = min(dt, output_dt, h_max)
    ground_impact = False
    samples_used = 0

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    for i in range(n_steps):
        if t > max_time + 1e-12:
            break
        samples_used = i + 1
        # Store time
        time_arr[i] = t

        # Extract state components
        state_position = state[0:3]
        state_velocity = state[3:6]
        q_dynamics = state[6:10]
        omega = state[10:13]
        m_prop = state[13]  # Current propellant mass

        if use_wgs84:
            position_ecef = eci_to_ecef(state_position, t)
            velocity_ecef = eci_to_ecef(
                state_velocity - np.cross(earth_rate_vector, state_position),
                t,
            )
            latitude, longitude, geodetic_altitude = ecef_to_geodetic(position_ecef)
            current_ecef_to_enu = ecef_to_enu_matrix(latitude, longitude)
            position_enu = launch_ecef_to_enu @ (position_ecef - launch_ecef)
            velocity_enu = current_ecef_to_enu @ velocity_ecef
            x, y = position_enu[0], position_enu[1]
            z = geodetic_altitude - launch_altitude_m
            vx, vy, vz = velocity_enu
            atmosphere_altitude = max(0.0, geodetic_altitude)
            body_to_local = (
                current_ecef_to_enu @ eci_to_ecef_matrix(t) @ q_to_rotation_matrix(q_dynamics)
            )
            q_output = q_from_rotation_matrix(body_to_local)
        else:
            x, y, z = state_position
            vx, vy, vz = state_velocity
            latitude = launch_latitude
            longitude = launch_longitude
            current_ecef_to_enu = launch_ecef_to_enu
            atmosphere_altitude = max(0.0, z)
            q_output = q_dynamics

        altitude = max(0.0, z)
        V = float(np.linalg.norm([vx, vy, vz]))

        wind_to_dir_rad = np.radians(wind_direction_deg + 180.0)
        local_wind_speed = wind_speed * (altitude / 10.0) ** 0.143 if altitude > 0.0 else 0.0
        wind_velocity_enu = np.array(
            [
                local_wind_speed * np.sin(wind_to_dir_rad),
                local_wind_speed * np.cos(wind_to_dir_rad),
                0.0,
            ]
        )
        if use_wgs84:
            wind_velocity_ecef = current_ecef_to_enu.T @ wind_velocity_enu
            atmosphere_velocity_eci = np.cross(earth_rate_vector, state_position) + ecef_to_eci(
                wind_velocity_ecef, t
            )
            v_rel_inertial = state_velocity - atmosphere_velocity_eci
        else:
            v_rel_inertial = state_velocity - wind_velocity_enu
        V_rel = float(np.linalg.norm(v_rel_inertial))

        # Store position and velocity
        pos_x[i] = x
        pos_y[i] = y
        pos_z[i] = altitude
        vel_x[i] = vx
        vel_y[i] = vy
        vel_z[i] = vz
        vel_mag[i] = V
        if use_wgs84:
            latitude_arr[i] = np.degrees(latitude)
            longitude_arr[i] = np.degrees(longitude)
            geodetic_altitude_arr[i] = geodetic_altitude

        # Store quaternion
        quat_w[i] = q_output[0]
        quat_x[i] = q_output[1]
        quat_y[i] = q_output[2]
        quat_z[i] = q_output[3]

        # Convert to Euler angles for output
        euler = q_to_euler(q_output)
        roll_arr[i] = np.degrees(euler[0])
        pitch_arr[i] = np.degrees(euler[1])
        yaw_arr[i] = np.degrees(euler[2])

        # Store angular velocity
        omega_x_arr[i] = omega[0]
        omega_y_arr[i] = omega[1]
        omega_z_arr[i] = omega[2]

        # =====================================================================
        # Atmospheric conditions
        # =====================================================================
        atm = get_atmosphere(atmosphere_altitude)
        rho = atm.density
        speed_of_sound = atm.speed_of_sound
        P_ambient = atm.pressure

        # Mach number
        M = V_rel / speed_of_sound if speed_of_sound > 0 else 0.0
        mach_arr[i] = M

        # Dynamic pressure
        q_dyn = 0.5 * rho * V_rel * V_rel
        q_dynamic[i] = q_dyn
        max_q = max(max_q, q_dyn)

        # =====================================================================
        # Mass and Inertia (using propellant mass from state[13])
        # =====================================================================
        current_mass = rocket.get_mass_from_propellant(m_prop)
        mass_arr[i] = current_mass
        prop_mass_arr[i] = max(0.0, m_prop)

        # Verify mass is physical
        if current_mass <= 0 or np.isnan(current_mass):
            raise PhysicsViolationError(f"Invalid mass at t={t:.2f}s: {current_mass} kg")

        # Calculate inertia tensor (cylindrical approximation)
        I_principal = calculate_inertia_principal(current_mass, rocket_length, rocket_radius)

        # =====================================================================
        # Thrust (with throttle and flow separation)
        # =====================================================================
        # Check if propellant is available
        has_propellant = m_prop > 0.0 and t < burn_time - 1e-12

        if has_propellant:
            # Throttled thrust and mass flow
            F_thrust_base = (thrust_vac * throttle_clamped) - (P_ambient * exit_area)
            F_thrust_base = max(0.0, F_thrust_base)

            thrust_loss = 1.0
            flow_regime = -1
            if nozzle_exit_pressure is not None and P_ambient > 0.0:
                pressure_ratio = nozzle_exit_pressure / P_ambient
                if pressure_ratio < 0.4:
                    flow_regime = 2
                elif pressure_ratio < 1.0:
                    flow_regime = 1
                elif pressure_ratio > 1.05:
                    flow_regime = 3
                else:
                    flow_regime = 0

            F_thrust_mag = F_thrust_base * thrust_loss
            thrust_arr[i] = F_thrust_mag
            thrust_loss_arr[i] = thrust_loss
            flow_regime_arr[i] = flow_regime

            # Current ISP (altitude-corrected)
            isp_current = F_thrust_mag / (throttled_mdot * G0) if throttled_mdot > 0 else isp_vac
            isp_arr[i] = isp_current

            # Thrust in body frame (along +X axis)
            F_thrust_body = np.array([F_thrust_mag, 0.0, 0.0])

            # Mass flow (throttled)
            mdot = throttled_mdot
        else:
            thrust_arr[i] = 0.0
            isp_arr[i] = 0.0
            thrust_loss_arr[i] = 0.0
            flow_regime_arr[i] = -1
            F_thrust_body = np.array([0.0, 0.0, 0.0])
            mdot = 0.0

            if not has_burnout:
                has_burnout = True
                burnout_time = t
                burnout_alt = altitude
                burnout_vel = V

        # =====================================================================
        # Aerodynamic Forces
        # =====================================================================
        # Transform relative velocity to body frame
        q_inv = q_conjugate(q_dynamics)
        v_body = q_rotate_vector(q_inv, v_rel_inertial)

        # Calculate aerodynamic angles from relative wind
        alpha, beta = calculate_aero_angles(v_body)
        alpha_arr[i] = np.degrees(alpha)
        beta_arr[i] = np.degrees(beta)
        max_alpha_val = max(max_alpha_val, abs(np.degrees(alpha)))

        cd = rocket.axial_drag_coefficient * cd_factor

        # Drag force magnitude: D = 0.5 ρ V² Cd A
        D = q_dyn * cd * A_ref
        drag_arr[i] = D

        # Drag in body frame (opposes air-relative velocity)
        F_normal_body = np.zeros(3)
        if V_rel > 1e-6:
            v_body_unit = v_body / V_rel
            F_aero_body = -D * v_body_unit
            if v_body[0] > 0.0 and abs(alpha) <= np.radians(20.0) and abs(beta) <= np.radians(20.0):
                normal_scale = q_dyn * A_ref * cn_alpha
                F_normal_body[1] = -normal_scale * beta
                F_normal_body[2] = normal_scale * alpha
                F_aero_body += F_normal_body
        else:
            F_aero_body = np.array([0.0, 0.0, 0.0])

        # Parachute recovery drag integration
        if chute_deployed and chute_diameter > 0:
            chute_cda = 1.5 * np.pi * (chute_diameter / 2) ** 2
            D_chute = q_dyn * chute_cda
            if V_rel > 1e-6:
                F_aero_body += -D_chute * v_body_unit

        # Gravity magnitude for output
        if use_wgs84:
            g_vec = j2_gravity_ecef(position_ecef)
            gravity_arr[i] = current_mass * np.linalg.norm(g_vec)
        else:
            g_vec = calculate_gravity_vector(altitude)
            gravity_arr[i] = current_mass * abs(g_vec[2])

        # =====================================================================
        # Aerodynamic Moments
        # =====================================================================
        cg_pos = rocket.get_cg_from_propellant(m_prop)
        cp_pos = cp_position

        # Stability margin
        stability_cal = (cp_pos - cg_pos) / rocket.reference_diameter
        stability_arr[i] = stability_cal

        # Aerodynamic moment about CG
        M_aero = calculate_aerodynamic_moment(cp_pos, cg_pos, F_normal_body)

        # Thrust moment (assuming no thrust vectoring)
        M_thrust = np.array([0.0, 0.0, 0.0])

        # =====================================================================
        # Liftoff detection
        # =====================================================================
        if not has_lifted:
            weight = current_mass * G0
            if thrust_arr[i] > weight:
                has_lifted = True
                liftoff_time = t

        # Acceleration (scalar, for max tracking)
        if current_mass > 0:
            accel = (
                np.sqrt((thrust_arr[i] - drag_arr[i]) ** 2 + gravity_arr[i] ** 2)
                / current_mass
                / G0
            )
            max_accel = max(max_accel, accel)

        max_vel = max(max_vel, V)
        max_mach_val = max(max_mach_val, M)

        # =====================================================================
        # Apogee detection
        # =====================================================================
        if has_lifted and prev_vz > 0.0 and vz <= 0.0 and not has_apogee:
            has_apogee = True
            interval = t - prev_time
            fraction = prev_vz / (prev_vz - vz)
            fraction = float(np.clip(fraction, 0.0, 1.0))
            apogee_time = prev_time + fraction * interval
            s2 = fraction * fraction
            s3 = s2 * fraction
            apogee_alt = (
                (2.0 * s3 - 3.0 * s2 + 1.0) * prev_z
                + (s3 - 2.0 * s2 + fraction) * interval * prev_vz
                + (-2.0 * s3 + 3.0 * s2) * z
                + (s3 - s2) * interval * vz
            )

        # =====================================================================
        # Parachute Deployment Logic
        # =====================================================================
        if has_apogee and chute_diameter > 0.0 and (deploy_at_apogee or altitude <= 300.0):
            chute_deployed = True

        # =====================================================================
        # Ground impact detection
        # =====================================================================
        # Only detect impact if we've actually gained altitude (avoid false trigger at t=0)
        if has_lifted and altitude > 1.0:
            max_alt_reached = True
        if has_lifted and z <= 0 and max_alt_reached:
            ground_impact = True
            interval = t - prev_time
            if prev_z > 0.0 and interval > 0.0:
                fraction = prev_z / (prev_z - z)
                impact_time = prev_time + fraction * interval
                time_arr[i] = impact_time
                pos_x[i] = pos_x[i - 1] + fraction * (x - pos_x[i - 1])
                pos_y[i] = pos_y[i - 1] + fraction * (y - pos_y[i - 1])
                pos_z[i] = 0.0
                vel_x[i] = vel_x[i - 1] + fraction * (vx - vel_x[i - 1])
                vel_y[i] = vel_y[i - 1] + fraction * (vy - vel_y[i - 1])
                vel_z[i] = vel_z[i - 1] + fraction * (vz - vel_z[i - 1])
                vel_mag[i] = np.linalg.norm([vel_x[i], vel_y[i], vel_z[i]])
                if use_wgs84:
                    latitude_arr[i] = latitude_arr[i - 1] + fraction * (
                        latitude_arr[i] - latitude_arr[i - 1]
                    )
                    longitude_arr[i] = longitude_arr[i - 1] + fraction * (
                        longitude_arr[i] - longitude_arr[i - 1]
                    )
                    geodetic_altitude_arr[i] = launch_altitude_m
            # Truncate arrays
            time_arr = time_arr[: i + 1]
            pos_x = pos_x[: i + 1]
            pos_y = pos_y[: i + 1]
            pos_z = pos_z[: i + 1]
            vel_x = vel_x[: i + 1]
            vel_y = vel_y[: i + 1]
            vel_z = vel_z[: i + 1]
            vel_mag = vel_mag[: i + 1]
            quat_w = quat_w[: i + 1]
            quat_x = quat_x[: i + 1]
            quat_y = quat_y[: i + 1]
            quat_z = quat_z[: i + 1]
            roll_arr = roll_arr[: i + 1]
            pitch_arr = pitch_arr[: i + 1]
            yaw_arr = yaw_arr[: i + 1]
            omega_x_arr = omega_x_arr[: i + 1]
            omega_y_arr = omega_y_arr[: i + 1]
            omega_z_arr = omega_z_arr[: i + 1]
            thrust_arr = thrust_arr[: i + 1]
            drag_arr = drag_arr[: i + 1]
            gravity_arr = gravity_arr[: i + 1]
            mach_arr = mach_arr[: i + 1]
            q_dynamic = q_dynamic[: i + 1]
            mass_arr = mass_arr[: i + 1]
            prop_mass_arr = prop_mass_arr[: i + 1]
            isp_arr = isp_arr[: i + 1]
            thrust_loss_arr = thrust_loss_arr[: i + 1]
            flow_regime_arr = flow_regime_arr[: i + 1]
            alpha_arr = alpha_arr[: i + 1]
            beta_arr = beta_arr[: i + 1]
            stability_arr = stability_arr[: i + 1]
            latitude_arr = latitude_arr[: i + 1]
            longitude_arr = longitude_arr[: i + 1]
            geodetic_altitude_arr = geodetic_altitude_arr[: i + 1]
            break

        if t >= max_time - 1e-12:
            break

        # =====================================================================
        # Integration step (RK4 fixed or RK45 adaptive with dense output)
        # =====================================================================
        if has_lifted:
            if use_adaptive:
                # Save previous state for interpolation
                state_prev = state.copy()
                t_prev = t

                # Compute derivative at previous state (for Hermite interpolation)
                dy_prev = derivatives_6dof(
                    state_prev,
                    current_mass,
                    I_principal,
                    F_thrust_body,
                    F_aero_body,
                    M_aero,
                    M_thrust,
                    altitude,
                    use_wgs84,
                )
                dy_prev[13] = -mdot

                # Adaptive RK45 mode with step-size control
                prev_err_norm = 1.0
                h = adaptive_h
                if t < burn_time < t + h:
                    h = burn_time - t
                h = min(h, max_time - t)

                # Keep trying until step is accepted
                step_accepted = False
                max_attempts = 20
                attempts = 0

                while not step_accepted and attempts < max_attempts:
                    attempts += 1

                    # Try RK45 step
                    y_new, y_error, dy_new = rk45_step_rocket(
                        state,
                        h,
                        current_mass,
                        I_principal,
                        F_thrust_body,
                        F_aero_body,
                        M_aero,
                        M_thrust,
                        altitude,
                        mdot,
                        use_wgs84,
                    )

                    # Compute error norm
                    if use_wgs84:
                        scaled_new = y_new.copy()
                        scaled_old = state.copy()
                        launch_position_old = ecef_to_eci(launch_ecef, t)
                        launch_position_new = ecef_to_eci(launch_ecef, t + h)
                        scaled_old[0:3] -= launch_position_old
                        scaled_new[0:3] -= launch_position_new
                        scaled_old[3:6] -= np.cross(earth_rate_vector, launch_position_old)
                        scaled_new[3:6] -= np.cross(earth_rate_vector, launch_position_new)
                        err_norm = rk45_error_norm(
                            scaled_new,
                            scaled_old,
                            y_error,
                            atol,
                            rtol,
                        )
                    else:
                        err_norm = rk45_error_norm(
                            y_new,
                            state,
                            y_error,
                            atol,
                            rtol,
                        )

                    if err_norm <= 1.0:
                        # Step accepted
                        state = y_new
                        step_accepted = True
                        accepted_h = h

                        # Compute new step size for next iteration
                        adaptive_h = min(
                            dt,
                            output_dt,
                            compute_new_step_size(h, err_norm, prev_err_norm, h_min, h_max),
                        )
                        prev_err_norm = max(err_norm, 1e-10)
                    else:
                        # Step rejected - reduce step size
                        h = compute_new_step_size(h, err_norm, prev_err_norm, h_min, h_max)
                        if h <= h_min:
                            state = y_new
                            step_accepted = True
                            accepted_h = h

                if not step_accepted:
                    raise PhysicsViolationError(
                        f"Adaptive integrator failed to accept a step at t={t:.6f}s"
                    )
                t = t_prev + accepted_h

            else:
                # Fixed step RK4 mode
                step_dt = min(dt, max_time - t)
                if t < burn_time < t + step_dt:
                    step_dt = burn_time - t
                state = rk4_step_6dof(
                    state,
                    step_dt,
                    current_mass,
                    I_principal,
                    F_thrust_body,
                    F_aero_body,
                    M_aero,
                    M_thrust,
                    altitude,
                    mdot,
                    use_wgs84,
                )
                t += step_dt

            # Check for NaN (physics violation)
            if np.any(np.isnan(state)):
                raise PhysicsViolationError(f"NaN detected in state at t={t:.2f}s")

            # Apply launch rail constraint (project velocity and lock orientation)
            if use_wgs84:
                constrained_ecef = eci_to_ecef(state[0:3], t)
                constrained_enu = launch_ecef_to_enu @ (constrained_ecef - launch_ecef)
                dist_traveled = float(np.linalg.norm(constrained_enu))
            else:
                dist_traveled = float(np.linalg.norm(state[0:3]))
            if dist_traveled < rail_length:
                if use_wgs84:
                    rail_body_to_eci = (
                        ecef_to_eci_matrix(t)
                        @ launch_ecef_to_enu.T
                        @ q_to_rotation_matrix(q_init_local)
                    )
                    rail_quaternion = q_from_rotation_matrix(rail_body_to_eci)
                    state[6:10] = rail_quaternion
                    state[10:13] = rail_body_to_eci.T @ earth_rate_vector
                    relative_speed = float(
                        np.linalg.norm(
                            eci_to_ecef(
                                state[3:6] - np.cross(earth_rate_vector, state[0:3]),
                                t,
                            )
                        )
                    )
                    rail_velocity_ecef = launch_ecef_to_enu.T @ (
                        relative_speed * rail_direction_enu
                    )
                    state[3:6] = np.cross(earth_rate_vector, state[0:3]) + ecef_to_eci(
                        rail_velocity_ecef, t
                    )
                else:
                    state[6:10] = q_init
                    state[10:13] = 0.0
                    relative_speed = float(np.linalg.norm(state[3:6]))
                    state[3:6] = relative_speed * rail_direction_enu
        else:
            t = min(t + dt, max_time)
            if use_wgs84:
                state[0:3] = ecef_to_eci(launch_ecef, t)
                state[3:6] = np.cross(earth_rate_vector, state[0:3])
                held_body_to_eci = (
                    ecef_to_eci_matrix(t)
                    @ launch_ecef_to_enu.T
                    @ q_to_rotation_matrix(q_init_local)
                )
                state[6:10] = q_from_rotation_matrix(held_body_to_eci)
                state[10:13] = held_body_to_eci.T @ earth_rate_vector

        prev_vz = vz
        prev_z = z
        prev_time = time_arr[i]

    if samples_used < len(time_arr):
        time_arr = time_arr[:samples_used]
        pos_x = pos_x[:samples_used]
        pos_y = pos_y[:samples_used]
        pos_z = pos_z[:samples_used]
        vel_x = vel_x[:samples_used]
        vel_y = vel_y[:samples_used]
        vel_z = vel_z[:samples_used]
        vel_mag = vel_mag[:samples_used]
        quat_w = quat_w[:samples_used]
        quat_x = quat_x[:samples_used]
        quat_y = quat_y[:samples_used]
        quat_z = quat_z[:samples_used]
        roll_arr = roll_arr[:samples_used]
        pitch_arr = pitch_arr[:samples_used]
        yaw_arr = yaw_arr[:samples_used]
        omega_x_arr = omega_x_arr[:samples_used]
        omega_y_arr = omega_y_arr[:samples_used]
        omega_z_arr = omega_z_arr[:samples_used]
        thrust_arr = thrust_arr[:samples_used]
        drag_arr = drag_arr[:samples_used]
        gravity_arr = gravity_arr[:samples_used]
        mach_arr = mach_arr[:samples_used]
        q_dynamic = q_dynamic[:samples_used]
        mass_arr = mass_arr[:samples_used]
        prop_mass_arr = prop_mass_arr[:samples_used]
        isp_arr = isp_arr[:samples_used]
        thrust_loss_arr = thrust_loss_arr[:samples_used]
        flow_regime_arr = flow_regime_arr[:samples_used]
        alpha_arr = alpha_arr[:samples_used]
        beta_arr = beta_arr[:samples_used]
        stability_arr = stability_arr[:samples_used]
        latitude_arr = latitude_arr[:samples_used]
        longitude_arr = longitude_arr[:samples_used]
        geodetic_altitude_arr = geodetic_altitude_arr[:samples_used]

    # =========================================================================
    # Build result
    # =========================================================================
    return FlightResult6DOF(
        time=time_arr,
        position_x=pos_x,
        position_y=pos_y,
        position_z=pos_z,
        velocity_x=vel_x,
        velocity_y=vel_y,
        velocity_z=vel_z,
        velocity_magnitude=vel_mag,
        quaternion_w=quat_w,
        quaternion_x=quat_x,
        quaternion_y=quat_y,
        quaternion_z=quat_z,
        roll=roll_arr,
        pitch=pitch_arr,
        yaw=yaw_arr,
        omega_x=omega_x_arr,
        omega_y=omega_y_arr,
        omega_z=omega_z_arr,
        thrust=thrust_arr,
        drag=drag_arr,
        gravity_force=gravity_arr,
        mach=mach_arr,
        dynamic_pressure=q_dynamic,
        mass=mass_arr,
        propellant_mass=prop_mass_arr,
        isp=isp_arr,
        thrust_loss_factor=thrust_loss_arr,
        flow_regime=flow_regime_arr,
        angle_of_attack=alpha_arr,
        sideslip_angle=beta_arr,
        stability_margin=stability_arr,
        geodetic_latitude=latitude_arr,
        geodetic_longitude=longitude_arr,
        geodetic_altitude=geodetic_altitude_arr,
        liftoff_time=liftoff_time,
        burnout_time=burnout_time,
        burnout_altitude=burnout_alt,
        burnout_velocity=burnout_vel,
        apogee_time=apogee_time,
        apogee_altitude=apogee_alt,
        max_velocity=max_vel,
        max_mach=max_mach_val,
        max_acceleration=max_accel,
        max_q=max_q,
        max_alpha=max_alpha_val,
        flight_time=time_arr[-1] if len(time_arr) > 0 else 0.0,
        success=has_lifted,
        abort_reason=None if has_lifted else "Vehicle did not lift off",
        apogee_reached=has_apogee,
        ground_impact=ground_impact,
        termination_reason="ground_impact" if ground_impact else "maximum_time",
        reference_frame="wgs84_eci" if use_wgs84 else "local_enu",
        launch_latitude_deg=launch_latitude_deg if use_wgs84 else None,
        launch_longitude_deg=launch_longitude_deg if use_wgs84 else None,
        launch_altitude_m=launch_altitude_m if use_wgs84 else 0.0,
    )
