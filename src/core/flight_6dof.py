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
from numba import jit

from src.core.aero import calculate_drag_coefficient
from src.core.integrators import (
    DP5_A,
    DP5_B5,
    DP5_E,
    compute_new_step_size,
    rk45_error_norm,
)
from src.core.math_utils import (
    cross_product,
    q_conjugate,
    q_derivative,
    q_from_euler,
    q_mult,
    q_normalize,
    q_rotate_vector,
    q_to_euler,
)
from src.core.mission import get_atmosphere
from src.core.rocket import Rocket

# =============================================================================
# Physical Constants
# =============================================================================

G0 = 9.80665      # m/s² - Standard gravity at sea level
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
    time: np.ndarray                # [s]

    # Position (inertial frame)
    position_x: np.ndarray          # [m] East
    position_y: np.ndarray          # [m] North
    position_z: np.ndarray          # [m] Up (altitude)

    # Velocity (inertial frame)
    velocity_x: np.ndarray          # [m/s]
    velocity_y: np.ndarray          # [m/s]
    velocity_z: np.ndarray          # [m/s]
    velocity_magnitude: np.ndarray  # [m/s]

    # Orientation (quaternion history)
    quaternion_w: np.ndarray        # Scalar part
    quaternion_x: np.ndarray        # Vector x
    quaternion_y: np.ndarray        # Vector y
    quaternion_z: np.ndarray        # Vector z

    # Euler angles (derived from quaternion)
    roll: np.ndarray                # [deg] - φ
    pitch: np.ndarray               # [deg] - θ
    yaw: np.ndarray                 # [deg] - ψ

    # Angular velocity (body frame)
    omega_x: np.ndarray             # [rad/s] - Roll rate
    omega_y: np.ndarray             # [rad/s] - Pitch rate
    omega_z: np.ndarray             # [rad/s] - Yaw rate

    # Forces (inertial frame magnitude)
    thrust: np.ndarray              # [N]
    drag: np.ndarray                # [N]
    gravity_force: np.ndarray       # [N]

    # Atmospheric data
    mach: np.ndarray                # [-]
    dynamic_pressure: np.ndarray    # [Pa]

    # Mass and inertia
    mass: np.ndarray                # [kg] - Total mass
    propellant_mass: np.ndarray     # [kg] - Remaining propellant (state[13])

    # Engine performance (from RocketEngine)
    isp: np.ndarray                 # [s] - Specific impulse
    thrust_loss_factor: np.ndarray  # [-] - Flow separation loss (0.5-1.0)
    flow_regime: np.ndarray         # [-] - 0=Attached, 1=Over, 2=Separated, 3=Under

    # Stability
    angle_of_attack: np.ndarray     # [deg] - α
    sideslip_angle: np.ndarray      # [deg] - β
    stability_margin: np.ndarray    # [calibers]

    # Key events
    liftoff_time: float = 0.0       # [s]
    burnout_time: float = 0.0       # [s]
    burnout_altitude: float = 0.0   # [m]
    burnout_velocity: float = 0.0   # [m/s]
    apogee_time: float = 0.0        # [s]
    apogee_altitude: float = 0.0    # [m]
    max_velocity: float = 0.0       # [m/s]
    max_mach: float = 0.0           # [-]
    max_acceleration: float = 0.0   # [G]
    max_q: float = 0.0              # [Pa]
    max_alpha: float = 0.0          # [deg]
    flight_time: float = 0.0        # [s] - Total flight duration

    # Status
    success: bool = True
    abort_reason: str | None = None


# =============================================================================
# Inertia Tensor Calculation (Cylindrical Approximation)
# =============================================================================

@jit(nopython=True, cache=True)
def calculate_inertia_tensor_cylindrical(
    mass: float,
    length: float,
    radius: float
) -> np.ndarray:
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
    I = np.zeros((3, 3))
    I[0, 0] = Ixx
    I[1, 1] = Iyy
    I[2, 2] = Izz

    return I


@jit(nopython=True, cache=True)
def calculate_inertia_principal(
    mass: float,
    length: float,
    radius: float
) -> np.ndarray:
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
def calculate_aero_angles(
    velocity_body: np.ndarray
) -> tuple[float, float]:
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

    V = np.sqrt(vx*vx + vy*vy + vz*vz)

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
    cp_position: float,
    cg_position: float,
    force_aero_body: np.ndarray
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
    altitude: float
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

    # Gravity (already in inertial frame)
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
    # dm_prop = -mdot is handled externally (constant during RK4 substep)
    deriv = np.array([
        dx, dy, dz,                         # Position [0:3]
        dvx, dvy, dvz,                      # Velocity [3:6]
        q_dot[0], q_dot[1], q_dot[2], q_dot[3],  # Quaternion [6:10]
        dwx, dwy, dwz,                      # Angular velocity [10:13]
        0.0                                 # dm_prop placeholder (set by caller)
    ])

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
    mdot: float = 0.0
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
    # k1
    k1 = derivatives_6dof(state, mass, I_principal,
                          F_thrust_body, F_aero_body, M_aero, M_thrust, altitude)
    k1[13] = -mdot  # Propellant depletion

    # k2 (at t + dt/2)
    state2 = state + 0.5 * dt * k1
    alt2 = max(0.0, state2[2])
    k2 = derivatives_6dof(state2, mass, I_principal,
                          F_thrust_body, F_aero_body, M_aero, M_thrust, alt2)
    k2[13] = -mdot

    # k3 (at t + dt/2)
    state3 = state + 0.5 * dt * k2
    alt3 = max(0.0, state3[2])
    k3 = derivatives_6dof(state3, mass, I_principal,
                          F_thrust_body, F_aero_body, M_aero, M_thrust, alt3)
    k3[13] = -mdot

    # k4 (at t + dt)
    state4 = state + dt * k3
    alt4 = max(0.0, state4[2])
    k4 = derivatives_6dof(state4, mass, I_principal,
                          F_thrust_body, F_aero_body, M_aero, M_thrust, alt4)
    k4[13] = -mdot

    # RK4 update
    state_new = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

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
    mdot: float = 0.0
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

    # Stage 1: k1 = f(t, y)
    k[0] = derivatives_6dof(state, mass, I_principal,
                            F_thrust_body, F_aero_body, M_aero, M_thrust, altitude)
    k[0, 13] = -mdot

    # Stage 2: k2 = f(t + c2*h, y + h*a21*k1)
    y2 = state + h * (DP5_A[1, 0] * k[0])
    alt2 = max(0.0, y2[2])
    k[1] = derivatives_6dof(y2, mass, I_principal,
                            F_thrust_body, F_aero_body, M_aero, M_thrust, alt2)
    k[1, 13] = -mdot

    # Stage 3: k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
    y3 = state + h * (DP5_A[2, 0] * k[0] + DP5_A[2, 1] * k[1])
    alt3 = max(0.0, y3[2])
    k[2] = derivatives_6dof(y3, mass, I_principal,
                            F_thrust_body, F_aero_body, M_aero, M_thrust, alt3)
    k[2, 13] = -mdot

    # Stage 4
    y4 = state + h * (DP5_A[3, 0] * k[0] + DP5_A[3, 1] * k[1] +
                      DP5_A[3, 2] * k[2])
    alt4 = max(0.0, y4[2])
    k[3] = derivatives_6dof(y4, mass, I_principal,
                            F_thrust_body, F_aero_body, M_aero, M_thrust, alt4)
    k[3, 13] = -mdot

    # Stage 5
    y5 = state + h * (DP5_A[4, 0] * k[0] + DP5_A[4, 1] * k[1] +
                      DP5_A[4, 2] * k[2] + DP5_A[4, 3] * k[3])
    alt5 = max(0.0, y5[2])
    k[4] = derivatives_6dof(y5, mass, I_principal,
                            F_thrust_body, F_aero_body, M_aero, M_thrust, alt5)
    k[4, 13] = -mdot

    # Stage 6
    y6 = state + h * (DP5_A[5, 0] * k[0] + DP5_A[5, 1] * k[1] +
                      DP5_A[5, 2] * k[2] + DP5_A[5, 3] * k[3] +
                      DP5_A[5, 4] * k[4])
    alt6 = max(0.0, y6[2])
    k[5] = derivatives_6dof(y6, mass, I_principal,
                            F_thrust_body, F_aero_body, M_aero, M_thrust, alt6)
    k[5, 13] = -mdot

    # Compute 5th-order solution (y_new)
    y_new = state.copy()
    for i in range(n):
        for j in range(6):
            y_new[i] += h * DP5_B5[j] * k[j, i]

    # Stage 7 (FSAL): k7 = f(t + h, y_new) - this becomes k1 for next step
    alt_new = max(0.0, y_new[2])
    k[6] = derivatives_6dof(y_new, mass, I_principal,
                            F_thrust_body, F_aero_body, M_aero, M_thrust, alt_new)
    k[6, 13] = -mdot

    # Compute error estimate: y_error = h * sum(E_i * k_i)
    y_error = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(7):
            y_error[i] += h * DP5_E[j] * k[j, i]

    # Normalize quaternion
    q = y_new[6:10].copy()
    q_norm = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
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
    fin_misalignment_deg: float = 0.0
) -> FlightResult6DOF:
    """
    Simulate rocket flight using 6-DOF rigid body dynamics.

    Coordinate System (ENU - East-North-Up):
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

    # Setup engine parameters
    rocket.engine.thrust_vac = thrust_vac
    rocket.engine.isp_vac = isp_vac
    rocket.engine.burn_time = burn_time
    rocket.engine.mass_flow_rate = thrust_vac / (isp_vac * G0) if isp_vac > 0 else 0

    n_steps = int(max_time / dt) + 1

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
    isp_arr = np.zeros(n_steps)         # Specific impulse [s]
    thrust_loss_arr = np.zeros(n_steps) # Flow separation loss (0.5-1.0)
    flow_regime_arr = np.zeros(n_steps) # 0=Attached, 1=Over, 2=Separated, 3=Under
    alpha_arr = np.zeros(n_steps)
    beta_arr = np.zeros(n_steps)
    stability_arr = np.zeros(n_steps)

    # =========================================================================
    # Initial State
    # =========================================================================
    # Convert launch angles to quaternion
    # Pitch from horizontal, azimuth from North
    pitch_rad = np.radians(launch_angle_deg - 90.0)  # Convert to nose-up from vertical
    yaw_rad = np.radians(launch_azimuth_deg)

    # Initial quaternion (ZYX convention: yaw-pitch-roll)
    q_init = q_from_euler(0.0, pitch_rad, yaw_rad)

    # Apply fin misalignment perturbation (small rotation in pitch/yaw)
    if abs(fin_misalignment_deg) > 1e-6:
        # Convert misalignment to radians and apply as small pitch perturbation
        misalign_rad = np.radians(fin_misalignment_deg)
        q_perturb = q_from_euler(0.0, misalign_rad, misalign_rad * 0.5)
        q_init = q_mult(q_init, q_perturb)
        q_init = q_normalize(q_init)

    # Initial propellant mass
    m_prop_initial = rocket.initial_propellant_mass

    # Throttled mass flow rate (used throughout simulation)
    throttle_clamped = max(0.0, min(1.0, throttle))
    base_mdot = thrust_vac / (isp_vac * G0) if isp_vac > 0 else 0
    throttled_mdot = base_mdot * throttle_clamped

    # Initial state vector: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m_prop]
    # 14 elements: 3 position + 3 velocity + 4 quaternion + 3 angular velocity + 1 propellant mass
    state = np.array([
        0.0, 0.0, 0.0,          # Position (at origin)
        0.0, 0.0, 0.001,        # Velocity (tiny upward to start)
        q_init[0], q_init[1], q_init[2], q_init[3],  # Quaternion
        0.0, 0.0, 0.0,          # Angular velocity (stationary)
        m_prop_initial          # Propellant mass (kg)
    ])

    t = 0.0
    A_ref = rocket.reference_area
    rocket_length = rocket.total_length
    rocket_radius = rocket.reference_diameter / 2.0

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

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    for i in range(n_steps):
        # Store time
        time_arr[i] = t

        # Extract state components
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        q = state[6:10]
        omega = state[10:13]
        m_prop = state[13]  # Current propellant mass

        altitude = max(0.0, z)

        # Velocity magnitude
        V = np.sqrt(vx*vx + vy*vy + vz*vz)

        # Store position and velocity
        pos_x[i] = x
        pos_y[i] = y
        pos_z[i] = altitude
        vel_x[i] = vx
        vel_y[i] = vy
        vel_z[i] = vz
        vel_mag[i] = V

        # Store quaternion
        quat_w[i] = q[0]
        quat_x[i] = q[1]
        quat_y[i] = q[2]
        quat_z[i] = q[3]

        # Convert to Euler angles for output
        euler = q_to_euler(q)
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
        atm = get_atmosphere(altitude)
        rho = atm.density
        speed_of_sound = atm.speed_of_sound
        P_ambient = atm.pressure

        # Mach number
        M = V / speed_of_sound if speed_of_sound > 0 else 0.0
        mach_arr[i] = M

        # Dynamic pressure
        q_dyn = 0.5 * rho * V * V
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
        has_propellant = m_prop > 0.01  # 10g minimum

        if has_propellant:
            # Throttled thrust and mass flow
            F_thrust_base = (thrust_vac * throttle_clamped) - (P_ambient * exit_area)
            F_thrust_base = max(0.0, F_thrust_base)

            # Calculate exit pressure for flow separation analysis
            # Using isentropic relations: Pe/Pc ≈ (exit area ratio)^-gamma
            # Simplified: Pe ≈ Pc * 0.01 for typical expansion
            P_exit = P_ambient * 0.5 + 1000  # Simplified exit pressure estimate
            pressure_ratio = P_exit / P_ambient if P_ambient > 100 else 10.0

            # Flow separation detection (Summerfield criterion: Pe/Pa < 0.4)
            if pressure_ratio < 0.4:
                # Flow separated - significant thrust loss
                thrust_loss = 0.5 + 0.5 * (pressure_ratio / 0.4)  # 50-100% range
                flow_regime = 2  # Separated
            elif pressure_ratio < 0.9:
                # Overexpanded but attached
                thrust_loss = 0.9 + 0.1 * ((pressure_ratio - 0.4) / 0.5)
                flow_regime = 1  # Overexpanded
            elif altitude > 50000:  # Above 50km, essentially vacuum
                thrust_loss = 1.0
                flow_regime = 3  # Underexpanded (vacuum)
            else:
                thrust_loss = 1.0
                flow_regime = 0  # Attached

            # Apply flow separation loss to thrust
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
            flow_regime_arr[i] = 0
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
        # Transform velocity to body frame
        q_inv = q_conjugate(q)
        v_inertial = np.array([vx, vy, vz])
        v_body = q_rotate_vector(q_inv, v_inertial)

        # Calculate aerodynamic angles
        alpha, beta = calculate_aero_angles(v_body)
        alpha_arr[i] = np.degrees(alpha)
        beta_arr[i] = np.degrees(beta)
        max_alpha_val = max(max_alpha_val, abs(np.degrees(alpha)))

        # Drag coefficient (simple model based on Mach)
        cd = calculate_drag_coefficient(rocket, M) * cd_factor  # Apply perturbation

        # Drag force magnitude: D = 0.5 ρ V² Cd A
        D = q_dyn * cd * A_ref
        drag_arr[i] = D

        # Drag in body frame (opposes velocity)
        if V > 1e-6:
            v_body_unit = v_body / np.sqrt(v_body[0]**2 + v_body[1]**2 + v_body[2]**2)
            F_aero_body = -D * v_body_unit
        else:
            F_aero_body = np.array([0.0, 0.0, 0.0])

        # Gravity magnitude for output
        g_vec = calculate_gravity_vector(altitude)
        gravity_arr[i] = current_mass * abs(g_vec[2])

        # =====================================================================
        # Aerodynamic Moments
        # =====================================================================
        # Get CP and CG positions
        try:
            cg_pos = rocket.get_cg_at_time(t)
            # Approximate CP (for now, use fixed position near nose)
            cp_pos = rocket_length * 0.3  # 30% from nose (typical)
        except Exception:
            cg_pos = rocket_length * 0.5
            cp_pos = rocket_length * 0.3

        # Stability margin
        stability_cal = (cg_pos - cp_pos) / rocket.reference_diameter
        stability_arr[i] = stability_cal

        # Aerodynamic moment about CG
        M_aero = calculate_aerodynamic_moment(cp_pos, cg_pos, F_aero_body)

        # Thrust moment (assuming no thrust vectoring)
        M_thrust = np.array([0.0, 0.0, 0.0])

        # =====================================================================
        # Liftoff detection
        # =====================================================================
        if not has_lifted:
            weight = current_mass * G0
            if thrust_arr[i] > weight * 1.05:  # 5% T/W margin
                has_lifted = True
                liftoff_time = t

        # Acceleration (scalar, for max tracking)
        if current_mass > 0:
            accel = np.sqrt(
                (thrust_arr[i] - drag_arr[i])**2 +
                gravity_arr[i]**2
            ) / current_mass / G0
            max_accel = max(max_accel, accel)

        max_vel = max(max_vel, V)
        max_mach_val = max(max_mach_val, M)

        # =====================================================================
        # Apogee detection
        # =====================================================================
        if has_lifted and prev_vz > 0 and vz <= 0 and not has_apogee:
            has_apogee = True
            apogee_time = t
            apogee_alt = altitude
        prev_vz = vz

        # =====================================================================
        # Ground impact detection
        # =====================================================================
        # Only detect impact if we've actually gained altitude (avoid false trigger at t=0)
        if has_lifted and altitude > 1.0:
            max_alt_reached = True
        if has_lifted and z <= 0 and max_alt_reached:
            # Truncate arrays
            time_arr = time_arr[:i+1]
            pos_x = pos_x[:i+1]
            pos_y = pos_y[:i+1]
            pos_z = pos_z[:i+1]
            vel_x = vel_x[:i+1]
            vel_y = vel_y[:i+1]
            vel_z = vel_z[:i+1]
            vel_mag = vel_mag[:i+1]
            quat_w = quat_w[:i+1]
            quat_x = quat_x[:i+1]
            quat_y = quat_y[:i+1]
            quat_z = quat_z[:i+1]
            roll_arr = roll_arr[:i+1]
            pitch_arr = pitch_arr[:i+1]
            yaw_arr = yaw_arr[:i+1]
            omega_x_arr = omega_x_arr[:i+1]
            omega_y_arr = omega_y_arr[:i+1]
            omega_z_arr = omega_z_arr[:i+1]
            thrust_arr = thrust_arr[:i+1]
            drag_arr = drag_arr[:i+1]
            gravity_arr = gravity_arr[:i+1]
            mach_arr = mach_arr[:i+1]
            q_dynamic = q_dynamic[:i+1]
            mass_arr = mass_arr[:i+1]
            prop_mass_arr = prop_mass_arr[:i+1]
            isp_arr = isp_arr[:i+1]
            thrust_loss_arr = thrust_loss_arr[:i+1]
            flow_regime_arr = flow_regime_arr[:i+1]
            alpha_arr = alpha_arr[:i+1]
            beta_arr = beta_arr[:i+1]
            stability_arr = stability_arr[:i+1]
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
                dy_prev = derivatives_6dof(state_prev, current_mass, I_principal,
                                          F_thrust_body, F_aero_body, M_aero, M_thrust, altitude)
                dy_prev[13] = -mdot

                # Adaptive RK45 mode with step-size control
                prev_err_norm = 1.0
                h = min(dt, h_max)  # Start with reasonable step

                # Keep trying until step is accepted
                step_accepted = False
                max_attempts = 20
                attempts = 0

                while not step_accepted and attempts < max_attempts:
                    attempts += 1

                    # Try RK45 step
                    y_new, y_error, dy_new = rk45_step_rocket(
                        state, h, current_mass, I_principal,
                        F_thrust_body, F_aero_body, M_aero, M_thrust, altitude, mdot
                    )

                    # Compute error norm
                    err_norm = rk45_error_norm(y_new, state, y_error, atol, rtol)

                    if err_norm <= 1.0:
                        # Step accepted
                        state = y_new
                        step_accepted = True

                        # Compute new step size for next iteration
                        h = compute_new_step_size(h, err_norm, prev_err_norm, h_min, h_max)
                        prev_err_norm = max(err_norm, 1e-10)
                    else:
                        # Step rejected - reduce step size
                        h = compute_new_step_size(h, err_norm, prev_err_norm, h_min, h_max)
                        if h <= h_min:
                            state = y_new
                            step_accepted = True

                # Actual time advance (variable step)
                t_prev + h

                # Dense output: interpolate for any output times in [t_prev, t_new]
                # Note: For now, we advance by output_dt to maintain fixed output
                # The state is interpolated if needed
                t = t_prev + output_dt  # Fixed output step

            else:
                # Fixed step RK4 mode
                state = rk4_step_6dof(
                    state, dt, current_mass, I_principal,
                    F_thrust_body, F_aero_body, M_aero, M_thrust, altitude, mdot
                )
                t += dt

            # Check for NaN (physics violation)
            if np.any(np.isnan(state)):
                raise PhysicsViolationError(f"NaN detected in state at t={t:.2f}s")
        else:
            t += dt

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
        success=True
    )
