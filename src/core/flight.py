"""
RK4 2D Flight Simulator (Gravity Turn).

Simulates rocket flight trajectory using 2D point-mass equations of motion
in a gravity field with aerodynamic drag. Supports gravity turns.

Physics Reference:
    - Variable gravity: g(h) = g0 × (R_e / (R_e + h))²
    - Gravity turn: dγ/dt = -(g × cos(γ)) / v
    - Drag: D = 0.5 × ρ × v² × Cd × A_ref

References:
    - Wiesel, W.E. "Spaceflight Dynamics", 3rd ed.
    - Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed.
"""

from dataclasses import dataclass

import numpy as np
from numba import jit

from src.core.aero import calculate_drag_coefficient, calculate_stability_margin
from src.core.mission import get_atmosphere
from src.core.rocket import Rocket

# =============================================================================
# Physical Constants
# =============================================================================

G0 = 9.80665  # m/s² - Standard gravity at sea level
R_EARTH = 6371000.0  # m - Mean Earth radius


@dataclass
class FlightResult:
    """Complete flight simulation results."""
    # Time history
    time: np.ndarray
    altitude: np.ndarray
    range: np.ndarray  # Downrange distance (m)
    velocity: np.ndarray
    acceleration: np.ndarray
    mach: np.ndarray
    path_angle: np.ndarray  # Flight path angle (degrees)

    # Forces
    thrust: np.ndarray
    drag: np.ndarray
    q: np.ndarray  # Dynamic pressure (Pa)

    # Stability
    stability_margin: np.ndarray
    angle_of_attack: np.ndarray

    # Key events
    liftoff_time: float
    burnout_time: float
    burnout_altitude: float
    burnout_velocity: float
    apogee_time: float
    apogee_altitude: float
    max_velocity: float
    max_mach: float
    max_acceleration: float  # G's
    max_q: float  # Dynamic pressure (Pa)

    # Status
    success: bool
    max_aoa: float = 0.0
    abort_reason: str | None = None


@jit(nopython=True, cache=True)
def calculate_gravity(altitude: float) -> float:
    """
    Calculate gravitational acceleration at altitude.

    Uses inverse square law:
        g(h) = g0 × (R_e / (R_e + h))²

    This correction is important for high-altitude flights (>10km)
    where g can decrease by ~0.3% per kilometer.

    Args:
        altitude: Altitude above sea level (m)

    Returns:
        Local gravitational acceleration (m/s²)

    Reference:
        Wiesel, W.E. "Spaceflight Dynamics", Eq. 1.2
    """
    return G0 * (R_EARTH / (R_EARTH + altitude)) ** 2


@jit(nopython=True, cache=True)
def _rk4_derivatives_2d(
    y: np.ndarray,  # [x, z, v, gamma]
    t: float,
    mass: float,
    thrust: float,
    drag_coeff: float,
    rho: float,
    A_ref: float,
    altitude: float,
    rail_length: float = 0.0,
    rail_angle_rad: float = 1.5708  # 90 deg (vertical)
) -> np.ndarray:
    """
    Calculate derivatives for 2D Gravity Turn.

    State:
    y[0] = x (range)
    y[1] = z (altitude)
    y[2] = v (velocity magnitude)
    y[3] = gamma (flight path angle, rads from horizontal)

    Equations:
    dx/dt = v * cos(gamma)
    dz/dt = v * sin(gamma)
    dv/dt = (T - D)/m - g * sin(gamma)
    dgamma/dt = - (g * cos(gamma)) / v  (Gravity Turn)
    """
    _x, z, v, gamma = y[0], y[1], y[2], y[3]

    # Ground constraint - return zero derivatives if below ground
    if z < 0:
        return np.array([0.0, 0.0, 0.0, 0.0])

    # Calculate altitude-dependent gravity
    g = calculate_gravity(z if z > 0 else 0.0)

    # Rail constraint - lock direction while on launch rail
    on_rail = z < rail_length * np.sin(rail_angle_rad)
    if on_rail:
        gamma = rail_angle_rad

    # Drag force: D = 0.5 × ρ × v² × Cd × A_ref
    F_drag = 0.5 * rho * v * v * drag_coeff * A_ref

    # Acceleration (dv/dt) along velocity vector
    # dv/dt = (T - D)/m - g × sin(γ)
    accel = (thrust - F_drag) / mass - g * np.sin(gamma)

    # Path angle change (dγ/dt) - Gravity Turn equation
    # dγ/dt = -(g × cos(γ)) / v
    #
    # CRITICAL: Avoid singularity when v → 0
    # Physical interpretation: At low speeds, the rocket can't turn
    # because aerodynamic/gravitational forces dominate inertia
    VELOCITY_THRESHOLD = 1.0  # m/s - minimum velocity for gravity turn

    if v < VELOCITY_THRESHOLD:
        # Below threshold: no gravity turn (rocket maintains current angle)
        d_gamma = 0.0
    elif on_rail:
        # On launch rail: direction locked to rail angle
        d_gamma = 0.0
    else:
        # Normal gravity turn dynamics
        d_gamma = -(g * np.cos(gamma)) / v

    # Velocity components in inertial frame
    dx = v * np.cos(gamma)  # Downrange velocity
    dz = v * np.sin(gamma)  # Vertical velocity

    return np.array([dx, dz, accel, d_gamma])


@jit(nopython=True, cache=True)
def _rk4_step_2d(
    y: np.ndarray,
    t: float,
    dt: float,
    mass: float,
    thrust: float,
    cd: float,
    rho: float,
    A_ref: float,
    rail_length: float = 0.0,
    rail_angle_rad: float = 1.5708
) -> np.ndarray:
    """
    Single RK4 integration step for 2D flight solver.

    Uses 4th-order Runge-Kutta method for numerical integration.
    Gravity is calculated based on current altitude for each substep.
    """
    # Current altitude for gravity calculation
    altitude = max(0.0, y[1])

    k1 = _rk4_derivatives_2d(y, t, mass, thrust, cd, rho, A_ref, altitude, rail_length, rail_angle_rad)

    y2 = y + 0.5 * dt * k1
    alt2 = max(0.0, y2[1])
    k2 = _rk4_derivatives_2d(y2, t + 0.5*dt, mass, thrust, cd, rho, A_ref, alt2, rail_length, rail_angle_rad)

    y3 = y + 0.5 * dt * k2
    alt3 = max(0.0, y3[1])
    k3 = _rk4_derivatives_2d(y3, t + 0.5*dt, mass, thrust, cd, rho, A_ref, alt3, rail_length, rail_angle_rad)

    y4 = y + dt * k3
    alt4 = max(0.0, y4[1])
    k4 = _rk4_derivatives_2d(y4, t + dt, mass, thrust, cd, rho, A_ref, alt4, rail_length, rail_angle_rad)

    y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y_new


def simulate_flight(
    rocket: Rocket,
    thrust_vac: float,
    isp_vac: float,
    burn_time: float,
    exit_area: float = 0.01,
    dt: float = 0.01,
    max_time: float = 300.0,
    wind_speed: float = 0.0,
    rail_length: float = 3.0,
    launch_angle_deg: float = 85.0
) -> FlightResult:
    """
    Simulate rocket flight (2D Gravity Turn).

    Args:
        launch_angle_deg: Launch angle from HORIZONTAL (90 = vertical).
    """
    G0 = 9.80665

    # Setup engine
    rocket.engine.thrust_vac = thrust_vac
    rocket.engine.isp_vac = isp_vac
    rocket.engine.burn_time = burn_time
    rocket.engine.mass_flow_rate = thrust_vac / (isp_vac * G0) if isp_vac > 0 else 0

    n_steps = int(max_time / dt) + 1

    # Data arrays
    time = np.zeros(n_steps)
    altitude = np.zeros(n_steps)
    downrange = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    acceleration = np.zeros(n_steps)
    mach = np.zeros(n_steps)
    path_angle = np.zeros(n_steps)

    thrust = np.zeros(n_steps)
    drag = np.zeros(n_steps)
    q_arr = np.zeros(n_steps)
    stability = np.zeros(n_steps)
    aoa = np.zeros(n_steps)

    # Initial State [x, z, v, gamma]
    # gamma in radians from horizontal
    launch_rad = np.radians(launch_angle_deg)
    y = np.array([0.0, 0.0, 0.001, launch_rad])
    t = 0.0

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

    has_lifted = False
    has_burnout = False
    has_apogee = False

    A_ref = rocket.reference_area

    for i in range(n_steps):
        time[i] = t
        downrange[i] = y[0]
        altitude[i] = max(0, y[1])
        velocity[i] = y[2]
        path_angle[i] = np.degrees(y[3])

        # Atmosphere
        atm = get_atmosphere(altitude[i])
        rho = atm.density
        a = atm.speed_of_sound
        P_amb = atm.pressure

        # Mach
        M = abs(velocity[i]) / a if a > 0 else 0
        mach[i] = M

        # Mass
        m = rocket.get_mass_at_time(t)

        # Thrust
        if t < burn_time:
            F_thrust = thrust_vac - P_amb * exit_area
            thrust[i] = max(0, F_thrust)
        else:
            thrust[i] = 0.0
            if not has_burnout:
                has_burnout = True
                burnout_time = t
                burnout_alt = altitude[i]
                burnout_vel = velocity[i]

        # Drag
        cd = calculate_drag_coefficient(rocket, M)
        q = 0.5 * rho * velocity[i]**2
        q_arr[i] = q
        max_q = max(max_q, q)
        drag[i] = q * cd * A_ref

        # Stability margin calculation
        # Only calculate every 10th step for performance (stability changes slowly)
        if i % 10 == 0:
            try:
                stab = calculate_stability_margin(rocket, t)
            except Exception:
                stab = 2.0  # Safe fallback if calculation fails
        stability[i] = stab

        # Angle of Attack (Simple Wind Model)
        # alpha = atan(V_wind_normal / V_rocket)
        if velocity[i] > 1.0 and wind_speed > 0:
             # Wind shears from 0 to wind_speed at 1000m
            wind_at_alt = wind_speed * min(1.0, altitude[i] / 1000.0)

            # Simple assumption: Wind is purely horizontal, headwind
            # Rocket vector: (v cos gamma, v sin gamma)
            # Wind vector: (-w, 0)
            # Relative wind: (-w - v cos gamma, -v sin gamma)
            # This is complex in 2D, simplified here to "Wind Induced AoA"
            flight_gamma = y[3]
            # Angle of relative velocity vector
            v_rel_x = velocity[i] * np.cos(flight_gamma) + wind_at_alt
            v_rel_z = velocity[i] * np.sin(flight_gamma)

            gamma_eff = np.arctan2(v_rel_z, v_rel_x)
            alpha = np.degrees(flight_gamma - gamma_eff)
            aoa[i] = alpha
        else:
            aoa[i] = 0.0

        # Liftoff Logic
        if not has_lifted:
            weight = m * G0
            if thrust[i] > weight * 1.05:  # 5% T/W margin
                has_lifted = True
                liftoff_time = t

        # Acceleration (Scalar along path) - use altitude-dependent gravity
        g_local = calculate_gravity(altitude[i])
        accel = (thrust[i] - drag[i]) / m - g_local * np.sin(y[3])
        acceleration[i] = accel
        max_accel = max(max_accel, abs(accel) / G0)  # Normalize to sea-level G

        max_vel = max(max_vel, velocity[i])
        max_mach_val = max(max_mach_val, M)

        # Apogee
        if has_lifted and y[3] <= 0 and not has_apogee:
            # Path angle crosses 0 -> Apogee
            has_apogee = True
            apogee_time = t
            apogee_alt = altitude[i]

        # Impact
        if has_lifted and y[1] <= 0:
            # Crash - Truncate ALL arrays to same length
            time = time[:i+1]
            altitude = altitude[:i+1]
            downrange = downrange[:i+1]
            velocity = velocity[:i+1]
            path_angle = path_angle[:i+1]
            mach = mach[:i+1]
            acceleration = acceleration[:i+1]
            thrust = thrust[:i+1]
            drag = drag[:i+1]
            q_arr = q_arr[:i+1]
            stability = stability[:i+1]
            aoa = aoa[:i+1]
            break

        # Integration Step
        if has_lifted:
            y = _rk4_step_2d(y, t, dt, m, thrust[i], cd, rho, A_ref, rail_length, launch_rad)

        t += dt

    return FlightResult(
        time=time,
        altitude=altitude,
        range=downrange,
        velocity=velocity,
        acceleration=acceleration,
        mach=mach,
        path_angle=path_angle,
        thrust=thrust,
        drag=drag,
        q=q_arr,
        stability_margin=stability,
        angle_of_attack=aoa,
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
        max_aoa=np.max(np.abs(aoa)) if len(aoa) > 0 else 0,
        success=True
    )
