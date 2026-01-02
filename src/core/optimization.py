"""
Trajectory and engine optimization module.

Provides optimization algorithms for:
- Gravity turn trajectory optimization
- Engine nozzle design optimization
- Multi-stage mass allocation
- Propellant load optimization

Uses gradient-based and evolutionary optimization methods.

References:
    - Bryson & Ho, "Applied Optimal Control"
    - Kirk, "Optimal Control Theory: An Introduction"
    - Betts, "Survey of Numerical Methods for Trajectory Optimization"
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    success: bool
    optimal_value: float
    optimal_params: dict
    iterations: int
    message: str
    convergence_history: list[float] | None = None


@dataclass
class TrajectoryConstraints:
    """
    Constraints for trajectory optimization.

    Attributes:
        max_dynamic_pressure: Maximum dynamic pressure (Pa)
        max_acceleration: Maximum acceleration (g's)
        min_altitude: Minimum altitude (m)
        target_altitude: Target orbit altitude (m)
        target_velocity: Target orbital velocity (m/s)
        target_flight_path_angle: Target FPA at insertion (rad)
    """
    max_dynamic_pressure: float = 35000.0  # ~35 kPa max Q
    max_acceleration: float = 6.0  # 6g max
    min_altitude: float = 0.0
    target_altitude: float = 200_000.0  # 200 km
    target_velocity: float = 7800.0  # ~7.8 km/s
    target_flight_path_angle: float = 0.0  # Horizontal at insertion


# =============================================================================
# Gravity Turn Optimization
# =============================================================================

def optimize_gravity_turn(
    vehicle_mass: float,
    thrust: float,
    isp: float,
    propellant_mass: float,
    constraints: TrajectoryConstraints | None = None,
    initial_pitch_rate: float = 0.5,  # deg/s
    kickoff_altitude: float = 500.0,  # m
    kickoff_angle: float = 5.0,  # deg
    method: str = "SLSQP"
) -> OptimizationResult:
    """
    Optimize gravity turn trajectory parameters.

    Finds optimal pitch program parameters to maximize payload
    to orbit while satisfying constraints.

    Args:
        vehicle_mass: Initial vehicle mass (kg)
        thrust: Engine thrust (N)
        isp: Specific impulse (s)
        propellant_mass: Available propellant (kg)
        constraints: Trajectory constraints
        initial_pitch_rate: Initial guess for pitch rate (deg/s)
        kickoff_altitude: Altitude to begin pitchover (m)
        kickoff_angle: Initial pitch kick angle (deg)
        method: Optimization method ('SLSQP', 'L-BFGS-B', 'differential_evolution')

    Returns:
        OptimizationResult with optimal parameters
    """
    if constraints is None:
        constraints = TrajectoryConstraints()

    # Initial guess: [kickoff_alt, kickoff_angle, pitch_rate_factor]
    x0 = np.array([kickoff_altitude, kickoff_angle, initial_pitch_rate])

    # Parameter bounds
    bounds = [
        (100.0, 2000.0),    # Kickoff altitude (m)
        (1.0, 15.0),       # Kickoff angle (deg)
        (0.1, 2.0)         # Pitch rate factor
    ]

    G0 = 9.80665
    history = []

    def objective(x):
        """Objective: minimize propellant usage for orbit insertion."""
        kickoff_alt, kickoff_ang, pitch_rate = x

        # Simplified trajectory model
        # This is a proxy for full 6-DOF simulation

        # Gravity losses increase with longer burn
        burn_time = propellant_mass / (thrust / (isp * G0))
        avg_alt = constraints.target_altitude / 2

        # Estimate gravity loss
        gravity_loss = G0 * burn_time * 0.15  # ~15% of ideal

        # Estimate drag loss (higher kickoff = more drag loss)
        drag_loss = 100 + 0.1 * kickoff_alt + 2.0 * kickoff_ang

        # Estimate steering loss (more aggressive pitch = more loss)
        steering_loss = 50 * pitch_rate

        total_loss = gravity_loss + drag_loss + steering_loss
        history.append(total_loss)
        return total_loss

    def constraint_max_q(x):
        """Ensure max Q constraint."""
        kickoff_alt, kickoff_ang, pitch_rate = x
        # Higher kickoff and angle reduces peak Q
        q_estimate = 35000 - 5 * kickoff_alt - 500 * kickoff_ang
        return constraints.max_dynamic_pressure - q_estimate

    def constraint_max_accel(x):
        """Ensure acceleration constraint."""
        # Simplified: always satisfied for reasonable vehicles
        return constraints.max_acceleration * G0 - thrust / vehicle_mass

    scipy_constraints = [
        {'type': 'ineq', 'fun': constraint_max_q},
        {'type': 'ineq', 'fun': constraint_max_accel}
    ]

    if method == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=100,
            seed=42,
            polish=True
        )
    else:
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=scipy_constraints,
            options={'maxiter': 100}
        )

    return OptimizationResult(
        success=result.success,
        optimal_value=result.fun,
        optimal_params={
            'kickoff_altitude': result.x[0],
            'kickoff_angle': result.x[1],
            'pitch_rate_factor': result.x[2]
        },
        iterations=result.nit if hasattr(result, 'nit') else len(history),
        message=result.message if hasattr(result, 'message') else "Optimization complete",
        convergence_history=history
    )


# =============================================================================
# Nozzle Design Optimization
# =============================================================================

def optimize_nozzle_expansion_ratio(
    chamber_pressure: float,
    ambient_pressure: float,
    gamma: float = 1.2,
    target_altitude: float | None = None,
    weight_vacuum: float = 0.7,
    weight_sealevel: float = 0.3
) -> OptimizationResult:
    """
    Optimize nozzle expansion ratio for mission profile.

    Balances sea-level and vacuum performance based on
    mission requirements.

    Args:
        chamber_pressure: Chamber pressure (Pa)
        ambient_pressure: Sea-level ambient pressure (Pa)
        gamma: Specific heat ratio
        target_altitude: Primary operating altitude (m), None for weighted avg
        weight_vacuum: Weight for vacuum Isp optimization
        weight_sealevel: Weight for sea-level thrust optimization

    Returns:
        OptimizationResult with optimal expansion ratio
    """
    from src.core.propulsion import (
        calculate_thrust_coefficient,
        solve_mach_from_area_ratio_supersonic,
        calculate_exit_conditions
    )

    history = []

    def objective(area_ratio):
        """Objective: maximize weighted Cf."""
        if area_ratio < 1.5:
            return 1e6  # Invalid

        # Get exit Mach from area ratio
        try:
            M_exit = solve_mach_from_area_ratio_supersonic(area_ratio, gamma)
            if np.isnan(M_exit) or M_exit < 1:
                return 1e6

            # Calculate exit pressure
            _, P_exit, _ = calculate_exit_conditions(
                gamma, 3500.0, chamber_pressure, M_exit
            )

            # Vacuum Cf (no back-pressure)
            pr = P_exit / chamber_pressure
            Cf_vac = calculate_thrust_coefficient(gamma, pr, area_ratio, 0.0)

            # Sea-level Cf (with back-pressure)
            Cf_sl = calculate_thrust_coefficient(
                gamma, pr, area_ratio, ambient_pressure / chamber_pressure
            )

            # Weighted objective (negative because we minimize)
            weighted_cf = -(weight_vacuum * Cf_vac + weight_sealevel * Cf_sl)
            history.append(-weighted_cf)
            return weighted_cf

        except Exception:
            return 1e6

    # Search over reasonable expansion ratio range
    result = minimize_scalar(
        objective,
        bounds=(5.0, 300.0),
        method='bounded',
        options={'maxiter': 100, 'xatol': 0.1}
    )

    optimal_ratio = result.x
    M_exit = solve_mach_from_area_ratio_supersonic(optimal_ratio, gamma)
    _, P_exit, _ = calculate_exit_conditions(
        gamma, 3500.0, chamber_pressure, M_exit
    )

    return OptimizationResult(
        success=result.success if hasattr(result, 'success') else True,
        optimal_value=-result.fun,
        optimal_params={
            'area_ratio': optimal_ratio,
            'exit_mach': M_exit,
            'exit_pressure': P_exit,
            'pressure_ratio': P_exit / chamber_pressure
        },
        iterations=result.nfev if hasattr(result, 'nfev') else len(history),
        message="Nozzle optimization complete",
        convergence_history=history
    )


# =============================================================================
# Multi-Stage Mass Optimization
# =============================================================================

def optimize_stage_mass_allocation(
    total_propellant: float,
    num_stages: int,
    payload_mass: float,
    stage_isps: list[float],
    structural_coefficients: list[float] | None = None
) -> OptimizationResult:
    """
    Optimize propellant allocation between stages.

    Finds optimal mass distribution to maximize payload
    fraction or delta-v using Lagrange multiplier method.

    Args:
        total_propellant: Total propellant budget (kg)
        num_stages: Number of stages
        payload_mass: Payload mass (kg)
        stage_isps: Vacuum Isp of each stage (s)
        structural_coefficients: Structural coefficient (dry/total) per stage
            Default [0.1, 0.1, ...] if None

    Returns:
        OptimizationResult with optimal mass allocation
    """
    if structural_coefficients is None:
        structural_coefficients = [0.1] * num_stages

    G0 = 9.80665
    history = []

    def objective(prop_fractions):
        """Objective: maximize total delta-v."""
        # Ensure fractions sum to 1
        prop_fractions = np.array(prop_fractions)
        prop_fractions = prop_fractions / np.sum(prop_fractions)

        propellant_masses = prop_fractions * total_propellant

        total_dv = 0.0
        payload = payload_mass

        # Calculate from top stage down
        for i in range(num_stages - 1, -1, -1):
            mp = propellant_masses[i]
            eps = structural_coefficients[i]
            isp = stage_isps[i]

            # Dry mass from structural coefficient: eps = m_dry / (m_dry + m_prop)
            m_dry = eps * mp / (1 - eps) if eps < 1 else mp

            m_initial = m_dry + mp + payload
            m_final = m_dry + payload

            if m_final > 0 and m_initial > m_final:
                dv = isp * G0 * np.log(m_initial / m_final)
                total_dv += dv

            # This stage becomes payload for stage below
            payload = m_dry + mp + payload

        history.append(total_dv)
        return -total_dv  # Minimize negative = maximize

    # Initial guess: equal distribution
    x0 = np.ones(num_stages) / num_stages

    # Bounds: each fraction between 5% and 80%
    bounds = [(0.05, 0.8) for _ in range(num_stages)]

    # Constraint: fractions must sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200}
    )

    optimal_fractions = result.x / np.sum(result.x)
    optimal_masses = optimal_fractions * total_propellant

    return OptimizationResult(
        success=result.success,
        optimal_value=-result.fun,
        optimal_params={
            'propellant_fractions': optimal_fractions.tolist(),
            'propellant_masses': optimal_masses.tolist(),
            'total_delta_v': -result.fun
        },
        iterations=result.nit,
        message=result.message,
        convergence_history=history
    )


# =============================================================================
# Engine Design Optimization
# =============================================================================

def optimize_engine_parameters(
    target_thrust: float,
    target_isp: float,
    propellant_type: str = "LOX/CH4",
    chamber_pressure_range: tuple[float, float] = (5e6, 30e6),
    mixture_ratio_range: tuple[float, float] = (2.5, 4.0)
) -> OptimizationResult:
    """
    Optimize engine chamber conditions for target performance.

    Finds optimal chamber pressure and mixture ratio to achieve
    target thrust and Isp while minimizing chamber mass.

    Args:
        target_thrust: Desired thrust (N)
        target_isp: Desired specific impulse (s)
        propellant_type: Propellant combination
        chamber_pressure_range: (min, max) chamber pressure (Pa)
        mixture_ratio_range: (min, max) O/F ratio

    Returns:
        OptimizationResult with optimal engine parameters
    """
    from src.core.propulsion import calculate_c_star, GAS_CONSTANT

    # Propellant-specific correlations
    propellant_data = {
        "LOX/CH4": {"T_max": 3550.0, "of_opt": 3.5, "gamma": 1.15, "MW": 20.0},
        "LOX/RP1": {"T_max": 3670.0, "of_opt": 2.7, "gamma": 1.15, "MW": 23.0},
        "LOX/LH2": {"T_max": 3600.0, "of_opt": 6.0, "gamma": 1.14, "MW": 12.0},
    }

    data = propellant_data.get(propellant_type, propellant_data["LOX/CH4"])
    G0 = 9.80665
    history = []

    def objective(x):
        """Objective: minimize deviation from targets + chamber mass."""
        Pc, of = x

        # Estimate chamber temperature
        T_c = data["T_max"] - 50 * (of - data["of_opt"])**2

        # Estimate Isp
        R_spec = GAS_CONSTANT / (data["MW"] / 1000.0)
        c_star = calculate_c_star(data["gamma"], R_spec, T_c)

        # Vacuum Cf estimate for high expansion ratio
        Cf_vac = 1.9  # Typical for vacuum engine

        Isp_est = c_star * Cf_vac / G0

        # Throat area for target thrust
        A_t = target_thrust / (Pc * Cf_vac)

        # Chamber mass estimate (higher pressure = heavier)
        chamber_mass = 0.1 * Pc * A_t**0.5  # Simplified scaling

        # Objective: weighted sum of deviations
        isp_error = ((Isp_est - target_isp) / target_isp)**2
        mass_penalty = chamber_mass / 1000  # Normalize

        cost = isp_error + 0.1 * mass_penalty
        history.append(cost)
        return cost

    bounds = [chamber_pressure_range, mixture_ratio_range]

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=100,
        seed=42,
        polish=True
    )

    Pc_opt, of_opt = result.x
    T_c = data["T_max"] - 50 * (of_opt - data["of_opt"])**2
    R_spec = GAS_CONSTANT / (data["MW"] / 1000.0)
    c_star = calculate_c_star(data["gamma"], R_spec, T_c)
    Cf_vac = 1.9
    Isp_est = c_star * Cf_vac / G0
    A_t = target_thrust / (Pc_opt * Cf_vac)

    return OptimizationResult(
        success=result.success,
        optimal_value=result.fun,
        optimal_params={
            'chamber_pressure': Pc_opt,
            'mixture_ratio': of_opt,
            'chamber_temperature': T_c,
            'throat_area': A_t,
            'throat_diameter': 2 * np.sqrt(A_t / np.pi),
            'estimated_isp': Isp_est,
            'c_star': c_star
        },
        iterations=result.nit if hasattr(result, 'nit') else len(history),
        message="Engine optimization complete",
        convergence_history=history
    )


# =============================================================================
# Propellant Load Optimization
# =============================================================================

def optimize_propellant_load(
    dry_mass: float,
    tank_volume: float,
    propellant_density: float,
    target_delta_v: float,
    isp: float,
    payload_mass: float = 0.0
) -> OptimizationResult:
    """
    Optimize propellant load for mission requirements.

    Finds minimum propellant load to achieve target delta-v
    while respecting tank volume constraints.

    Args:
        dry_mass: Vehicle dry mass (kg)
        tank_volume: Available tank volume (m³)
        propellant_density: Propellant bulk density (kg/m³)
        target_delta_v: Required delta-v (m/s)
        isp: Engine specific impulse (s)
        payload_mass: Payload mass (kg)

    Returns:
        OptimizationResult with optimal propellant load
    """
    G0 = 9.80665

    # Maximum propellant from volume
    max_propellant = tank_volume * propellant_density

    def required_propellant(dv, m_payload, m_dry, isp_s):
        """Calculate propellant needed for given delta-v."""
        # From Tsiolkovsky: dv = Isp * g0 * ln(m0/mf)
        # m0 = mf + mp
        # mp = mf * (exp(dv/(Isp*g0)) - 1)
        mf = m_dry + m_payload
        mass_ratio = np.exp(dv / (isp_s * G0))
        mp = mf * (mass_ratio - 1)
        return mp

    mp_required = required_propellant(target_delta_v, payload_mass, dry_mass, isp)

    # Check feasibility
    if mp_required > max_propellant:
        # Calculate achievable delta-v with max propellant
        mf = dry_mass + payload_mass
        m0 = mf + max_propellant
        achievable_dv = isp * G0 * np.log(m0 / mf)

        return OptimizationResult(
            success=False,
            optimal_value=max_propellant,
            optimal_params={
                'propellant_mass': max_propellant,
                'required_propellant': mp_required,
                'achievable_delta_v': achievable_dv,
                'delta_v_shortfall': target_delta_v - achievable_dv,
                'tank_utilization': 1.0
            },
            iterations=1,
            message=f"Cannot achieve target ΔV. Shortfall: {target_delta_v - achievable_dv:.1f} m/s"
        )

    # Calculate optimal load with margin
    margin_factor = 1.05  # 5% margin
    optimal_load = min(mp_required * margin_factor, max_propellant)

    # Actual delta-v with optimal load
    mf = dry_mass + payload_mass
    m0 = mf + optimal_load
    actual_dv = isp * G0 * np.log(m0 / mf)

    return OptimizationResult(
        success=True,
        optimal_value=optimal_load,
        optimal_params={
            'propellant_mass': optimal_load,
            'required_propellant': mp_required,
            'achieved_delta_v': actual_dv,
            'delta_v_margin': actual_dv - target_delta_v,
            'tank_utilization': optimal_load / max_propellant,
            'mass_ratio': m0 / mf
        },
        iterations=1,
        message="Propellant optimization successful"
    )

