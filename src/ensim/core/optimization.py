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

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar


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


@dataclass(frozen=True)
class GravityTurnEvaluation:
    """Terminal state and path constraints from a planar powered ascent."""

    final_altitude: float
    final_velocity: float
    final_horizontal_velocity: float
    final_vertical_velocity: float
    final_flight_path_angle: float
    downrange: float
    max_dynamic_pressure: float
    max_acceleration: float
    min_altitude: float
    burn_time: float


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
    method: str = "SLSQP",
    reference_area: float = 0.0,
    drag_coefficient: float = 0.5,
    integration_step: float = 0.25,
) -> OptimizationResult:
    """
    Optimize gravity turn trajectory parameters.

    Finds a three-parameter pitch program that minimizes normalized terminal
    insertion error while satisfying path constraints. The evaluation model is
    planar 3-DOF with propellant depletion, inverse-square gravity, the EnSim
    atmosphere, and optional constant-Cd drag.

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
    if (
        vehicle_mass <= 0.0
        or thrust <= 0.0
        or isp <= 0.0
        or propellant_mass <= 0.0
        or propellant_mass >= vehicle_mass
    ):
        raise ValueError("Vehicle, propulsion, and propellant masses are invalid")
    if reference_area < 0.0 or drag_coefficient < 0.0 or integration_step <= 0.0:
        raise ValueError("Aerodynamic inputs and integration step are invalid")

    x0 = np.array([kickoff_altitude, kickoff_angle, initial_pitch_rate])

    # Parameter bounds
    bounds = [
        (100.0, 2000.0),  # Kickoff altitude (m)
        (1.0, 15.0),  # Kickoff angle (deg)
        (0.01, 3.0),  # Pitch rate after kick (deg/s)
    ]

    history = []
    evaluation_cache: dict[tuple[float, ...], GravityTurnEvaluation] = {}

    def evaluate(x) -> GravityTurnEvaluation:
        key = tuple(float(value) for value in np.round(x, 12))
        if key not in evaluation_cache:
            evaluation_cache[key] = _evaluate_gravity_turn(
                vehicle_mass=vehicle_mass,
                thrust=thrust,
                isp=isp,
                propellant_mass=propellant_mass,
                kickoff_altitude=float(x[0]),
                kickoff_angle_deg=float(x[1]),
                pitch_rate_deg_s=float(x[2]),
                reference_area=reference_area,
                drag_coefficient=drag_coefficient,
                integration_step=integration_step,
            )
        return evaluation_cache[key]

    def objective(x):
        evaluation = evaluate(x)
        altitude_scale = max(constraints.target_altitude, 1.0)
        velocity_scale = max(constraints.target_velocity, 1.0)
        fpa_scale = np.radians(5.0)
        cost = (
            ((evaluation.final_altitude - constraints.target_altitude) / altitude_scale) ** 2
            + ((evaluation.final_velocity - constraints.target_velocity) / velocity_scale) ** 2
            + (
                (evaluation.final_flight_path_angle - constraints.target_flight_path_angle)
                / fpa_scale
            )
            ** 2
        )
        history.append(float(cost))
        return float(cost)

    def constraint_max_q(x):
        """Ensure max Q constraint."""
        return constraints.max_dynamic_pressure - evaluate(x).max_dynamic_pressure

    def constraint_max_accel(x):
        """Ensure acceleration constraint."""
        return constraints.max_acceleration - evaluate(x).max_acceleration

    def constraint_min_altitude(x):
        return evaluate(x).min_altitude - constraints.min_altitude

    scipy_constraints = [
        {"type": "ineq", "fun": constraint_max_q},
        {"type": "ineq", "fun": constraint_max_accel},
        {"type": "ineq", "fun": constraint_min_altitude},
    ]

    if method == "differential_evolution":

        def penalized_objective(x):
            evaluation = evaluate(x)
            violations = (
                max(0.0, evaluation.max_dynamic_pressure - constraints.max_dynamic_pressure)
                / max(constraints.max_dynamic_pressure, 1.0)
                + max(0.0, evaluation.max_acceleration - constraints.max_acceleration)
                / max(constraints.max_acceleration, 1.0)
                + max(0.0, constraints.min_altitude - evaluation.min_altitude)
                / max(abs(constraints.min_altitude), 1.0)
            )
            return objective(x) + 1000.0 * violations**2

        result = differential_evolution(
            penalized_objective, bounds=bounds, maxiter=100, seed=42, polish=True
        )
    else:
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 100},
        )

    optimum = evaluate(result.x)
    constraints_satisfied = (
        optimum.max_dynamic_pressure <= constraints.max_dynamic_pressure * (1.0 + 1e-8)
        and optimum.max_acceleration <= constraints.max_acceleration * (1.0 + 1e-8)
        and optimum.min_altitude >= constraints.min_altitude - 1e-8
    )
    return OptimizationResult(
        success=result.success,
        optimal_value=result.fun,
        optimal_params={
            "kickoff_altitude": result.x[0],
            "kickoff_angle": result.x[1],
            "pitch_rate_deg_s": result.x[2],
            "final_altitude": optimum.final_altitude,
            "final_velocity": optimum.final_velocity,
            "final_horizontal_velocity": optimum.final_horizontal_velocity,
            "final_vertical_velocity": optimum.final_vertical_velocity,
            "final_flight_path_angle_deg": np.degrees(optimum.final_flight_path_angle),
            "downrange": optimum.downrange,
            "max_dynamic_pressure": optimum.max_dynamic_pressure,
            "max_acceleration_g": optimum.max_acceleration,
            "burn_time": optimum.burn_time,
            "constraints_satisfied": constraints_satisfied,
            "model": "planar_3dof_powered_ascent",
        },
        iterations=result.nit if hasattr(result, "nit") else len(history),
        message=result.message if hasattr(result, "message") else "Optimization complete",
        convergence_history=history,
    )


def _evaluate_gravity_turn(
    *,
    vehicle_mass: float,
    thrust: float,
    isp: float,
    propellant_mass: float,
    kickoff_altitude: float,
    kickoff_angle_deg: float,
    pitch_rate_deg_s: float,
    reference_area: float,
    drag_coefficient: float,
    integration_step: float,
) -> GravityTurnEvaluation:
    """Integrate a parameterized planar powered ascent with RK4."""
    from ensim.core.mission import get_atmosphere

    earth_radius = 6_371_000.0
    standard_gravity = 9.80665
    mass_flow = thrust / (isp * standard_gravity)
    burn_time = propellant_mass / mass_flow
    state = np.array([0.0, 0.0, 0.0, 0.0, vehicle_mass])
    time = 0.0
    kickoff_time: float | None = None
    max_q = 0.0
    max_acceleration = 0.0
    min_altitude = 0.0

    def thrust_angle(elapsed: float, altitude: float) -> float:
        nonlocal kickoff_time
        if altitude < kickoff_altitude:
            return np.pi / 2.0
        if kickoff_time is None:
            kickoff_time = elapsed
        degrees = 90.0 - kickoff_angle_deg - pitch_rate_deg_s * (elapsed - kickoff_time)
        return np.radians(np.clip(degrees, 0.0, 90.0))

    def derivative(local_state: np.ndarray, angle: float) -> np.ndarray:
        _, altitude, velocity_x, velocity_z, mass = local_state
        atmosphere = get_atmosphere(max(0.0, altitude))
        speed = float(np.hypot(velocity_x, velocity_z))
        dynamic_pressure = 0.5 * atmosphere.density * speed**2
        drag = dynamic_pressure * drag_coefficient * reference_area
        drag_x = drag * velocity_x / speed if speed > 1e-12 else 0.0
        drag_z = drag * velocity_z / speed if speed > 1e-12 else 0.0
        gravity = standard_gravity * (earth_radius / (earth_radius + max(altitude, 0.0))) ** 2
        return np.array(
            [
                velocity_x,
                velocity_z,
                thrust * np.cos(angle) / mass - drag_x / mass,
                thrust * np.sin(angle) / mass - drag_z / mass - gravity,
                -mass_flow,
            ]
        )

    while time < burn_time - 1e-12:
        step = min(integration_step, burn_time - time)
        angle = thrust_angle(time, state[1])
        k1 = derivative(state, angle)
        k2 = derivative(state + 0.5 * step * k1, angle)
        k3 = derivative(state + 0.5 * step * k2, angle)
        k4 = derivative(state + step * k3, angle)
        state += step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += step
        speed = float(np.hypot(state[2], state[3]))
        atmosphere = get_atmosphere(max(0.0, state[1]))
        max_q = max(max_q, 0.5 * atmosphere.density * speed**2)
        drag = 0.5 * atmosphere.density * speed**2 * drag_coefficient * reference_area
        max_acceleration = max(max_acceleration, (thrust - drag) / state[4] / standard_gravity)
        min_altitude = min(min_altitude, state[1])

    final_speed = float(np.hypot(state[2], state[3]))
    final_fpa = float(np.arctan2(state[3], state[2]))
    return GravityTurnEvaluation(
        final_altitude=float(state[1]),
        final_velocity=final_speed,
        final_horizontal_velocity=float(state[2]),
        final_vertical_velocity=float(state[3]),
        final_flight_path_angle=final_fpa,
        downrange=float(state[0]),
        max_dynamic_pressure=max_q,
        max_acceleration=max_acceleration,
        min_altitude=min_altitude,
        burn_time=burn_time,
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
    weight_sealevel: float = 0.3,
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
    from ensim.core.propulsion import (
        calculate_exit_conditions,
        calculate_thrust_coefficient,
        solve_mach_from_area_ratio_supersonic,
    )

    if chamber_pressure <= 0.0 or ambient_pressure < 0.0 or not 1.0 < gamma < 2.0:
        raise ValueError("Pressure and specific-heat-ratio inputs are invalid")
    if target_altitude is not None:
        if target_altitude < 0.0:
            raise ValueError("Target altitude cannot be negative")
        from ensim.core.mission import get_atmosphere

        ambient_pressure = get_atmosphere(target_altitude).pressure
    if weight_vacuum < 0.0 or weight_sealevel < 0.0:
        raise ValueError("Mission weights cannot be negative")
    weight_sum = weight_vacuum + weight_sealevel
    if weight_sum <= 0.0:
        raise ValueError("At least one mission weight must be positive")
    weight_vacuum /= weight_sum
    weight_sealevel /= weight_sum

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
            _, P_exit, _ = calculate_exit_conditions(gamma, 3500.0, chamber_pressure, M_exit)

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
        objective, bounds=(5.0, 300.0), method="bounded", options={"maxiter": 100, "xatol": 0.1}
    )

    optimal_ratio = result.x
    M_exit = solve_mach_from_area_ratio_supersonic(optimal_ratio, gamma)
    _, P_exit, _ = calculate_exit_conditions(gamma, 3500.0, chamber_pressure, M_exit)

    return OptimizationResult(
        success=result.success if hasattr(result, "success") else True,
        optimal_value=-result.fun,
        optimal_params={
            "area_ratio": optimal_ratio,
            "exit_mach": M_exit,
            "exit_pressure": P_exit,
            "pressure_ratio": P_exit / chamber_pressure,
        },
        iterations=result.nfev if hasattr(result, "nfev") else len(history),
        message="Nozzle optimization complete",
        convergence_history=history,
    )


# =============================================================================
# Multi-Stage Mass Optimization
# =============================================================================


def optimize_stage_mass_allocation(
    total_propellant: float,
    num_stages: int,
    payload_mass: float,
    stage_isps: list[float],
    structural_coefficients: list[float] | None = None,
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
    if total_propellant <= 0.0 or payload_mass < 0.0 or num_stages < 1:
        raise ValueError("Stage masses and stage count are invalid")
    if len(stage_isps) != num_stages or len(structural_coefficients) != num_stages:
        raise ValueError("One Isp and structural coefficient are required per stage")
    if any(isp <= 0.0 for isp in stage_isps):
        raise ValueError("Stage Isp values must be positive")
    if any(not 0.0 < coefficient < 1.0 for coefficient in structural_coefficients):
        raise ValueError("Structural coefficients must lie strictly between zero and one")
    if num_stages * 0.05 > 1.0 or num_stages * 0.8 < 1.0:
        raise ValueError("Stage count is incompatible with the allocation bounds")

    G0 = 9.80665
    history = []

    def objective(prop_fractions):
        """Objective: maximize total delta-v."""
        prop_fractions = np.array(prop_fractions)
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
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200},
    )

    optimal_fractions = result.x / np.sum(result.x)
    optimal_masses = optimal_fractions * total_propellant

    return OptimizationResult(
        success=result.success,
        optimal_value=-result.fun,
        optimal_params={
            "propellant_fractions": optimal_fractions.tolist(),
            "propellant_masses": optimal_masses.tolist(),
            "total_delta_v": -result.fun,
            "model": "ideal_impulsive_rocket_equation",
        },
        iterations=result.nit,
        message=result.message,
        convergence_history=history,
    )


# =============================================================================
# Engine Design Optimization
# =============================================================================


def optimize_engine_parameters(
    target_thrust: float,
    target_isp: float,
    propellant_type: str = "LOX/CH4",
    chamber_pressure_range: tuple[float, float] = (5e6, 30e6),
    mixture_ratio_range: tuple[float, float] = (2.5, 4.0),
    expansion_ratio: float = 40.0,
    ambient_pressure: float = 0.0,
) -> OptimizationResult:
    """
    Optimize engine chamber conditions for target performance.

    Evaluates each design with EnSim's Gibbs-equilibrium solver and ideal frozen
    nozzle model. The objective minimizes target-Isp error with a small,
    explicitly reported pressure regularization; thrust sets the throat area.

    Args:
        target_thrust: Desired thrust (N)
        target_isp: Desired specific impulse (s)
        propellant_type: Propellant combination
        chamber_pressure_range: (min, max) chamber pressure (Pa)
        mixture_ratio_range: (min, max) O/F ratio

    Returns:
        OptimizationResult with optimal engine parameters
    """
    from ensim.core.chemistry import CombustionProblem
    from ensim.core.propulsion import NozzleConditions, calculate_performance
    from ensim.utils.nasa_parser import load_default_database

    propellants = {
        "LOX/CH4": ("CH4", "O2"),
        "LOX/RP1": ("RP1", "O2"),
        "LOX/LH2": ("H2", "O2"),
    }
    if propellant_type not in propellants:
        raise ValueError(f"Unsupported propellant combination: {propellant_type}")
    if target_thrust <= 0.0 or target_isp <= 0.0:
        raise ValueError("Target thrust and Isp must be positive")
    if (
        chamber_pressure_range[0] <= 0.0
        or chamber_pressure_range[0] >= chamber_pressure_range[1]
        or mixture_ratio_range[0] <= 0.0
        or mixture_ratio_range[0] >= mixture_ratio_range[1]
        or expansion_ratio <= 1.0
        or ambient_pressure < 0.0
    ):
        raise ValueError("Engine optimization bounds are invalid")

    fuel, oxidizer = propellants[propellant_type]
    database = load_default_database()
    fuel_mw = database[fuel].molecular_weight
    oxidizer_mw = database[oxidizer].molecular_weight
    history = []
    evaluations: dict[tuple[float, float], tuple] = {}

    def evaluate(pressure_mpa: float, mixture_ratio: float):
        key = (round(pressure_mpa, 9), round(mixture_ratio, 9))
        if key not in evaluations:
            pressure = pressure_mpa * 1e6
            problem = CombustionProblem(database)
            problem.add_fuel(fuel, moles=1.0, temperature=298.15)
            problem.add_oxidizer(
                oxidizer,
                moles=mixture_ratio * fuel_mw / oxidizer_mw,
                temperature=298.15,
            )
            equilibrium = problem.solve(pressure=pressure)
            if not equilibrium.converged:
                raise RuntimeError("Equilibrium solver did not converge during optimization")
            performance = calculate_performance(
                T_chamber=equilibrium.temperature,
                P_chamber=pressure,
                gamma=equilibrium.gamma,
                mean_molecular_weight=equilibrium.mean_molecular_weight,
                nozzle=NozzleConditions(
                    area_ratio=expansion_ratio,
                    chamber_pressure=pressure,
                    ambient_pressure=ambient_pressure,
                ),
            )
            evaluations[key] = (equilibrium, performance)
        return evaluations[key]

    def objective(x):
        pressure_mpa, mixture_ratio = x
        _, performance = evaluate(pressure_mpa, mixture_ratio)
        isp_error = ((performance.isp - target_isp) / target_isp) ** 2
        pressure_fraction = (pressure_mpa - chamber_pressure_range[0] / 1e6) / (
            (chamber_pressure_range[1] - chamber_pressure_range[0]) / 1e6
        )
        cost = isp_error + 1e-4 * pressure_fraction**2
        history.append(cost)
        return cost

    bounds = [
        (chamber_pressure_range[0] / 1e6, chamber_pressure_range[1] / 1e6),
        mixture_ratio_range,
    ]
    result = minimize(
        objective,
        x0=np.array([np.mean(bounds[0]), np.mean(bounds[1])]),
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 35, "xatol": 2e-3, "fatol": 1e-8},
    )

    pressure_mpa, of_opt = result.x
    Pc_opt = pressure_mpa * 1e6
    equilibrium, performance = evaluate(pressure_mpa, of_opt)
    A_t = target_thrust / (Pc_opt * performance.c_f)

    return OptimizationResult(
        success=result.success,
        optimal_value=result.fun,
        optimal_params={
            "chamber_pressure": Pc_opt,
            "mixture_ratio": of_opt,
            "chamber_temperature": equilibrium.temperature,
            "throat_area": A_t,
            "throat_diameter": 2 * np.sqrt(A_t / np.pi),
            "estimated_isp": performance.isp,
            "c_star": performance.c_star,
            "thrust_coefficient": performance.c_f,
            "gamma": equilibrium.gamma,
            "mean_molecular_weight": equilibrium.mean_molecular_weight,
            "equilibrium_converged": equilibrium.converged,
            "element_balance_error": equilibrium.element_balance_error,
            "enthalpy_error": equilibrium.enthalpy_error,
            "expansion_ratio": expansion_ratio,
            "ambient_pressure": ambient_pressure,
            "model": "gibbs_equilibrium_frozen_nozzle",
            "pressure_regularization_weight": 1e-4,
        },
        iterations=result.nit if hasattr(result, "nit") else len(history),
        message="Engine optimization complete",
        convergence_history=history,
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
    payload_mass: float = 0.0,
    reserve_fraction: float = 0.0,
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
        reserve_fraction: Extra loaded fraction above the ideal requirement

    Returns:
        OptimizationResult with optimal propellant load
    """
    G0 = 9.80665
    if (
        dry_mass <= 0.0
        or tank_volume <= 0.0
        or propellant_density <= 0.0
        or target_delta_v < 0.0
        or isp <= 0.0
        or payload_mass < 0.0
        or reserve_fraction < 0.0
    ):
        raise ValueError("Propellant-load inputs are invalid")

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

    load_required = mp_required * (1.0 + reserve_fraction)
    if load_required > max_propellant:
        # Calculate achievable delta-v with max propellant
        mf = dry_mass + payload_mass
        m0 = mf + max_propellant
        achievable_dv = isp * G0 * np.log(m0 / mf)

        return OptimizationResult(
            success=False,
            optimal_value=max_propellant,
            optimal_params={
                "propellant_mass": max_propellant,
                "required_propellant": mp_required,
                "required_load_with_reserve": load_required,
                "reserve_fraction": reserve_fraction,
                "achievable_delta_v": achievable_dv,
                "delta_v_shortfall": target_delta_v - achievable_dv,
                "tank_utilization": 1.0,
            },
            iterations=1,
            message=f"Cannot achieve target ΔV. Shortfall: {target_delta_v - achievable_dv:.1f} m/s",
        )

    optimal_load = load_required

    # Actual delta-v with optimal load
    mf = dry_mass + payload_mass
    m0 = mf + optimal_load
    actual_dv = isp * G0 * np.log(m0 / mf)

    return OptimizationResult(
        success=True,
        optimal_value=optimal_load,
        optimal_params={
            "propellant_mass": optimal_load,
            "required_propellant": mp_required,
            "reserve_fraction": reserve_fraction,
            "achieved_delta_v": actual_dv,
            "delta_v_margin": actual_dv - target_delta_v,
            "tank_utilization": optimal_load / max_propellant,
            "mass_ratio": m0 / mf,
        },
        iterations=1,
        message="Propellant optimization successful",
    )
