"""
Thermal Analysis Module - Bartz Heat Transfer Model.

Implements the Bartz equation for convective heat transfer
in rocket nozzles and calculates wall temperature profiles.
"""

from dataclasses import dataclass

import numpy as np
from numba import jit


@dataclass
class ThermalResult:
    """Results from thermal analysis."""
    x_position: np.ndarray  # m (axial position from throat)
    area_ratio: np.ndarray  # A/At
    heat_flux: np.ndarray  # W/m² (local heat flux)
    wall_temp_gas: np.ndarray  # K (gas-side wall temp)
    wall_temp_coolant: np.ndarray  # K (coolant-side wall temp)
    adiabatic_wall_temp: np.ndarray  # K
    h_gas: np.ndarray  # W/(m²·K) (heat transfer coeff)
    mach: np.ndarray  # Local Mach number

    # Thermal limits
    max_heat_flux: float  # W/m²
    max_wall_temp: float  # K
    critical_x: float | None  # Position where T > T_melt (if any)
    is_safe: bool  # True if max temp < material limit


@jit(nopython=True, cache=True)
def _bartz_sigma(M: float, gamma: float, T_wall_ratio: float) -> float:
    """
    Calculate Bartz sigma factor (boundary layer correction).

    σ = [0.5 × (Tw/Tc) × (1 + (γ-1)/2 × M²) + 0.5]^(-0.68) ×
        [1 + (γ-1)/2 × M²]^(-0.12)
    """
    gm1_2 = (gamma - 1.0) / 2.0
    M_factor = 1.0 + gm1_2 * M * M

    term1 = 0.5 * T_wall_ratio * M_factor + 0.5
    term2 = M_factor

    sigma = (term1 ** (-0.68)) * (term2 ** (-0.12))
    return sigma


@jit(nopython=True, cache=True)
def _bartz_core(
    Dt: float,
    mu: float,
    Cp: float,
    Pr: float,
    Pc: float,
    c_star: float,
    At: float,
    A: float,
    sigma: float
) -> float:
    """
    Core Bartz equation for heat transfer coefficient.

    h_g = (0.026/Dt^0.2) × (μ^0.2 × Cp / Pr^0.6) ×
          (Pc/c*)^0.8 × (At/A)^0.9 × σ

    Args:
        Dt: Throat diameter (m)
        mu: Dynamic viscosity (Pa·s)
        Cp: Specific heat at constant pressure (J/kg·K)
        Pr: Prandtl number
        Pc: Chamber pressure (Pa)
        c_star: Characteristic velocity (m/s)
        At: Throat area (m²)
        A: Local area (m²)
        sigma: Boundary layer correction factor

    Returns:
        h_g: Heat transfer coefficient (W/m²·K)
    """
    # Bartz equation components
    coeff = 0.026 / (Dt ** 0.2)
    prop_factor = (mu ** 0.2) * Cp / (Pr ** 0.6)
    press_factor = (Pc / c_star) ** 0.8
    area_factor = (At / A) ** 0.9

    h_g = coeff * prop_factor * press_factor * area_factor * sigma
    return h_g


def calculate_thermal_profile(
    T_chamber: float,
    P_chamber: float,
    c_star: float,
    gamma: float,
    throat_diameter: float,
    expansion_ratio: float,
    contraction_ratio: float = 3.0,
    wall_thickness: float = 0.003,  # m
    wall_conductivity: float = 385.0,  # W/(m·K) - Copper
    coolant_temp: float = 300.0,  # K
    coolant_htc: float = 10000.0,  # W/(m²·K) - typical regen cooling
    material_limit: float = 700.0,  # K
    n_points: int = 100
) -> ThermalResult:
    """
    Calculate thermal profile along the nozzle using Bartz equation.

    Args:
        T_chamber: Chamber temperature (K)
        P_chamber: Chamber pressure (Pa)
        c_star: Characteristic velocity (m/s)
        gamma: Ratio of specific heats
        throat_diameter: Throat diameter (m)
        expansion_ratio: Nozzle area ratio (Ae/At)
        contraction_ratio: Chamber to throat area ratio
        wall_thickness: Nozzle wall thickness (m)
        wall_conductivity: Thermal conductivity (W/m·K)
        coolant_temp: Coolant bulk temperature (K)
        coolant_htc: Coolant heat transfer coefficient (W/m²·K)
        material_limit: Max service temperature (K)
        n_points: Number of calculation points

    Returns:
        ThermalResult with full thermal profile
    """
    # Geometry
    Dt = throat_diameter
    At = np.pi * (Dt / 2) ** 2

    # Area ratio array from contraction through expansion
    # Negative x = chamber, positive x = nozzle divergent
    eps_chamber = contraction_ratio
    eps_exit = expansion_ratio

    # Create area ratio profile
    area_ratio = np.concatenate([
        np.linspace(eps_chamber, 1.0, n_points // 3),  # Convergent
        np.linspace(1.0, eps_exit, 2 * n_points // 3)  # Divergent
    ])
    n_total = len(area_ratio)

    # Axial position (approximate)
    # Using conical approximation: x = (D - Dt) / (2 * tan(15°))
    D = Dt * np.sqrt(area_ratio)
    x_position = (D - Dt) / (2 * np.tan(np.radians(15)))

    # Gas properties (approximate at chamber conditions)
    R_gas = 8314.0 / 18.0  # J/(kg·K) - assuming H2O products
    P_chamber / (R_gas * T_chamber)
    mu = 7.5e-5  # Pa·s typical for hot combustion gases
    Cp = gamma * R_gas / (gamma - 1)  # J/(kg·K)
    Pr = 0.72  # Prandtl number for combustion gases

    # Recovery factor for adiabatic wall temperature
    r = Pr ** (1/3)  # Turbulent recovery factor

    # Initialize arrays
    h_gas = np.zeros(n_total)
    T_aw = np.zeros(n_total)
    q = np.zeros(n_total)
    T_wall_gas = np.zeros(n_total)
    T_wall_coolant = np.zeros(n_total)
    mach = np.zeros(n_total)

    for i in range(n_total):
        eps = area_ratio[i]
        A = At * eps

        # Calculate local Mach number
        if eps < 1.0001:
            # Subsonic (convergent section)
            M = _solve_mach_subsonic(eps, gamma)
        else:
            # Supersonic (divergent section)
            M = _solve_mach_supersonic(eps, gamma)
        mach[i] = M

        # Local static temperature
        T_local = T_chamber / (1 + (gamma - 1) / 2 * M * M)

        # Adiabatic wall temperature
        T_aw[i] = T_local * (1 + r * (gamma - 1) / 2 * M * M)

        # Initial wall temp guess for sigma calculation
        T_wall_guess = 0.5 * (T_aw[i] + coolant_temp)
        T_wall_ratio = T_wall_guess / T_chamber

        # Bartz sigma factor
        sigma = _bartz_sigma(M, gamma, T_wall_ratio)

        # Heat transfer coefficient
        h_gas[i] = _bartz_core(Dt, mu, Cp, Pr, P_chamber, c_star, At, A, sigma)

        # 1D thermal resistance network:
        # q = (T_aw - T_coolant) / (1/h_gas + t/k + 1/h_coolant)
        R_gas = 1.0 / h_gas[i]
        R_wall = wall_thickness / wall_conductivity
        R_coolant = 1.0 / coolant_htc
        R_total = R_gas + R_wall + R_coolant

        q[i] = (T_aw[i] - coolant_temp) / R_total

        # Wall temperatures
        T_wall_gas[i] = T_aw[i] - q[i] / h_gas[i]
        T_wall_coolant[i] = coolant_temp + q[i] / coolant_htc

    # Find maximum values and critical locations
    max_q = np.max(q)
    max_T = np.max(T_wall_gas)

    # Check if temperature exceeds limit
    over_limit = T_wall_gas > material_limit
    if np.any(over_limit):
        critical_idx = np.argmax(over_limit)
        critical_x = x_position[critical_idx]
        is_safe = False
    else:
        critical_x = None
        is_safe = True

    return ThermalResult(
        x_position=x_position,
        area_ratio=area_ratio,
        heat_flux=q,
        wall_temp_gas=T_wall_gas,
        wall_temp_coolant=T_wall_coolant,
        adiabatic_wall_temp=T_aw,
        h_gas=h_gas,
        mach=mach,
        max_heat_flux=max_q,
        max_wall_temp=max_T,
        critical_x=critical_x,
        is_safe=is_safe
    )


@jit(nopython=True, cache=True)
def _solve_mach_subsonic(area_ratio: float, gamma: float) -> float:
    """Solve for subsonic Mach number from area ratio."""
    M = 0.5  # Initial guess
    for _ in range(50):
        gp1 = gamma + 1
        gm1 = gamma - 1
        A_At = (1/M) * ((2/gp1) * (1 + gm1/2 * M*M)) ** (gp1/(2*gm1))
        dA_dM = A_At * (-1/M + gm1*M / (1 + gm1/2*M*M))

        f = A_At - area_ratio
        M_new = M - f / dA_dM

        if M_new <= 0:
            M_new = 0.01
        if M_new >= 1:
            M_new = 0.99

        if abs(M_new - M) < 1e-8:
            break
        M = M_new
    return M


@jit(nopython=True, cache=True)
def _solve_mach_supersonic(area_ratio: float, gamma: float) -> float:
    """Solve for supersonic Mach number from area ratio."""
    M = 2.0  # Initial guess
    for _ in range(50):
        gp1 = gamma + 1
        gm1 = gamma - 1
        A_At = (1/M) * ((2/gp1) * (1 + gm1/2 * M*M)) ** (gp1/(2*gm1))
        dA_dM = A_At * (-1/M + gm1*M / (1 + gm1/2*M*M))

        f = A_At - area_ratio
        M_new = M - f / dA_dM

        if M_new <= 1:
            M_new = 1.01

        if abs(M_new - M) < 1e-8:
            break
        M = M_new
    return M
