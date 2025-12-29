"""
Rocket propulsion calculations for de Laval nozzle.

Implements 1-D isentropic compressible flow equations for:
- Characteristic velocity (C*)
- Exit velocity (Ve)
- Thrust (F)
- Specific impulse (Isp)

Assumes frozen flow (constant composition) through the nozzle.

References:
    - Sutton, G.P. & Biblarz, O. "Rocket Propulsion Elements", 9th ed.
    - NASA RP-1311 (Gordon & McBride)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import jit

from .constants import GAS_CONSTANT, G0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NozzleConditions:
    """
    Nozzle geometry and operating conditions.
    
    Attributes:
        area_ratio: Nozzle expansion ratio (Ae/At)
        chamber_pressure: Chamber pressure (Pa)
        ambient_pressure: Ambient pressure (Pa), 0 for vacuum
        throat_area: Throat area (m²), optional
    """
    area_ratio: float  # Ae/At
    chamber_pressure: float  # Pa
    ambient_pressure: float = 0.0  # Pa (0 = vacuum)
    throat_area: Optional[float] = None  # m²
    
    def __repr__(self) -> str:
        return (f"NozzleConditions(ε={self.area_ratio:.1f}, "
                f"Pc={self.chamber_pressure/1e5:.1f} bar, "
                f"Pa={self.ambient_pressure/1e5:.2f} bar)")


@dataclass
class PerformanceResult:
    """
    Rocket engine performance results.
    
    Attributes:
        c_star: Characteristic velocity (m/s)
        c_f: Thrust coefficient (dimensionless)
        exit_velocity: Nozzle exit velocity (m/s)
        exit_mach: Exit Mach number
        exit_pressure: Exit pressure (Pa)
        exit_temperature: Exit temperature (K)
        thrust: Thrust force (N)
        isp: Specific impulse (s)
        mass_flow_rate: Mass flow rate (kg/s), if throat area provided
    """
    c_star: float
    c_f: float
    exit_velocity: float
    exit_mach: float
    exit_pressure: float
    exit_temperature: float
    thrust: float
    isp: float
    mass_flow_rate: Optional[float] = None
    
    def __repr__(self) -> str:
        return (f"PerformanceResult(Isp={self.isp:.1f} s, "
                f"C*={self.c_star:.1f} m/s, "
                f"Ve={self.exit_velocity:.1f} m/s, "
                f"Cf={self.c_f:.3f})")


# =============================================================================
# Core Physics Functions (Numba Optimized)
# =============================================================================

@jit(nopython=True, cache=True)
def calculate_c_star(gamma: float, R_specific: float, T_chamber: float) -> float:
    """
    Calculate characteristic velocity C*.
    
    C* = sqrt(R * Tc) / Γ
    where Γ = sqrt(γ) * (2/(γ+1))^((γ+1)/(2*(γ-1)))
    
    Args:
        gamma: Ratio of specific heats (Cp/Cv)
        R_specific: Specific gas constant (J/(kg·K))
        T_chamber: Chamber temperature (K)
        
    Returns:
        Characteristic velocity (m/s)
    """
    term = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    capital_gamma = np.sqrt(gamma) * term
    
    c_star = np.sqrt(R_specific * T_chamber) / capital_gamma
    return c_star


@jit(nopython=True, cache=True)
def calculate_throat_conditions(
    gamma: float, 
    T_chamber: float, 
    P_chamber: float
) -> Tuple[float, float]:
    """
    Calculate conditions at the nozzle throat (M=1).
    
    T_t / T_c = 2 / (γ + 1)
    P_t / P_c = (2 / (γ + 1))^(γ / (γ - 1))
    
    Returns:
        (throat_temperature, throat_pressure) in (K, Pa)
    """
    T_ratio = 2.0 / (gamma + 1.0)
    P_ratio = T_ratio ** (gamma / (gamma - 1.0))
    
    T_throat = T_chamber * T_ratio
    P_throat = P_chamber * P_ratio
    
    return T_throat, P_throat


@jit(nopython=True, cache=True)
def area_mach_function(M: float, gamma: float) -> float:
    """
    Calculate A/A* as a function of Mach number.
    
    A/A* = (1/M) * [(2/(γ+1)) * (1 + (γ-1)/2 * M²)]^((γ+1)/(2*(γ-1)))
    """
    if M <= 0:
        return 1e30
    
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + (gamma - 1.0) / 2.0 * M * M
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    
    area_ratio = (1.0 / M) * (term1 * term2) ** exponent
    return area_ratio


@jit(nopython=True, cache=True)
def solve_mach_from_area_ratio_supersonic(
    area_ratio: float, 
    gamma: float,
    tol: float = 1e-8,
    max_iter: int = 50
) -> float:
    """
    Solve for supersonic Mach number given area ratio using Newton-Raphson.
    """
    M = 1.0 + 0.5 * (area_ratio - 1.0)
    if M < 1.1:
        M = 1.1
    if M > 10.0:
        M = 10.0
    
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    exponent = gp1 / (2.0 * gm1)
    
    for _ in range(max_iter):
        term = (2.0 / gp1) * (1.0 + gm1 / 2.0 * M * M)
        f = (1.0 / M) * term ** exponent - area_ratio
        
        df = -term ** exponent / (M * M) + \
             (exponent * gm1 * M * term ** (exponent - 1.0) * (2.0 / gp1)) / M
        
        if np.abs(df) < 1e-30:
            break
            
        dM = -f / df
        
        if np.abs(dM) > 0.5:
            dM = 0.5 * np.sign(dM)
        
        M_new = M + dM
        
        if M_new < 1.001:
            M_new = 1.001
        if M_new > 20.0:
            M_new = 20.0
            
        M = M_new
        
        if np.abs(f) < tol:
            break
    
    return M


@jit(nopython=True, cache=True)
def calculate_exit_conditions(
    gamma: float,
    T_chamber: float,
    P_chamber: float,
    exit_mach: float
) -> Tuple[float, float, float]:
    """
    Calculate exit conditions for given Mach number.
    
    Returns:
        (exit_temperature, exit_pressure, T_ratio)
    """
    T_ratio = 1.0 / (1.0 + (gamma - 1.0) / 2.0 * exit_mach * exit_mach)
    P_ratio = T_ratio ** (gamma / (gamma - 1.0))
    
    T_exit = T_chamber * T_ratio
    P_exit = P_chamber * P_ratio
    
    return T_exit, P_exit, T_ratio


@jit(nopython=True, cache=True)
def calculate_exit_velocity(
    gamma: float,
    R_specific: float,
    T_chamber: float,
    pressure_ratio: float  # Pe/Pc
) -> float:
    """
    Calculate nozzle exit velocity.
    
    V_e = sqrt(2 * γ/(γ-1) * R * T_c * [1 - (P_e/P_c)^((γ-1)/γ)])
    """
    if pressure_ratio <= 0 or pressure_ratio >= 1:
        return 0.0
    
    exponent = (gamma - 1.0) / gamma
    term = 1.0 - pressure_ratio ** exponent
    
    V_e = np.sqrt(2.0 * gamma / (gamma - 1.0) * R_specific * T_chamber * term)
    return V_e


@jit(nopython=True, cache=True)
def calculate_thrust_coefficient(
    gamma: float,
    pressure_ratio: float,  # Pe/Pc
    area_ratio: float,
    ambient_ratio: float  # Pa/Pc
) -> float:
    """
    Calculate thrust coefficient Cf.
    
    C_f = √(2γ²/(γ-1) * (2/(γ+1))^((γ+1)/(γ-1)) * [1 - (Pe/Pc)^((γ-1)/γ)])
          + ε * (Pe - Pa) / Pc
    """
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    
    term1 = 2.0 * gamma * gamma / gm1
    term2 = (2.0 / gp1) ** (gp1 / gm1)
    term3 = 1.0 - pressure_ratio ** (gm1 / gamma)
    
    Cf_momentum = np.sqrt(term1 * term2 * term3)
    Cf_pressure = area_ratio * (pressure_ratio - ambient_ratio)
    
    Cf = Cf_momentum + Cf_pressure
    return Cf


# =============================================================================
# High-Level Interface
# =============================================================================

def calculate_performance(
    T_chamber: float,
    P_chamber: float,
    gamma: float,
    mean_molecular_weight: float,
    nozzle: NozzleConditions,
    eta_cstar: float = 1.0,
    eta_cf: float = 1.0,
    alpha_deg: float = 15.0,
) -> PerformanceResult:
    """
    Calculate complete rocket engine performance.
    
    Args:
        T_chamber: Combustion chamber temperature (K)
        P_chamber: Chamber pressure (Pa)
        gamma: Ratio of specific heats of combustion products
        mean_molecular_weight: Mean molecular weight (g/mol)
        nozzle: Nozzle conditions (geometry, ambient pressure)
        eta_cstar: Combustion efficiency (0.5-1.0), default 1.0 (ideal)
        eta_cf: Nozzle efficiency (0.5-1.0), default 1.0 (ideal)
        alpha_deg: Nozzle half-angle in degrees (default 15°)
        
    Returns:
        PerformanceResult with all performance metrics
        
    Note:
        Real-world performance uses:
        - C*_real = C*_ideal × η_c*
        - λ = (1 + cos(α)) / 2  (divergence factor)
        - Cf_real = Cf_ideal × η_Cf × λ
        - Isp_real = (C*_real × Cf_real) / g0
    """
    # Specific gas constant (J/(kg·K))
    R_specific = GAS_CONSTANT / (mean_molecular_weight / 1000.0)
    
    # Characteristic velocity (ideal)
    c_star_ideal = calculate_c_star(gamma, R_specific, T_chamber)
    
    # Apply combustion efficiency
    c_star = c_star_ideal * eta_cstar
    
    # Exit Mach number from area ratio
    M_exit = solve_mach_from_area_ratio_supersonic(nozzle.area_ratio, gamma)
    
    # Exit conditions
    T_exit, P_exit, _ = calculate_exit_conditions(
        gamma, T_chamber, P_chamber, M_exit
    )
    
    # Pressure ratios
    pressure_ratio = P_exit / P_chamber
    ambient_ratio = nozzle.ambient_pressure / P_chamber
    
    # Exit velocity
    V_exit = calculate_exit_velocity(
        gamma, R_specific, T_chamber, pressure_ratio
    )
    
    # Thrust coefficient (ideal)
    Cf_ideal = calculate_thrust_coefficient(
        gamma, pressure_ratio, nozzle.area_ratio, ambient_ratio
    )
    
    # Divergence loss factor: λ = (1 + cos(α)) / 2
    # For conical nozzle, accounts for non-axial exit velocity component
    alpha_rad = np.radians(alpha_deg)
    divergence_factor = (1.0 + np.cos(alpha_rad)) / 2.0
    
    # Apply nozzle efficiency AND divergence loss
    Cf = Cf_ideal * eta_cf * divergence_factor
    
    # Specific impulse (real) - combines all efficiencies
    Isp = Cf * c_star / G0
    
    # Mass flow rate and thrust (if throat area provided)
    if nozzle.throat_area is not None:
        mass_flow = P_chamber * nozzle.throat_area / c_star
        # Thrust with efficiency and divergence corrections
        thrust = mass_flow * V_exit * eta_cf * divergence_factor + (P_exit - nozzle.ambient_pressure) * (
            nozzle.throat_area * nozzle.area_ratio
        ) * eta_cf * divergence_factor
    else:
        mass_flow = None
        thrust = Cf * P_chamber  # F/At if At not provided
    
    return PerformanceResult(
        c_star=c_star,
        c_f=Cf,
        exit_velocity=V_exit,
        exit_mach=M_exit,
        exit_pressure=P_exit,
        exit_temperature=T_exit,
        thrust=thrust,
        isp=Isp,
        mass_flow_rate=mass_flow,
    )


def calculate_ideal_expansion_ratio(
    gamma: float,
    pressure_ratio: float  # Pc/Pe
) -> float:
    """
    Calculate ideal expansion ratio for given pressure ratio.
    
    Uses isentropic flow relations to determine the nozzle area ratio
    needed for a given chamber-to-exit pressure ratio.
    
    Args:
        gamma: Ratio of specific heats
        pressure_ratio: Chamber to exit pressure ratio (Pc/Pe)
        
    Returns:
        Area ratio (Ae/At) for ideal expansion
        
    Reference:
        Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed., Eq. 3-25
    """
    Pe_Pc = 1.0 / pressure_ratio
    gm1 = gamma - 1.0
    
    # From isentropic relations:
    # M² = (2/(γ-1)) × [(Pe/Pc)^(-(γ-1)/γ) - 1]
    temp = Pe_Pc ** (-gm1 / gamma) - 1.0
    M_squared = 2.0 / gm1 * temp
    
    if M_squared < 0:
        return 1.0  # Subsonic, return unity area ratio
    
    M_exit = np.sqrt(M_squared)
    area_ratio = area_mach_function(M_exit, gamma)
    
    return area_ratio


def check_flow_separation(
    P_exit: float,
    P_ambient: float,
    criterion: str = "summerfield"
) -> tuple:
    """
    Check if nozzle flow separation will occur.
    
    Uses the Summerfield criterion: flow separates if Pe < 0.4 * Pa.
    Rocket nozzles can sustain over-expansion up to this limit.
    
    Args:
        P_exit: Nozzle exit pressure (Pa)
        P_ambient: Ambient pressure (Pa)
        criterion: "summerfield" (default) or "simple"
        
    Returns:
        Tuple of (will_separate: bool, margin: float, message: str)
        margin > 0 means safe, margin < 0 means separation likely
        
    Reference:
        Summerfield, M. (1951). "A Simple Criterion for Predicting the 
        Separation of Turbulent Boundary Layers in Presence of Pressure Rise"
        Journal of Aeronautical Sciences
    """
    if P_ambient <= 0:
        # Vacuum - no separation possible
        return False, 1.0, "Vacuum operation - no separation risk"
    
    if criterion == "summerfield":
        # Summerfield criterion: Pe/Pa > 0.4 for attached flow
        # Separation occurs when Pe < 0.4 * Pa
        separation_threshold = 0.4
        pressure_ratio = P_exit / P_ambient if P_ambient > 0 else float('inf')
        margin = pressure_ratio - separation_threshold
        
        if pressure_ratio < separation_threshold:
            return True, margin, f"Flow separation likely: Pe/Pa = {pressure_ratio:.3f} < 0.4"
        elif pressure_ratio < 0.6:
            return False, margin, f"Warning: Pe/Pa = {pressure_ratio:.3f} - approaching separation"
        else:
            return False, margin, f"Flow attached: Pe/Pa = {pressure_ratio:.3f}"
    else:
        # Simple criterion: Pe < Pa
        if P_exit < P_ambient:
            return True, P_exit - P_ambient, "Over-expanded: Pe < Pa"
        return False, P_exit - P_ambient, "Under-expanded or adapted"


@jit(nopython=True, cache=True)
def _compute_profile_arrays(
    gamma: float,
    T_chamber: float,
    P_chamber: float,
    area_ratios: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized profile computation.
    
    Returns:
        (mach_array, temp_array, pressure_array, velocity_array)
    """
    n = len(area_ratios)
    mach = np.zeros(n, dtype=np.float64)
    temp = np.zeros(n, dtype=np.float64)
    pressure = np.zeros(n, dtype=np.float64)
    velocity = np.zeros(n, dtype=np.float64)
    
    gm1 = gamma - 1.0
    
    for i in range(n):
        eps = area_ratios[i]
        
        # Solve for Mach number (supersonic branch)
        M = solve_mach_from_area_ratio_supersonic(eps, gamma)
        mach[i] = M
        
        # Isentropic relations
        T_ratio = 1.0 / (1.0 + gm1 / 2.0 * M * M)
        P_ratio = T_ratio ** (gamma / gm1)
        
        temp[i] = T_chamber * T_ratio
        pressure[i] = P_chamber * P_ratio
        
        # Local velocity (from Mach and sound speed)
        # a = sqrt(γRT), V = M*a
        # But we need R_specific which isn't passed, so use T ratio approach
        # V² = 2*γ/(γ-1) * R * Tc * (1 - T/Tc)
        velocity[i] = M * np.sqrt(gamma * T_ratio)  # Normalized by sqrt(γRTc)
    
    return mach, temp, pressure, velocity


def get_nozzle_profile(
    gamma: float,
    T_chamber: float,
    P_chamber: float,
    exit_area_ratio: float,
    R_specific: float = 500.0,
    n_points: int = 100,
) -> dict:
    """
    Generate nozzle flow property profiles for plotting.
    
    Computes Mach number, temperature, pressure, and velocity 
    at discrete stations along the nozzle from throat to exit.
    
    Args:
        gamma: Ratio of specific heats
        T_chamber: Chamber/stagnation temperature (K)
        P_chamber: Chamber/stagnation pressure (Pa)
        exit_area_ratio: Exit area ratio (Ae/At)
        R_specific: Specific gas constant (J/kg/K)
        n_points: Number of points in profile
        
    Returns:
        Dictionary with arrays:
            'area_ratio': A/A* from 1 to exit_area_ratio
            'x_norm': Normalized axial position (0=throat, 1=exit)
            'mach': Mach number
            'temperature': Static temperature (K)
            'pressure': Static pressure (Pa)
            'velocity': Flow velocity (m/s)
            'T_ratio': T/Tc
            'P_ratio': P/Pc
    """
    # Generate area ratio stations
    area_ratios = np.linspace(1.0, exit_area_ratio, n_points)
    
    # Compute profiles using Numba-optimized function
    mach, temp, pressure, _ = _compute_profile_arrays(
        gamma, T_chamber, P_chamber, area_ratios
    )
    
    # Compute actual velocities
    # V = M * sqrt(γ * R * T)
    velocity = mach * np.sqrt(gamma * R_specific * temp)
    
    # Normalized axial position (assuming conical nozzle: A/At = (r/rt)² = (1 + x*tan(θ))²)
    # Simplified: x_norm proportional to sqrt(A/At) - 1
    x_norm = (np.sqrt(area_ratios) - 1.0) / (np.sqrt(exit_area_ratio) - 1.0)
    
    return {
        'area_ratio': area_ratios,
        'x_norm': x_norm,
        'mach': mach,
        'temperature': temp,
        'pressure': pressure,
        'velocity': velocity,
        'T_ratio': temp / T_chamber,
        'P_ratio': pressure / P_chamber,
    }


def get_nozzle_geometry(
    exit_area_ratio: float,
    throat_radius: float = 0.05,
    convergent_length: float = 0.1,
    divergent_half_angle: float = 15.0,
    n_points: int = 50,
) -> dict:
    """
    Generate nozzle geometry coordinates for 3D visualization.
    
    Args:
        exit_area_ratio: Ae/At
        throat_radius: Throat radius (m)
        convergent_length: Length of convergent section (m)
        divergent_half_angle: Half-angle of conical divergent section (degrees)
        n_points: Points per section
        
    Returns:
        Dictionary with:
            'x': Axial coordinates (m), throat at x=0
            'r': Radius at each x (m)
            'area_ratio': A/At at each station
    """
    # Exit radius from area ratio
    exit_radius = throat_radius * np.sqrt(exit_area_ratio)
    
    # Divergent length from geometry
    angle_rad = np.deg2rad(divergent_half_angle)
    divergent_length = (exit_radius - throat_radius) / np.tan(angle_rad)
    
    # Convergent section (simple conical, 30° half-angle)
    conv_angle = np.deg2rad(30.0)
    inlet_radius = throat_radius + convergent_length * np.tan(conv_angle)
    
    # Generate points
    x_conv = np.linspace(-convergent_length, 0, n_points)
    r_conv = throat_radius + (-x_conv) * np.tan(conv_angle)
    
    x_div = np.linspace(0, divergent_length, n_points)[1:]  # Skip throat (already in conv)
    r_div = throat_radius + x_div * np.tan(angle_rad)
    
    # Combine
    x = np.concatenate([x_conv, x_div])
    r = np.concatenate([r_conv, r_div])
    area_ratio = (r / throat_radius) ** 2
    
    return {
        'x': x,
        'r': r,
        'area_ratio': area_ratio,
        'throat_radius': throat_radius,
        'exit_radius': exit_radius,
        'divergent_length': divergent_length,
    }

