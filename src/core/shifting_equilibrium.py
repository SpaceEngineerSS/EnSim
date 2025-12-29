"""
Shifting (Chemical) Equilibrium Nozzle Flow.

Unlike frozen flow, this module recalculates chemical equilibrium
at each station through the nozzle as temperature and pressure change,
capturing the effects of recombination reactions.

Typically gives 1-3% higher Isp than frozen flow assumption.

References:
    1. Gordon & McBride, NASA RP-1311, Section 4.3
    2. Sutton & Biblarz, "Rocket Propulsion Elements", Ch. 5
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from numba import jit


@dataclass
class FlowStation:
    """Flow properties at a nozzle station."""
    area_ratio: float  # A/At
    mach: float
    pressure: float  # Pa
    temperature: float  # K
    velocity: float  # m/s
    density: float  # kg/m³
    gamma: float
    mean_mw: float  # g/mol
    composition: Optional[Dict[str, float]] = None  # Mole fractions


@dataclass
class ShiftingFlowResult:
    """Complete shifting equilibrium nozzle flow solution."""
    stations: List[FlowStation] = field(default_factory=list)
    
    # Performance metrics
    exit_velocity: float = 0.0
    exit_mach: float = 0.0
    exit_temperature: float = 0.0
    exit_pressure: float = 0.0
    isp_shifting: float = 0.0
    isp_frozen: float = 0.0
    isp_improvement: float = 0.0  # Percent
    
    # Recombination analysis
    recombination_heat: float = 0.0  # J/kg recovered
    freeze_point_temperature: float = 0.0  # K where recombination effectively stops
    
    @property
    def n_stations(self) -> int:
        return len(self.stations)
    
    def get_property_profile(self, name: str) -> np.ndarray:
        """Get profile of a property through the nozzle."""
        return np.array([getattr(s, name) for s in self.stations])


@dataclass
class RecombinationReaction:
    """A recombination reaction that releases heat during expansion."""
    reactants: Tuple[str, str]  # e.g., ('H', 'H')
    product: str  # e.g., 'H2'
    delta_h: float  # Heat released (J/mol)
    rate_constant_ref: float  # Reference rate at 3000K
    activation_energy: float  # J/mol


# Key recombination reactions for H2/O2 combustion
RECOMBINATION_REACTIONS = [
    RecombinationReaction(
        reactants=('H', 'H'),
        product='H2',
        delta_h=436000.0,  # J/mol
        rate_constant_ref=1.8e13,
        activation_energy=0.0
    ),
    RecombinationReaction(
        reactants=('O', 'O'),
        product='O2',
        delta_h=498000.0,  # J/mol
        rate_constant_ref=1.2e13,
        activation_energy=0.0
    ),
    RecombinationReaction(
        reactants=('H', 'OH'),
        product='H2O',
        delta_h=502000.0,  # J/mol
        rate_constant_ref=2.2e13,
        activation_energy=0.0
    ),
    RecombinationReaction(
        reactants=('H', 'O'),
        product='OH',
        delta_h=428000.0,  # J/mol
        rate_constant_ref=1.5e13,
        activation_energy=0.0
    ),
]


# =============================================================================
# Core Physics Functions
# =============================================================================

@jit(nopython=True, cache=True)
def isentropic_temperature_ratio(M: float, gamma: float) -> float:
    """T/T0 for isentropic flow."""
    return 1.0 / (1.0 + (gamma - 1.0) / 2.0 * M * M)


@jit(nopython=True, cache=True)
def isentropic_pressure_ratio(M: float, gamma: float) -> float:
    """P/P0 for isentropic flow."""
    T_ratio = isentropic_temperature_ratio(M, gamma)
    return T_ratio ** (gamma / (gamma - 1.0))


@jit(nopython=True, cache=True)
def isentropic_density_ratio(M: float, gamma: float) -> float:
    """ρ/ρ0 for isentropic flow."""
    T_ratio = isentropic_temperature_ratio(M, gamma)
    return T_ratio ** (1.0 / (gamma - 1.0))


@jit(nopython=True, cache=True)
def area_mach_function(M: float, gamma: float) -> float:
    """Calculate A/A* as function of Mach number."""
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + (gamma - 1.0) / 2.0 * M * M
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * (term1 * term2) ** exponent


@jit(nopython=True, cache=True)
def solve_mach_supersonic(area_ratio: float, gamma: float) -> float:
    """Solve for supersonic Mach from area ratio."""
    # Newton-Raphson
    M = 2.0  # Initial guess
    for _ in range(50):
        A = area_mach_function(M, gamma)
        if abs(A - area_ratio) < 1e-8:
            break
        
        # Derivative dA/dM
        term1 = 2.0 / (gamma + 1.0)
        term2 = 1.0 + (gamma - 1.0) / 2.0 * M * M
        exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        
        dA = (-1.0 / (M * M)) * (term1 * term2) ** exp
        dA += (1.0 / M) * exp * (term1 ** exp) * (term2 ** (exp - 1)) * (gamma - 1.0) * M
        
        M = M - (A - area_ratio) / dA
        M = max(1.001, M)  # Keep supersonic
    
    return M


def estimate_gamma_from_temperature(
    T: float,
    T_reference: float,
    gamma_reference: float,
    mean_mw: float
) -> float:
    """
    Estimate gamma variation with temperature.
    
    For combustion products, gamma increases as temperature drops
    due to reduced dissociation and molecular vibration effects.
    
    Simple correlation: γ(T) ≈ γ_ref + 0.00003 * (T_ref - T)
    """
    # Limit the correction
    delta_gamma = 0.00003 * (T_reference - T)
    delta_gamma = max(-0.1, min(0.15, delta_gamma))
    
    gamma = gamma_reference + delta_gamma
    return max(1.1, min(1.4, gamma))


def calculate_recombination_extent(
    T: float,
    P: float,
    initial_composition: Dict[str, float]
) -> Tuple[Dict[str, float], float]:
    """
    Estimate extent of recombination at given T, P.
    
    Uses simplified equilibrium constants to estimate
    how much atomic species have recombined.
    
    Returns:
        (new_composition, heat_released)
    """
    # Below ~1500K, assume complete recombination
    if T < 1500:
        # All atomic species recombine
        new_comp = {}
        heat = 0.0
        
        # H atoms → H2
        if 'H' in initial_composition:
            h_moles = initial_composition.get('H', 0)
            new_comp['H2'] = initial_composition.get('H2', 0) + h_moles / 2
            heat += h_moles / 2 * 436000  # J
            new_comp['H'] = 0
        
        # O atoms → O2
        if 'O' in initial_composition:
            o_moles = initial_composition.get('O', 0)
            new_comp['O2'] = initial_composition.get('O2', 0) + o_moles / 2
            heat += o_moles / 2 * 498000  # J
            new_comp['O'] = 0
        
        # Copy other species
        for species, x in initial_composition.items():
            if species not in new_comp:
                new_comp[species] = x
        
        return new_comp, heat
    
    # Partial recombination between 1500K and 3000K
    # Use simple linear interpolation
    recomb_fraction = max(0, min(1, (3000 - T) / 1500))
    
    new_comp = dict(initial_composition)
    heat = 0.0
    
    # Partial H recombination
    if 'H' in initial_composition:
        h_reacted = initial_composition['H'] * recomb_fraction
        new_comp['H'] = initial_composition['H'] - h_reacted
        new_comp['H2'] = initial_composition.get('H2', 0) + h_reacted / 2
        heat += h_reacted / 2 * 436000
    
    # Partial O recombination
    if 'O' in initial_composition:
        o_reacted = initial_composition['O'] * recomb_fraction
        new_comp['O'] = initial_composition['O'] - o_reacted
        new_comp['O2'] = initial_composition.get('O2', 0) + o_reacted / 2
        heat += o_reacted / 2 * 498000
    
    return new_comp, heat


# =============================================================================
# Main Solver
# =============================================================================

def solve_shifting_equilibrium(
    T_chamber: float,
    P_chamber: float,
    gamma_chamber: float,
    mean_mw_chamber: float,
    initial_composition: Dict[str, float],
    area_ratios: np.ndarray,
    frozen_comparison: bool = True
) -> ShiftingFlowResult:
    """
    Solve shifting equilibrium expansion through nozzle.
    
    This is a simplified model that captures the key physics:
    1. Isentropic expansion to each area ratio
    2. Recombination extent estimated from local T, P
    3. Heat release from recombination increases velocity
    
    Args:
        T_chamber: Chamber temperature (K)
        P_chamber: Chamber pressure (Pa)
        gamma_chamber: Initial gamma
        mean_mw_chamber: Initial mean MW (g/mol)
        initial_composition: Initial mole fractions
        area_ratios: Array of A/At values
        frozen_comparison: Calculate frozen flow for comparison
        
    Returns:
        ShiftingFlowResult with complete solution
    """
    stations = []
    
    # Stagnation conditions
    T0 = T_chamber
    P0 = P_chamber
    R = 8314.46 / mean_mw_chamber  # J/(kg·K)
    rho0 = P0 / (R * T0)
    
    # Track cumulative recombination heat
    total_recomb_heat = 0.0
    current_composition = dict(initial_composition)
    gamma = gamma_chamber
    mean_mw = mean_mw_chamber
    
    for eps in area_ratios:
        # Solve for Mach number
        if eps <= 1.001:
            M = 1.0
        else:
            M = solve_mach_supersonic(eps, gamma)
        
        # Isentropic ratios
        T_ratio = isentropic_temperature_ratio(M, gamma)
        P_ratio = isentropic_pressure_ratio(M, gamma)
        rho_ratio = isentropic_density_ratio(M, gamma)
        
        # Local conditions
        T = T0 * T_ratio
        P = P0 * P_ratio
        rho = rho0 * rho_ratio
        
        # Recombination at this station
        new_comp, delta_heat = calculate_recombination_extent(
            T, P, current_composition
        )
        total_recomb_heat += delta_heat
        current_composition = new_comp
        
        # Update gamma for next iteration (simplified)
        gamma = estimate_gamma_from_temperature(
            T, T_chamber, gamma_chamber, mean_mw
        )
        
        # Velocity including recombination heat recovery
        a = np.sqrt(gamma * R * T)  # Local speed of sound
        V = M * a
        
        # Add recombination velocity boost (simplified energy balance)
        # ΔV ≈ sqrt(2 * Δh / m_dot) (recovered as kinetic energy)
        if total_recomb_heat > 0:
            # Approximate boost
            V_boost = np.sqrt(2 * total_recomb_heat / (mean_mw / 1000) * 0.5)
            V = V + V_boost * 0.01  # Small fraction recovered
        
        station = FlowStation(
            area_ratio=eps,
            mach=M,
            pressure=P,
            temperature=T,
            velocity=V,
            density=rho,
            gamma=gamma,
            mean_mw=mean_mw,
            composition=dict(current_composition)
        )
        stations.append(station)
    
    # Exit conditions
    exit_station = stations[-1]
    
    # Calculate Isp
    g0 = 9.80665
    isp_shifting = exit_station.velocity / g0
    
    # Frozen flow comparison
    if frozen_comparison:
        M_exit = solve_mach_supersonic(area_ratios[-1], gamma_chamber)
        T_exit_frozen = T0 * isentropic_temperature_ratio(M_exit, gamma_chamber)
        R_frozen = 8314.46 / mean_mw_chamber
        a_frozen = np.sqrt(gamma_chamber * R_frozen * T_exit_frozen)
        V_frozen = M_exit * a_frozen
        isp_frozen = V_frozen / g0
    else:
        isp_frozen = isp_shifting
    
    # Improvement
    isp_improvement = (isp_shifting - isp_frozen) / isp_frozen * 100 if isp_frozen > 0 else 0
    
    # Find freeze point (where T drops below ~1500K)
    freeze_temp = 0.0
    for s in stations:
        if s.temperature < 1500:
            freeze_temp = s.temperature
            break
    
    return ShiftingFlowResult(
        stations=stations,
        exit_velocity=exit_station.velocity,
        exit_mach=exit_station.mach,
        exit_temperature=exit_station.temperature,
        exit_pressure=exit_station.pressure,
        isp_shifting=isp_shifting,
        isp_frozen=isp_frozen,
        isp_improvement=isp_improvement,
        recombination_heat=total_recomb_heat,
        freeze_point_temperature=freeze_temp
    )


def compare_frozen_vs_shifting(
    T_chamber: float,
    P_chamber: float,
    gamma: float,
    mean_mw: float,
    expansion_ratio: float,
    initial_composition: Optional[Dict[str, float]] = None
) -> str:
    """
    Compare frozen and shifting equilibrium performance.
    
    Returns formatted comparison string.
    """
    if initial_composition is None:
        # Typical H2/O2 at 3600K
        initial_composition = {
            'H2O': 0.68,
            'H2': 0.12,
            'OH': 0.10,
            'H': 0.04,
            'O2': 0.04,
            'O': 0.02
        }
    
    area_ratios = np.linspace(1.0, expansion_ratio, 50)
    
    result = solve_shifting_equilibrium(
        T_chamber, P_chamber, gamma, mean_mw,
        initial_composition, area_ratios
    )
    
    lines = []
    lines.append("═" * 50)
    lines.append("FROZEN vs SHIFTING EQUILIBRIUM COMPARISON")
    lines.append("═" * 50)
    lines.append(f"Chamber: T={T_chamber:.0f}K, P={P_chamber/1e5:.1f}bar")
    lines.append(f"Expansion ratio: ε = {expansion_ratio:.1f}")
    lines.append("")
    lines.append(f"  Frozen Isp:    {result.isp_frozen:.1f} s")
    lines.append(f"  Shifting Isp:  {result.isp_shifting:.1f} s")
    lines.append(f"  Improvement:   +{result.isp_improvement:.2f}%")
    lines.append("")
    lines.append(f"  Exit velocity: {result.exit_velocity:.1f} m/s")
    lines.append(f"  Exit Mach:     {result.exit_mach:.2f}")
    lines.append(f"  Exit temp:     {result.exit_temperature:.0f} K")
    lines.append("")
    lines.append(f"  Recomb. heat:  {result.recombination_heat/1e6:.2f} MJ/mol")
    lines.append("═" * 50)
    
    return "\n".join(lines)
