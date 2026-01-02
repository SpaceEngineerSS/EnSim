"""
Thermodynamic property calculations using NASA 7-term polynomials.

This module provides Numba-accelerated functions for computing:
- Heat capacity (Cp/R)
- Enthalpy (H/RT)
- Entropy (S/R)
- Gibbs free energy (G/RT)

All core functions accept raw numpy arrays to maintain Numba nopython compatibility.
String handling and coefficient selection should be done outside these functions.

References:
    - Gordon, S. & McBride, B.J. (1994). "Computer Program for Calculation
      of Complex Chemical Equilibrium Compositions and Applications"
      NASA Reference Publication 1311.
    - McBride, B.J., Zehe, M.J., & Gordon, S. (2002). "NASA Glenn Coefficients
      for Calculating Thermodynamic Properties of Individual Species"
      NASA/TP-2002-211556.
"""

import numpy as np
from numba import jit
from numpy.typing import NDArray

from .constants import GAS_CONSTANT


@jit(nopython=True, cache=True)
def cp_over_r(T: float, coeffs: NDArray[np.float64]) -> float:
    """
    Calculate dimensionless heat capacity Cp/R from NASA polynomial.

    Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4

    Args:
        T: Temperature in Kelvin
        coeffs: 7-element coefficient array [a1, a2, a3, a4, a5, a6, a7]

    Returns:
        Cp/R (dimensionless)
    """
    a1, a2, a3, a4, a5 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]

    return a1 + a2 * T + a3 * T**2 + a4 * T**3 + a5 * T**4


@jit(nopython=True, cache=True)
def h_over_rt(T: float, coeffs: NDArray[np.float64]) -> float:
    """
    Calculate dimensionless enthalpy H/(R*T) from NASA polynomial.

    H/(RT) = a1 + (a2/2)*T + (a3/3)*T^2 + (a4/4)*T^3 + (a5/5)*T^4 + a6/T

    The a6 coefficient is the integration constant that includes
    the heat of formation at the reference temperature.

    Args:
        T: Temperature in Kelvin
        coeffs: 7-element coefficient array [a1, a2, a3, a4, a5, a6, a7]

    Returns:
        H/(R*T) (dimensionless)
    """
    a1 = coeffs[0]
    a2 = coeffs[1]
    a3 = coeffs[2]
    a4 = coeffs[3]
    a5 = coeffs[4]
    a6 = coeffs[5]

    return (
        a1
        + (a2 / 2.0) * T
        + (a3 / 3.0) * T**2
        + (a4 / 4.0) * T**3
        + (a5 / 5.0) * T**4
        + a6 / T
    )


@jit(nopython=True, cache=True)
def s_over_r(T: float, coeffs: NDArray[np.float64]) -> float:
    """
    Calculate dimensionless entropy S/R from NASA polynomial.

    S/R = a1*ln(T) + a2*T + (a3/2)*T^2 + (a4/3)*T^3 + (a5/4)*T^4 + a7

    The a7 coefficient is the integration constant for entropy.

    Args:
        T: Temperature in Kelvin
        coeffs: 7-element coefficient array [a1, a2, a3, a4, a5, a6, a7]

    Returns:
        S/R (dimensionless)
    """
    a1 = coeffs[0]
    a2 = coeffs[1]
    a3 = coeffs[2]
    a4 = coeffs[3]
    a5 = coeffs[4]
    a7 = coeffs[6]

    return (
        a1 * np.log(T)
        + a2 * T
        + (a3 / 2.0) * T**2
        + (a4 / 3.0) * T**3
        + (a5 / 4.0) * T**4
        + a7
    )


@jit(nopython=True, cache=True)
def g_over_rt(T: float, coeffs: NDArray[np.float64]) -> float:
    """
    Calculate dimensionless Gibbs free energy G/(R*T) from NASA polynomial.

    G/(RT) = H/(RT) - S/R

    This is the key quantity for chemical equilibrium calculations
    via Gibbs free energy minimization.

    Args:
        T: Temperature in Kelvin
        coeffs: 7-element coefficient array [a1, a2, a3, a4, a5, a6, a7]

    Returns:
        G/(R*T) (dimensionless)
    """
    return h_over_rt(T, coeffs) - s_over_r(T, coeffs)


@jit(nopython=True, cache=True)
def select_coefficients(
    T: float,
    coeffs_low: NDArray[np.float64],
    coeffs_high: NDArray[np.float64],
    T_mid: float = 1000.0
) -> NDArray[np.float64]:
    """
    Select appropriate coefficient set based on temperature.

    NASA polynomials use piecewise fits:
    - Low temperature set: T_low <= T < T_mid
    - High temperature set: T_mid <= T <= T_high

    Args:
        T: Temperature in Kelvin
        coeffs_low: Low-T coefficients [a1..a7]
        coeffs_high: High-T coefficients [a1..a7]
        T_mid: Transition temperature (typically 1000 K)

    Returns:
        Selected coefficient array
    """
    if T_mid <= T:
        return coeffs_high
    return coeffs_low


@jit(nopython=True, cache=True)
def get_thermo_properties(
    T: float,
    coeffs_low: NDArray[np.float64],
    coeffs_high: NDArray[np.float64],
    T_mid: float = 1000.0
) -> tuple:
    """
    Calculate all thermodynamic properties at given temperature.

    This is the main workhorse function for equilibrium calculations.
    Returns all properties in a single call for efficiency.

    Args:
        T: Temperature in Kelvin
        coeffs_low: Low-T coefficients [a1..a7]
        coeffs_high: High-T coefficients [a1..a7]
        T_mid: Transition temperature (typically 1000 K)

    Returns:
        Tuple of (Cp/R, H/RT, S/R, G/RT)
    """
    coeffs = select_coefficients(T, coeffs_low, coeffs_high, T_mid)

    cp_r = cp_over_r(T, coeffs)
    h_rt = h_over_rt(T, coeffs)
    s_r = s_over_r(T, coeffs)
    g_rt = h_rt - s_r

    return cp_r, h_rt, s_r, g_rt


# Non-JIT wrapper functions for convenience (handle SpeciesData objects)

def calculate_cp(T: float, species) -> float:
    """
    Calculate heat capacity Cp in J/(mol·K).

    Args:
        T: Temperature in Kelvin
        species: SpeciesData object

    Returns:
        Cp in J/(mol·K)
    """
    coeffs = species.get_coeffs_for_temp(T)
    return cp_over_r(T, coeffs) * GAS_CONSTANT


def calculate_enthalpy(T: float, species) -> float:
    """
    Calculate enthalpy H in J/mol.

    Args:
        T: Temperature in Kelvin
        species: SpeciesData object

    Returns:
        H in J/mol
    """
    coeffs = species.get_coeffs_for_temp(T)
    return h_over_rt(T, coeffs) * GAS_CONSTANT * T


def calculate_entropy(T: float, species, P: float = 101325.0) -> float:
    """
    Calculate entropy S in J/(mol·K).

    Note: For ideal gases, S depends on pressure:
    S(T, P) = S°(T) - R * ln(P/P°)

    Args:
        T: Temperature in Kelvin
        species: SpeciesData object
        P: Pressure in Pa (default: 1 atm)

    Returns:
        S in J/(mol·K)
    """
    coeffs = species.get_coeffs_for_temp(T)
    s0_r = s_over_r(T, coeffs)

    # Standard pressure correction
    P_ref = 101325.0  # Pa
    s_r = s0_r - np.log(P / P_ref)

    return s_r * GAS_CONSTANT


def calculate_gibbs(T: float, species, P: float = 101325.0) -> float:
    """
    Calculate Gibbs free energy G in J/mol.

    G = H - T*S

    Args:
        T: Temperature in Kelvin
        species: SpeciesData object
        P: Pressure in Pa (default: 1 atm)

    Returns:
        G in J/mol
    """
    h = calculate_enthalpy(T, species)
    s = calculate_entropy(T, species, P)
    return h - T * s


# =============================================================================
# Real Gas Corrections
# =============================================================================

@jit(nopython=True, cache=True)
def calculate_compressibility_factor_virial(
    T: float,
    P: float,
    Tc: float,
    Pc: float,
    omega: float = 0.0
) -> float:
    """
    Calculate compressibility factor Z using truncated virial equation.

    Z = PV/(nRT) = 1 + B*P/(RT) + C*(P/(RT))² + ...

    Uses Pitzer correlation for second virial coefficient B.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)
        Tc: Critical temperature (K)
        Pc: Critical pressure (Pa)
        omega: Acentric factor (dimensionless, 0 for simple molecules)

    Returns:
        Compressibility factor Z (dimensionless)
        Z = 1.0 for ideal gas
        Z < 1.0 indicates attractive forces dominate
        Z > 1.0 indicates repulsive forces dominate

    Reference:
        Pitzer, K.S. (1955). "The Volumetric and Thermodynamic Properties
        of Fluids. I. Theoretical Basis and Virial Coefficients"
        J. Am. Chem. Soc. 77(13): 3427-3433.
    """
    if Tc <= 0 or Pc <= 0:
        return 1.0  # No critical data, assume ideal

    # Reduced temperature and pressure
    Tr = T / Tc
    Pr = P / Pc

    # Pitzer correlation for B*Pc/(R*Tc)
    # B0 = 0.083 - 0.422/Tr^1.6
    # B1 = 0.139 - 0.172/Tr^4.2
    # B*Pc/(R*Tc) = B0 + omega*B1

    if Tr > 0.1:
        B0 = 0.083 - 0.422 / (Tr ** 1.6)
        B1 = 0.139 - 0.172 / (Tr ** 4.2)
    else:
        # Very low reduced temperature - use limiting behavior
        B0 = -0.422 / (0.1 ** 1.6)
        B1 = -0.172 / (0.1 ** 4.2)

    B_reduced = B0 + omega * B1

    # Z = 1 + B*P/(RT) = 1 + B*Pc/(R*Tc) * Pr/Tr
    Z = 1.0 + B_reduced * Pr / Tr

    # Clamp to physical range (0.1 to 2.0)
    return max(0.1, min(2.0, Z))


@jit(nopython=True, cache=True)
def calculate_compressibility_factor_rk(
    T: float,
    P: float,
    Tc: float,
    Pc: float
) -> float:
    """
    Calculate compressibility factor using Redlich-Kwong equation of state.

    P = RT/(V-b) - a/(T^0.5 * V(V+b))

    More accurate than virial for higher pressures.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)
        Tc: Critical temperature (K)
        Pc: Critical pressure (Pa)

    Returns:
        Compressibility factor Z

    Reference:
        Redlich, O. & Kwong, J.N.S. (1949). "On the Thermodynamics of Solutions"
        Chem. Rev. 44(1): 233-244.
    """
    if Tc <= 0 or Pc <= 0 or T <= 0 or P <= 0:
        return 1.0

    R = 8.314462  # J/(mol·K)

    # RK parameters
    a = 0.42748 * R * R * (Tc ** 2.5) / Pc
    b = 0.08664 * R * Tc / Pc

    # Reduced form
    A = a * P / (R * R * T ** 2.5)
    B = b * P / (R * T)

    # Solve cubic: Z³ - Z² + (A - B - B²)Z - AB = 0
    # Using Newton-Raphson iteration starting from ideal gas

    Z = 1.0  # Initial guess (ideal gas)

    for _ in range(50):
        f = Z**3 - Z**2 + (A - B - B*B) * Z - A * B
        df = 3*Z**2 - 2*Z + (A - B - B*B)

        if abs(df) < 1e-30:
            break

        dZ = -f / df
        Z_new = Z + dZ

        # Clamp to physical range
        Z_new = max(0.1, min(3.0, Z_new))

        if abs(dZ) < 1e-8:
            break

        Z = Z_new

    return Z


# Critical properties for common combustion species
# Format: (Tc [K], Pc [Pa], omega)
CRITICAL_PROPERTIES = {
    'H2': (33.19, 1.313e6, -0.216),
    'O2': (154.58, 5.043e6, 0.022),
    'N2': (126.20, 3.398e6, 0.037),
    'H2O': (647.14, 22.064e6, 0.344),
    'CO2': (304.13, 7.375e6, 0.224),
    'CO': (132.86, 3.499e6, 0.045),
    'CH4': (190.56, 4.599e6, 0.011),
    'C2H6': (305.32, 4.872e6, 0.099),
    'C3H8': (369.83, 4.248e6, 0.152),
    'NH3': (405.40, 11.333e6, 0.257),
    'N2O': (309.60, 7.245e6, 0.160),
    'NO': (180.00, 6.480e6, 0.582),
    'NO2': (431.35, 10.132e6, 0.851),
    'OH': (400.00, 8.000e6, 0.100),  # Estimated
    'H': (33.00, 1.300e6, 0.000),    # Estimated (similar to H2)
    'O': (155.00, 5.000e6, 0.000),   # Estimated (similar to O2)
}


def get_mixture_compressibility(
    T: float,
    P: float,
    composition: dict[str, float],
    method: str = "virial"
) -> float:
    """
    Calculate compressibility factor for a gas mixture.

    Uses Kay's mixing rules for pseudo-critical properties.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)
        composition: Dictionary of species name to mole fraction
        method: "virial" or "rk" (Redlich-Kwong)

    Returns:
        Mixture compressibility factor Z

    Reference:
        Kay, W.B. (1936). "Density of Hydrocarbon Gases and Vapors
        at High Temperature and Pressure"
    """
    # Kay's mixing rules for pseudo-critical properties
    Tc_mix = 0.0
    Pc_mix = 0.0
    omega_mix = 0.0
    total_fraction = 0.0

    for species, mole_frac in composition.items():
        if species in CRITICAL_PROPERTIES:
            Tc, Pc, omega = CRITICAL_PROPERTIES[species]
            Tc_mix += mole_frac * Tc
            Pc_mix += mole_frac * Pc
            omega_mix += mole_frac * omega
            total_fraction += mole_frac

    # Normalize if not all species have critical data
    if total_fraction > 0 and total_fraction < 0.99:
        # Scale up pseudo-critical properties
        Tc_mix /= total_fraction
        Pc_mix /= total_fraction
        omega_mix /= total_fraction
    elif total_fraction < 0.01:
        # No critical data available, assume ideal gas
        return 1.0

    # Calculate Z using selected method
    if method == "rk":
        return calculate_compressibility_factor_rk(T, P, Tc_mix, Pc_mix)
    else:
        return calculate_compressibility_factor_virial(T, P, Tc_mix, Pc_mix, omega_mix)


@jit(nopython=True, cache=True)
def correct_density_for_real_gas(
    rho_ideal: float,
    Z: float
) -> float:
    """
    Correct ideal gas density for real gas effects.

    For ideal gas: ρ = PM/(RT)
    For real gas:  ρ = PM/(ZRT)

    Args:
        rho_ideal: Ideal gas density (kg/m³)
        Z: Compressibility factor

    Returns:
        Real gas density (kg/m³)
    """
    if Z <= 0:
        return rho_ideal
    return rho_ideal / Z


@jit(nopython=True, cache=True)
def correct_enthalpy_departure(
    T: float,
    P: float,
    Tc: float,
    Pc: float,
    omega: float = 0.0
) -> float:
    """
    Calculate enthalpy departure from ideal gas behavior.

    ΔH = H_real - H_ideal

    Uses generalized correlation based on Pitzer's acentric factor.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)
        Tc: Critical temperature (K)
        Pc: Critical pressure (Pa)
        omega: Acentric factor

    Returns:
        Enthalpy departure (J/mol)

    Reference:
        Lee, B.I. & Kesler, M.G. (1975). "A Generalized Thermodynamic
        Correlation Based on Three-Parameter Corresponding States"
    """
    if Tc <= 0 or Pc <= 0:
        return 0.0

    R = 8.314462  # J/(mol·K)

    Tr = T / Tc
    Pr = P / Pc

    if Tr < 0.1:
        Tr = 0.1

    # Simple correlation for enthalpy departure
    # (H - H_ig)/(R*Tc) = Pr/Tr * (0.083 - 1.097/Tr^1.6 + omega*(0.139 - 0.894/Tr^4.2))

    term0 = 0.083 - 1.097 / (Tr ** 1.6)
    term1 = 0.139 - 0.894 / (Tr ** 4.2)

    H_departure_reduced = Pr / Tr * (term0 + omega * term1)
    H_departure = H_departure_reduced * R * Tc

    return H_departure