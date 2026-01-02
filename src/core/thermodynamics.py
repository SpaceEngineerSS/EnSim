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
