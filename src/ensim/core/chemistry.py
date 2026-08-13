"""
Chemical equilibrium solver using Gibbs free energy minimization.

Uses a robust two-phase approach:
1. Element potential method for major species
2. Equilibrium constant correction for trace species

References:
    - Gordon, S. & McBride, B.J. (1994). NASA RP-1311.
"""

import re
from pathlib import Path

import numpy as np
from numba import jit
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, brentq, least_squares, linprog, minimize

from .constants import GAS_CONSTANT
from .types import (
    CalculationError,
    EquilibriumResult,
    Reactant,
    SpeciesData,
    SpeciesDatabase,
)

# =============================================================================
# NASA 7-Term Polynomial Thermo Data Loader
# =============================================================================

# Thermodynamic data shipped with the package.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NASA_THERMO_FILE = _DATA_DIR / "nasa_thermo.dat"


@jit(nopython=True, cache=True)
def nasa_get_cp_r(T: float, coeffs_low: np.ndarray, coeffs_high: np.ndarray, T_mid: float) -> float:
    """
    Calculate Cp/R for a species at temperature T using NASA 7-term polynomial.

    Cp/R = a1 + a2*T + a3*T² + a4*T³ + a5*T⁴

    Args:
        T: Temperature (K)
        coeffs_low: Low-temperature coefficients [a1..a7] for T < T_mid
        coeffs_high: High-temperature coefficients [a1..a7] for T >= T_mid
        T_mid: Mid-point temperature (K), typically 1000K

    Returns:
        Cp/R (dimensionless)
    """
    c = coeffs_high if T_mid <= T else coeffs_low
    return c[0] + c[1] * T + c[2] * T * T + c[3] * T * T * T + c[4] * T * T * T * T


@jit(nopython=True, cache=True)
def nasa_get_h_rt(T: float, coeffs_low: np.ndarray, coeffs_high: np.ndarray, T_mid: float) -> float:
    """
    Calculate H/(R*T) for a species at temperature T using NASA 7-term polynomial.

    H/(RT) = a1 + a2/2*T + a3/3*T² + a4/4*T³ + a5/5*T⁴ + a6/T

    Args:
        T: Temperature (K)
        coeffs_low: Low-temperature coefficients [a1..a7] for T < T_mid
        coeffs_high: High-temperature coefficients [a1..a7] for T >= T_mid
        T_mid: Mid-point temperature (K)

    Returns:
        H/(RT) (dimensionless)
    """
    c = coeffs_high if T_mid <= T else coeffs_low
    return (
        c[0]
        + c[1] / 2.0 * T
        + c[2] / 3.0 * T * T
        + c[3] / 4.0 * T * T * T
        + c[4] / 5.0 * T * T * T * T
        + c[5] / T
    )


@jit(nopython=True, cache=True)
def nasa_get_s_r(T: float, coeffs_low: np.ndarray, coeffs_high: np.ndarray, T_mid: float) -> float:
    """
    Calculate S/R for a species at temperature T using NASA 7-term polynomial.

    S/R = a1*ln(T) + a2*T + a3/2*T² + a4/3*T³ + a5/4*T⁴ + a7

    Args:
        T: Temperature (K)
        coeffs_low: Low-temperature coefficients [a1..a7] for T < T_mid
        coeffs_high: High-temperature coefficients [a1..a7] for T >= T_mid
        T_mid: Mid-point temperature (K)

    Returns:
        S/R (dimensionless)
    """
    c = coeffs_high if T_mid <= T else coeffs_low
    return (
        c[0] * np.log(T)
        + c[1] * T
        + c[2] / 2.0 * T * T
        + c[3] / 3.0 * T * T * T
        + c[4] / 4.0 * T * T * T * T
        + c[6]
    )


@jit(nopython=True, cache=True)
def bilinear_interpolate(
    x: float, y: float, x_grid: np.ndarray, y_grid: np.ndarray, values: np.ndarray
) -> float:
    """
    Bilinear interpolation for 2D lookup tables with clamping.

    Used for interpolating combustion properties (Tc, gamma, M_mol)
    from O/F ratio and chamber pressure grids.

    Args:
        x: First coordinate (e.g., O/F ratio)
        y: Second coordinate (e.g., Pc in Pa)
        x_grid: 1D array of x grid points (sorted ascending)
        y_grid: 1D array of y grid points (sorted ascending)
        values: 2D array of values at grid points, shape (len(x_grid), len(y_grid))

    Returns:
        Interpolated value (clamped at boundaries, no extrapolation)
    """
    nx = len(x_grid)
    ny = len(y_grid)

    # Clamp x to grid bounds (no extrapolation)
    if x <= x_grid[0]:
        x = x_grid[0]
        ix = 0
    elif x >= x_grid[nx - 1]:
        x = x_grid[nx - 1]
        ix = nx - 2
    else:
        # Find interval [x_grid[ix], x_grid[ix+1]] containing x
        ix = 0
        for i in range(nx - 1):
            if x_grid[i] <= x < x_grid[i + 1]:
                ix = i
                break

    # Clamp y to grid bounds (no extrapolation)
    if y <= y_grid[0]:
        y = y_grid[0]
        iy = 0
    elif y >= y_grid[ny - 1]:
        y = y_grid[ny - 1]
        iy = ny - 2
    else:
        # Find interval [y_grid[iy], y_grid[iy+1]] containing y
        iy = 0
        for j in range(ny - 1):
            if y_grid[j] <= y < y_grid[j + 1]:
                iy = j
                break

    # Clamp indices to valid range
    if ix >= nx - 1:
        ix = nx - 2
    if iy >= ny - 1:
        iy = ny - 2
    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0

    # Bilinear interpolation
    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]

    dx = x1 - x0
    dy = y1 - y0

    tx = 0.0 if dx < 1e-20 else (x - x0) / dx

    ty = 0.0 if dy < 1e-20 else (y - y0) / dy

    # Get corner values
    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]

    # Bilinear formula
    v = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11

    return v


def load_nasa_thermo_dat(filepath: Path | None = None) -> dict[str, dict]:
    """
    Load NASA 7-term polynomial coefficients from .dat file.

    File format (per species):
        Line 1: Name, description, formula, phase, MW, Hf0
        Line 2: Low-temp coefficients a1-a5 (card 1)
        Line 3: Low-temp a6-a7, High-temp a1-a3 (card 2)
        Line 4: High-temp a4-a7 (card 3)
        Line 5: Blank (card 4)

    Args:
        filepath: Path to nasa_thermo.dat (default: data/nasa_thermo.dat)

    Returns:
        Dictionary mapping species name to coefficient data:
        {
            'H2': {
                'name': 'H2',
                'M_mol': 2.01588,  # g/mol
                'h_f': 0.0,       # J/mol formation enthalpy
                'T_mid': 1000.0,  # K
                'coeffs_low': np.array([a1..a7]),
                'coeffs_high': np.array([a1..a7])
            },
            ...
        }
    """
    from ensim.utils.nasa_parser import parse_nasa_file

    database = parse_nasa_file(filepath or NASA_THERMO_FILE)
    return {
        name: {
            "name": species.name,
            "M_mol": species.molecular_weight,
            "h_f": species.h_formation_298 or 0.0,
            "T_mid": species.t_mid,
            "coeffs_low": species.coeffs_low.copy(),
            "coeffs_high": species.coeffs_high.copy(),
        }
        for name, species in database.items()
    }


def create_combustion_lookup_table(
    of_ratios: np.ndarray,
    Pc_values: np.ndarray,
    fuel: str = "H2",
    oxidizer: str = "O2",
    species_db: SpeciesDatabase | None = None,
    use_equilibrium_solver: bool = True,
) -> dict[str, np.ndarray]:
    """
    Pre-compute combustion properties lookup table for fast runtime interpolation.

    Creates tables for T_chamber, gamma, and M_mol as functions of O/F ratio
    and chamber pressure using the full Gordon-McBride equilibrium solver.

    Args:
        of_ratios: Array of O/F ratios to compute (e.g., [2.0, 4.0, 6.0, 8.0])
        Pc_values: Array of chamber pressures in Pa (e.g., [1e6, 3e6, 5e6, 7e6])
        fuel: Fuel species name in the selected database
        oxidizer: Oxidizer species name in the selected database
        species_db: Species database (loads default if None)
        use_equilibrium_solver: Must be True; empirical fallback tables are rejected

    Returns:
        Dictionary with:
            'of_grid': O/F ratio grid
            'Pc_grid': Chamber pressure grid (Pa)
            'T_chamber': 2D array of chamber temps (K), shape (n_of, n_Pc)
            'gamma': 2D array of gamma values
            'M_mol': 2D array of mean molecular weights (g/mol)

    Note:
        This function is computationally expensive - call once at simulation
        setup, NOT in the simulation loop. Results are cached for interpolation.
    """
    n_of = len(of_ratios)
    n_Pc = len(Pc_values)

    T_table = np.zeros((n_of, n_Pc), dtype=np.float64)
    gamma_table = np.zeros((n_of, n_Pc), dtype=np.float64)
    M_mol_table = np.zeros((n_of, n_Pc), dtype=np.float64)

    if not use_equilibrium_solver:
        raise ValueError("empirical combustion lookup-table generation is not supported")
    if species_db is None:
        from ensim.utils.nasa_parser import load_default_database

        species_db = load_default_database()
    if fuel not in species_db or oxidizer not in species_db:
        raise KeyError("fuel and oxidizer must both exist in the thermodynamic database")
    if np.any(~np.isfinite(of_ratios)) or np.any(of_ratios <= 0.0):
        raise ValueError("all mass O/F ratios must be finite and positive")
    if np.any(~np.isfinite(Pc_values)) or np.any(Pc_values <= 0.0):
        raise ValueError("all chamber pressures must be finite and positive")

    fuel_mw = species_db[fuel].molecular_weight
    oxidizer_mw = species_db[oxidizer].molecular_weight
    for i, of_ratio in enumerate(of_ratios):
        for j, chamber_pressure in enumerate(Pc_values):
            problem = CombustionProblem(species_db)
            problem.add_fuel(fuel, moles=1.0)
            problem.add_oxidizer(oxidizer, moles=of_ratio * fuel_mw / oxidizer_mw)
            result = problem.solve(
                pressure=float(chamber_pressure),
                initial_temp_guess=3000.0,
                max_iterations=100,
                tolerance=1e-6,
            )
            if not result.converged:
                raise CalculationError(
                    f"equilibrium lookup point did not converge at O/F={of_ratio:g}, "
                    f"Pc={chamber_pressure:g} Pa"
                )
            T_table[i, j] = result.temperature
            gamma_table[i, j] = result.gamma
            M_mol_table[i, j] = result.mean_molecular_weight

    return {
        "of_grid": of_ratios.copy(),
        "Pc_grid": Pc_values.copy(),
        "T_chamber": T_table,
        "gamma": gamma_table,
        "M_mol": M_mol_table,
    }


@jit(nopython=True, cache=True)
def lookup_combustion_properties(
    of_ratio: float,
    Pc: float,
    of_grid: np.ndarray,
    Pc_grid: np.ndarray,
    T_table: np.ndarray,
    gamma_table: np.ndarray,
    M_mol_table: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fast runtime lookup of combustion properties using bilinear interpolation.

    Args:
        of_ratio: O/F ratio
        Pc: Chamber pressure (Pa)
        of_grid: O/F ratio grid from lookup table
        Pc_grid: Pressure grid from lookup table (Pa)
        T_table: Chamber temperature table (K)
        gamma_table: Gamma table
        M_mol_table: Molecular weight table (g/mol)

    Returns:
        Tuple of (T_chamber, gamma, M_mol)
    """
    T_c = bilinear_interpolate(of_ratio, Pc, of_grid, Pc_grid, T_table)
    gamma = bilinear_interpolate(of_ratio, Pc, of_grid, Pc_grid, gamma_table)
    M_mol = bilinear_interpolate(of_ratio, Pc, of_grid, Pc_grid, M_mol_table)

    return T_c, gamma, M_mol


# =============================================================================
# Stoichiometry Utilities
# =============================================================================


def parse_formula(formula: str) -> dict[str, int]:
    """Parse chemical formula into element counts."""
    formula = re.sub(r"\([GLSC]\)$", "", formula)
    elements: dict[str, int] = {}
    pattern = r"([A-Z][a-z]?)(\d*)"
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        if element:
            elements[element] = elements.get(element, 0) + count
    return elements


def build_stoichiometry_matrix(
    species_list: list[SpeciesData], element_list: list[str]
) -> NDArray[np.float64]:
    """Build stoichiometry matrix a[i,j] = atoms of element i in species j."""
    n_elements = len(element_list)
    n_species = len(species_list)
    a_matrix = np.zeros((n_elements, n_species), dtype=np.float64)
    for j, species in enumerate(species_list):
        formula = parse_formula(species.formula or species.name)
        for i, element in enumerate(element_list):
            a_matrix[i, j] = formula.get(element, 0)
    return a_matrix


def calculate_element_totals(
    reactants: list[Reactant], species_db: SpeciesDatabase, element_list: list[str]
) -> NDArray[np.float64]:
    """Calculate total gram-atoms of each element from reactants."""
    b = np.zeros(len(element_list), dtype=np.float64)
    for reactant in reactants:
        species = species_db.get(reactant.species_name)
        formula = parse_formula(
            species.formula if species and species.formula else reactant.species_name
        )
        for i, element in enumerate(element_list):
            if element in formula:
                b[i] += reactant.moles * formula[element]
    return b


def filter_valid_species(
    species_list: list[SpeciesData], element_list: list[str]
) -> list[SpeciesData]:
    """Filter species to only those that can be formed from available elements."""
    valid = []
    element_set = set(element_list)
    for sp in species_list:
        sp_elements = set(parse_formula(sp.formula or sp.name).keys())
        if sp_elements.issubset(element_set):
            valid.append(sp)
    return valid


# =============================================================================
# Thermodynamic Functions
# =============================================================================


@jit(nopython=True, cache=True)
def compute_thermo(
    T: float,
    n_spec: int,
    coeffs_low: NDArray[np.float64],
    coeffs_high: NDArray[np.float64],
    t_mid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute G/RT, H/RT, Cp/R for all species."""
    g_rt = np.zeros(n_spec, dtype=np.float64)
    h_rt = np.zeros(n_spec, dtype=np.float64)
    cp_r = np.zeros(n_spec, dtype=np.float64)

    for j in range(n_spec):
        c = coeffs_high[j] if t_mid[j] <= T else coeffs_low[j]

        h = c[0] + c[1] / 2 * T + c[2] / 3 * T**2 + c[3] / 4 * T**3 + c[4] / 5 * T**4 + c[5] / T
        s = c[0] * np.log(T) + c[1] * T + c[2] / 2 * T**2 + c[3] / 3 * T**3 + c[4] / 4 * T**4 + c[6]

        g_rt[j] = h - s
        h_rt[j] = h
        cp_r[j] = c[0] + c[1] * T + c[2] * T**2 + c[3] * T**3 + c[4] * T**4

    return g_rt, h_rt, cp_r


@jit(nopython=True, cache=True)
def _solve_equilibrium_element_potential_legacy(
    T: float,
    P_atm: float,
    a_ij: NDArray[np.float64],  # (n_elem, n_spec)
    b_i: NDArray[np.float64],  # (n_elem,)
    g_rt: NDArray[np.float64],  # (n_spec,) - standard Gibbs g°/RT
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], bool]:
    """
    Solve equilibrium using Gordon-McBride iteration (NASA RP-1311 method).

    The method iterates on:
    1. Element potentials (π_i) to satisfy element balance
    2. Mole numbers derived from equilibrium condition

    At equilibrium for ideal gas:
        g°_j/RT + ln(n_j) + ln(P/n_tot) = Σ a_ij * π_i

    Where π_i are dimensionless element potentials (λ_i / RT).
    """
    n_elem = a_ij.shape[0]
    n_spec = a_ij.shape[1]

    # Initialize mole numbers with stoichiometric estimate
    n = np.zeros(n_spec, dtype=np.float64)
    n_tot = 0.0
    for i in range(n_elem):
        n_tot += b_i[i]
    n_tot /= 2.0
    if n_tot < 0.01:
        n_tot = 1.0

    # Initial guess: distribute moles based on element requirements
    for j in range(n_spec):
        atom_count = 0.0
        for i in range(n_elem):
            atom_count += a_ij[i, j]
        if atom_count > 0:
            n[j] = n_tot / (n_spec * atom_count)
        else:
            n[j] = 0.01
        if n[j] < 1e-20:
            n[j] = 1e-20

    # Initialize element potentials (π = λ/RT)
    pi = np.zeros(n_elem, dtype=np.float64)

    # Set initial pi based on average g_rt values for species containing each element
    for i in range(n_elem):
        sum_g = 0.0
        count = 0
        for j in range(n_spec):
            if a_ij[i, j] > 0:
                sum_g += g_rt[j] / a_ij[i, j]
                count += 1
        if count > 0:
            pi[i] = sum_g / count

    converged = False
    ln_P = np.log(P_atm)

    for _iteration in range(max_iter):
        # Current total moles
        n_tot = 0.0
        for j in range(n_spec):
            n_tot += n[j]
        if n_tot < 1e-20:
            n_tot = 1.0
        ln_n_tot = np.log(n_tot)

        # Compute correction using Newton-Raphson on the Lagrangian system
        # Variables: Δln(n_j) for each species, Δπ_i for each element

        # First, compute reduced potentials for each species
        # μ_j / RT = g°_j/RT + ln(n_j) + ln(P/n_tot)
        mu_rt = np.zeros(n_spec, dtype=np.float64)
        for j in range(n_spec):
            if n[j] > 1e-30:
                mu_rt[j] = g_rt[j] + np.log(n[j]) + ln_P - ln_n_tot
            else:
                mu_rt[j] = g_rt[j] - 70.0 + ln_P - ln_n_tot

        # Equilibrium residual for each species:
        # r_j = μ_j/RT - Σ a_ij * π_i
        # At equilibrium, r_j = 0

        # Element balance residual:
        # R_i = b_i - Σ a_ij * n_j
        R = np.zeros(n_elem, dtype=np.float64)
        for i in range(n_elem):
            R[i] = b_i[i]
            for j in range(n_spec):
                R[i] -= a_ij[i, j] * n[j]

        # Build the iteration matrix
        # We solve for Δπ_i using:
        # Σ_k A_ik * Δπ_k = R_i + Σ_j a_ij * n_j * r_j
        # where A_ik = Σ_j a_ij * a_kj * n_j

        # Compute species residuals
        r = np.zeros(n_spec, dtype=np.float64)
        for j in range(n_spec):
            r[j] = mu_rt[j]
            for i in range(n_elem):
                r[j] -= a_ij[i, j] * pi[i]

        # Modified RHS
        rhs = np.zeros(n_elem, dtype=np.float64)
        for i in range(n_elem):
            rhs[i] = R[i]
            for j in range(n_spec):
                rhs[i] += a_ij[i, j] * n[j] * r[j]

        # Build matrix A
        A = np.zeros((n_elem, n_elem), dtype=np.float64)
        for i in range(n_elem):
            for k in range(n_elem):
                for j in range(n_spec):
                    A[i, k] += a_ij[i, j] * a_ij[k, j] * n[j]
            A[i, i] += 1e-12  # Regularization

        # Solve for Δπ
        delta_pi = np.zeros(n_elem, dtype=np.float64)

        if n_elem == 1:
            if np.abs(A[0, 0]) > 1e-30:
                delta_pi[0] = rhs[0] / A[0, 0]
        elif n_elem == 2:
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if np.abs(det) > 1e-30:
                delta_pi[0] = (A[1, 1] * rhs[0] - A[0, 1] * rhs[1]) / det
                delta_pi[1] = (A[0, 0] * rhs[1] - A[1, 0] * rhs[0]) / det
        else:
            # Gaussian elimination
            AA = A.copy()
            bb = rhs.copy()
            for i in range(n_elem):
                max_idx = i
                for k in range(i + 1, n_elem):
                    if np.abs(AA[k, i]) > np.abs(AA[max_idx, i]):
                        max_idx = k
                for jj in range(n_elem):
                    AA[i, jj], AA[max_idx, jj] = AA[max_idx, jj], AA[i, jj]
                bb[i], bb[max_idx] = bb[max_idx], bb[i]
                if np.abs(AA[i, i]) > 1e-30:
                    for k in range(i + 1, n_elem):
                        f = AA[k, i] / AA[i, i]
                        for jj in range(n_elem):
                            AA[k, jj] -= f * AA[i, jj]
                        bb[k] -= f * bb[i]
            for i in range(n_elem - 1, -1, -1):
                delta_pi[i] = bb[i]
                for jj in range(i + 1, n_elem):
                    delta_pi[i] -= AA[i, jj] * delta_pi[jj]
                if np.abs(AA[i, i]) > 1e-30:
                    delta_pi[i] /= AA[i, i]

        # Compute species corrections
        # Δln(n_j) = -r_j + Σ a_ij * Δπ_i
        delta_ln_n = np.zeros(n_spec, dtype=np.float64)
        max_delta = 0.0
        for j in range(n_spec):
            delta_ln_n[j] = -r[j]
            for i in range(n_elem):
                delta_ln_n[j] += a_ij[i, j] * delta_pi[i]
            if np.abs(delta_ln_n[j]) > max_delta:
                max_delta = np.abs(delta_ln_n[j])

        # Damping
        damp = 1.0
        if max_delta > 2.0:
            damp = 2.0 / max_delta

        # Update π
        for i in range(n_elem):
            pi[i] += damp * delta_pi[i]

        # Update n
        for j in range(n_spec):
            ln_n_new = np.log(max(n[j], 1e-30)) + damp * delta_ln_n[j]
            if ln_n_new > 10:
                ln_n_new = 10
            elif ln_n_new < -70:
                ln_n_new = -70
            n[j] = np.exp(ln_n_new)

        # Check convergence (element balance)
        max_res = 0.0
        for i in range(n_elem):
            rel = np.abs(R[i]) / max(b_i[i], 1e-10)
            if rel > max_res:
                max_res = rel

        if max_res < tol and max_delta < tol:
            converged = True
            break

    return n, converged


def solve_equilibrium_gordon_mcbride(
    T: float,
    P_atm: float,
    a_ij: NDArray[np.float64],
    b_i: NDArray[np.float64],
    g_rt: NDArray[np.float64],
    max_iter: int,
    tol: float,
    initial_moles: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], bool]:
    """Minimize ideal-gas Gibbs energy subject to elemental conservation.

    The objective and constraints follow NASA RP-1311. The optimization is
    scaled by the total elemental inventory so convergence does not depend on
    the arbitrary mole basis chosen by the caller.
    """
    if T <= 0.0 or P_atm <= 0.0:
        raise ValueError("Temperature and pressure must be positive")
    if a_ij.ndim != 2 or b_i.ndim != 1 or g_rt.ndim != 1:
        raise ValueError("Invalid equilibrium array dimensions")
    if a_ij.shape != (b_i.size, g_rt.size):
        raise ValueError("Stoichiometry and thermodynamic arrays are inconsistent")
    if np.any(b_i < 0.0) or not np.all(np.isfinite(g_rt)):
        raise ValueError("Invalid elemental inventory or Gibbs energies")

    scale = max(float(np.sum(b_i)), 1.0)
    b_scaled = b_i / scale
    n_species = g_rt.size
    feasibility = linprog(
        np.ones(n_species),
        A_eq=a_ij,
        b_eq=b_scaled,
        bounds=[(0.0, None)] * n_species,
        method="highs",
    )
    if not feasibility.success:
        return np.zeros(n_species, dtype=np.float64), False

    pressure_log = float(np.log(P_atm))

    def objective(x: NDArray[np.float64]) -> float:
        total = max(float(np.sum(x)), 1e-300)
        positive = x > 0.0
        terms = np.zeros_like(x)
        terms[positive] = x[positive] * (
            g_rt[positive] + np.log(x[positive] / total) + pressure_log
        )
        return float(np.sum(terms))

    def gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
        total = max(float(np.sum(x)), 1e-300)
        safe = np.maximum(x, 1e-300)
        return g_rt + np.log(safe / total) + pressure_log

    starting_points = [feasibility.x]
    if initial_moles is not None:
        candidate = np.asarray(initial_moles, dtype=np.float64) / scale
        if (
            candidate.shape == (n_species,)
            and np.all(candidate >= 0.0)
            and np.max(np.abs(a_ij @ candidate - b_scaled)) <= 1e-7
        ):
            starting_points.insert(0, candidate)

    result = None
    bounds = Bounds(np.zeros(n_species), np.full(n_species, np.inf))
    constraint = LinearConstraint(a_ij, b_scaled, b_scaled)
    for starting_point in starting_points:
        attempt = minimize(
            objective,
            starting_point,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={
                "ftol": min(tol, 1e-10),
                "maxiter": max(max_iter, 200),
                "disp": False,
            },
        )
        if result is None or attempt.fun < result.fun:
            result = attempt
        if attempt.success:
            result = attempt
            break

    assert result is not None

    x_seed = np.maximum(np.asarray(result.x, dtype=np.float64), 1e-30)
    y_seed = np.log(x_seed)
    mole_fractions = x_seed / np.sum(x_seed)
    chemical_potential = g_rt + np.log(mole_fractions) + pressure_log
    lambda_seed = np.linalg.lstsq(a_ij.T, chemical_potential, rcond=None)[0]
    z_seed = np.concatenate((y_seed, lambda_seed))
    balance_weight = 100.0

    def optimality_residual(z: NDArray[np.float64]) -> NDArray[np.float64]:
        amounts = np.exp(z[:n_species])
        total = float(np.sum(amounts))
        potentials = g_rt + np.log(amounts / total) + pressure_log
        stationarity = potentials - a_ij.T @ z[n_species:]
        balance = balance_weight * (a_ij @ amounts - b_scaled)
        return np.concatenate((stationarity, balance))

    def optimality_jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
        amounts = np.exp(z[:n_species])
        fractions = amounts / np.sum(amounts)
        jacobian = np.zeros((n_species + b_i.size, n_species + b_i.size))
        jacobian[:n_species, :n_species] = np.eye(n_species) - fractions
        jacobian[:n_species, n_species:] = -a_ij.T
        jacobian[n_species:, :n_species] = balance_weight * a_ij * amounts
        return jacobian

    refined = least_squares(
        optimality_residual,
        z_seed,
        jac=optimality_jacobian,
        bounds=(
            np.concatenate((np.full(n_species, -80.0), np.full(b_i.size, -1e3))),
            np.concatenate((np.full(n_species, 20.0), np.full(b_i.size, 1e3))),
        ),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=max(max_iter * 3, 300),
    )
    x = np.exp(refined.x[:n_species])
    n = x * scale
    balance_scale = np.maximum(np.abs(b_i), 1.0)
    balance_error = float(np.max(np.abs(a_ij @ n - b_i) / balance_scale))
    stationarity_error = float(np.max(np.abs(optimality_residual(refined.x)[:n_species])))
    converged = bool(
        refined.success and balance_error <= max(tol, 1e-8) and stationarity_error <= 1e-6
    )
    return n, converged


# =============================================================================
# High-Level Interface
# =============================================================================


class CombustionProblem:
    """Combustion equilibrium problem solver."""

    def __init__(self, species_db: SpeciesDatabase):
        self.species_db = species_db
        self.reactants: list[Reactant] = []
        self.product_species: list[str] = []
        # Extended default products for comprehensive combustion equilibrium
        # Includes nitrogen oxides, radicals, and minor species for accuracy
        self.default_products = [
            # Major H/O species
            "H2O",
            "H2",
            "O2",
            "OH",
            "H",
            "O",
            "HO2",
            "H2O2",
            # Carbon species
            "CO2",
            "CO",
            "CH4",
            "C2H2",
            "C2H4",
            "CH2O",
            "CHO",
            # Nitrogen species (for RP-1, UDMH, etc.)
            "N2",
            "NO",
            "NO2",
            "N",
            "N2O",
            "NH3",
            "HCN",
            "CN",
            # Additional radicals for high-temperature accuracy
            "HNO",
            "NH",
            "NH2",
        ]

    def add_fuel(self, species_name: str, moles: float = 1.0, temperature: float = 298.15) -> None:
        self.reactants.append(Reactant(species_name, moles, temperature))

    def add_oxidizer(
        self, species_name: str, moles: float = 1.0, temperature: float = 298.15
    ) -> None:
        self.reactants.append(Reactant(species_name, moles, temperature))

    def set_products(self, species_names: list[str]) -> None:
        self.product_species = species_names

    def _get_elements(self) -> list[str]:
        elements: set = set()
        for r in self.reactants:
            species = self.species_db.get(r.species_name)
            formula = species.formula if species and species.formula else r.species_name
            elements.update(parse_formula(formula).keys())
        return sorted(elements)

    def _get_product_species(self, element_list: list[str]) -> list[SpeciesData]:
        names = self.product_species if self.product_species else self.default_products
        all_sp = [self.species_db[n] for n in names if n in self.species_db]
        return filter_valid_species(all_sp, element_list)

    def calculate_input_enthalpy(self) -> float:
        """Calculate total input enthalpy in J."""
        h_total = 0.0
        for r in self.reactants:
            if r.species_name in self.species_db:
                sp = self.species_db[r.species_name]
                T = r.temperature
                c = sp.coeffs_low if sp.t_mid > T else sp.coeffs_high
                h_rt = (
                    c[0]
                    + c[1] / 2 * T
                    + c[2] / 3 * T**2
                    + c[3] / 4 * T**3
                    + c[4] / 5 * T**4
                    + c[5] / T
                )
                h_total += r.moles * h_rt * GAS_CONSTANT * T
        return h_total

    def solve(
        self,
        pressure: float = 101325.0,
        initial_temp_guess: float = 3000.0,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
    ) -> EquilibriumResult:
        """Solve for equilibrium."""
        if not self.reactants:
            raise CalculationError("No reactants specified")

        element_list = self._get_elements()
        species_list = self._get_product_species(element_list)
        n_spec = len(species_list)

        if n_spec == 0:
            raise CalculationError("No valid product species")

        a_matrix = build_stoichiometry_matrix(species_list, element_list)
        b_elements = calculate_element_totals(self.reactants, self.species_db, element_list)

        coeffs_low = np.zeros((n_spec, 7), dtype=np.float64)
        coeffs_high = np.zeros((n_spec, 7), dtype=np.float64)
        t_mid = np.zeros(n_spec, dtype=np.float64)
        mw = np.zeros(n_spec, dtype=np.float64)

        for j, sp in enumerate(species_list):
            coeffs_low[j] = sp.coeffs_low
            coeffs_high[j] = sp.coeffs_high
            t_mid[j] = sp.t_mid
            mw[j] = sp.molecular_weight

        P_atm = pressure / 101325.0
        h_target = self.calculate_input_enthalpy()

        if pressure <= 0.0:
            raise ValueError("Pressure must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")

        lower_temperature = max(200.0, max(sp.t_low for sp in species_list))
        upper_temperature = min(6000.0, min(sp.t_high for sp in species_list))
        if lower_temperature >= upper_temperature:
            raise CalculationError("Product species do not share a valid temperature range")

        Evaluation = tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
            float,
        ]
        evaluations: dict[float, Evaluation] = {}

        def evaluate(temperature: float) -> Evaluation:
            key = float(temperature)
            if key in evaluations:
                return evaluations[key]
            g_rt, h_rt, cp_r = compute_thermo(key, n_spec, coeffs_low, coeffs_high, t_mid)
            initial_moles = None
            if evaluations:
                nearest_temperature = min(evaluations, key=lambda known: abs(known - key))
                initial_moles = evaluations[nearest_temperature][0]
            amounts, equilibrium_converged = solve_equilibrium_gordon_mcbride(
                key,
                P_atm,
                a_matrix,
                b_elements,
                g_rt,
                200,
                1e-8,
                initial_moles,
            )
            if not equilibrium_converged:
                raise CalculationError(f"Gibbs minimization did not converge at {key:.3f} K")
            enthalpy = float(np.sum(amounts * h_rt) * GAS_CONSTANT * key)
            residual = enthalpy - h_target
            value = amounts, g_rt, h_rt, cp_r, enthalpy, residual
            evaluations[key] = value
            return value

        sample_temperatures = np.linspace(lower_temperature, upper_temperature, 17)
        sample_temperatures = np.unique(
            np.append(
                sample_temperatures,
                np.clip(initial_temp_guess, lower_temperature, upper_temperature),
            )
        )
        samples: list[tuple[float, float]] = []
        for sample_temperature in sample_temperatures:
            try:
                residual = evaluate(float(sample_temperature))[5]
            except CalculationError:
                continue
            samples.append((float(sample_temperature), residual))

        exact = next((temperature for temperature, residual in samples if residual == 0.0), None)
        brackets = [
            (left[0], right[0])
            for left, right in zip(samples, samples[1:], strict=False)
            if np.signbit(left[1]) != np.signbit(right[1])
        ]
        if exact is not None:
            T = exact
        elif brackets:
            bracket = min(
                brackets,
                key=lambda bounds: abs(0.5 * (bounds[0] + bounds[1]) - initial_temp_guess),
            )
            T = float(
                brentq(
                    lambda temperature: evaluate(float(temperature))[5],
                    bracket[0],
                    bracket[1],
                    xtol=max(tolerance, 1e-6),
                    rtol=1e-12,
                    maxiter=max_iterations,
                )
            )
        else:
            raise CalculationError(
                "Adiabatic enthalpy balance has no root in the common "
                f"thermodynamic range {lower_temperature:.1f}-{upper_temperature:.1f} K"
            )

        n, g_rt, h_rt, cp_r, h_current, residual = evaluate(T)
        balance_scale = np.maximum(np.abs(b_elements), 1.0)
        element_balance_error = float(np.max(np.abs(a_matrix @ n - b_elements) / balance_scale))
        enthalpy_tolerance = abs(h_target) * tolerance + 1.0
        converged = bool(
            element_balance_error <= max(tolerance, 1e-8) and abs(residual) <= enthalpy_tolerance
        )

        n_total = np.sum(n)
        if n_total < 1e-20:
            n_total = 1.0
        x = n / n_total
        mean_mw = np.sum(x * mw)

        cp_mix = np.sum(x * cp_r) * GAS_CONSTANT
        cv_mix = cp_mix - GAS_CONSTANT
        gamma = cp_mix / cv_mix if cv_mix > 0 else 1.2
        positive = x > 0.0
        entropy = GAS_CONSTANT * np.sum(
            n[positive] * (h_rt[positive] - g_rt[positive] - np.log(x[positive] * P_atm))
        )

        return EquilibriumResult(
            temperature=T,
            pressure=pressure,
            species_names=[s.name for s in species_list],
            mole_fractions=x,
            moles=n,
            total_moles=n_total,
            mean_molecular_weight=mean_mw,
            enthalpy=h_current,
            entropy=float(entropy),
            gamma=gamma,
            converged=converged,
            iterations=len(evaluations),
            element_balance_error=element_balance_error,
            enthalpy_error=float(residual),
        )
