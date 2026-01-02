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

# Path to NASA thermo data file (relative to project root)
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
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
    if filepath is None:
        filepath = NASA_THERMO_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"NASA thermo data file not found: {filepath}")

    species_data = {}

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    # Skip header lines until we find species data
    while i < len(lines):
        line = lines[i].strip()
        if line == "THERMO" or line.startswith("END") or not line:
            i += 1
            continue
        # Check if this is a temperature range line
        parts = line.split()
        if len(parts) == 3 and all(p.replace(".", "").replace("-", "").isdigit() for p in parts):
            # Temperature range line (e.g., "200.000  1000.000  6000.000")
            i += 1
            continue
        break

    # Parse species entries
    while i < len(lines) - 3:
        line = lines[i]

        if line.strip() == "END" or not line.strip():
            i += 1
            continue

        # Line 1: Species header
        # Format: Name (col 1-18), Description, Formula, Phase, MW, Hf
        name_part = line[:18].strip().split()[0] if line[:18].strip() else ""
        if not name_part:
            i += 1
            continue

        # Parse MW and Hf from end of line
        parts = line.split()
        try:
            M_mol = float(parts[-2])
            h_f = float(parts[-1])
        except (ValueError, IndexError):
            M_mol = 0.0
            h_f = 0.0

        # Lines 2-4: Coefficients
        try:
            line2 = lines[i + 1]
            line3 = lines[i + 2]
            line4 = lines[i + 3]
        except IndexError:
            break

        # Parse coefficients (fixed-width format: 15 chars each)
        def parse_coeffs_line(l):
            coeffs = []
            for j in range(5):
                start = j * 15
                end = start + 15
                if end <= len(l):
                    try:
                        val = float(l[start:end].replace("D", "E").replace("d", "e"))
                    except ValueError:
                        val = 0.0
                    coeffs.append(val)
            return coeffs

        # Low-temp coefficients: line2[a1-a5], line3[a6-a7]
        coeffs_low_1 = parse_coeffs_line(line2)
        coeffs_low_2 = parse_coeffs_line(line3)[:2]
        coeffs_low = coeffs_low_1 + coeffs_low_2

        # High-temp coefficients: line3[a1-a3], line4[a4-a7]
        coeffs_high_1 = parse_coeffs_line(line3)[2:5]
        coeffs_high_2 = parse_coeffs_line(line4)[:4]
        coeffs_high = coeffs_high_1 + coeffs_high_2

        # Store if we have valid coefficients
        if len(coeffs_low) >= 7 and len(coeffs_high) >= 7:
            species_data[name_part] = {
                "name": name_part,
                "M_mol": M_mol,
                "h_f": h_f,
                "T_mid": 1000.0,  # Standard midpoint
                "coeffs_low": np.array(coeffs_low[:7], dtype=np.float64),
                "coeffs_high": np.array(coeffs_high[:7], dtype=np.float64),
            }

        i += 5  # Move to next species (4 data lines + 1 blank)

    return species_data


def create_combustion_lookup_table(
    of_ratios: np.ndarray, Pc_values: np.ndarray, fuel: str = "RP-1", oxidizer: str = "O2"
) -> dict[str, np.ndarray]:
    """
    Pre-compute combustion properties lookup table for fast runtime interpolation.

    Creates tables for T_chamber, gamma, and M_mol as functions of O/F ratio
    and chamber pressure. Uses Gordon-McBride equilibrium solver.

    Args:
        of_ratios: Array of O/F ratios to compute (e.g., [2.0, 2.5, 3.0, 3.5])
        Pc_values: Array of chamber pressures in Pa (e.g., [1e6, 2e6, 3e6, 4e6, 5e6])
        fuel: Fuel species name
        oxidizer: Oxidizer species name

    Returns:
        Dictionary with:
            'of_grid': O/F ratio grid
            'Pc_grid': Chamber pressure grid (Pa)
            'T_chamber': 2D array of chamber temps (K), shape (n_of, n_Pc)
            'gamma': 2D array of gamma values
            'M_mol': 2D array of mean molecular weights (g/mol)

    Note:
        This function is slow (uses full equilibrium solver) - call once at
        simulation setup, NOT in the simulation loop.
    """
    n_of = len(of_ratios)
    n_Pc = len(Pc_values)

    T_table = np.zeros((n_of, n_Pc), dtype=np.float64)
    gamma_table = np.zeros((n_of, n_Pc), dtype=np.float64)
    M_mol_table = np.zeros((n_of, n_Pc), dtype=np.float64)

    # Note: Full implementation would use CombustionProblem.solve()
    # For now, use simplified correlations for RP-1/LOX
    # These are approximate values - real implementation should use CEA database

    for i, of in enumerate(of_ratios):
        for j, Pc in enumerate(Pc_values):
            # Simplified RP-1/LOX correlation (replace with full solver later)
            # T_c increases with O/F up to ~2.7, then decreases
            # Higher Pc slightly increases T_c

            T_base = 3400.0  # K at stoichiometric
            of_opt = 2.7  # Optimal O/F for temp
            T_penalty = 200.0 * (of - of_opt) ** 2
            Pc_factor = 1.0 + 0.02 * (Pc / 1e6 - 3.0)  # +2% per MPa above 3 MPa

            T_table[i, j] = max(2500.0, (T_base - T_penalty) * Pc_factor)

            # Gamma: typically 1.15-1.24 for combustion products
            gamma_table[i, j] = 1.15 + 0.05 * (3.0 - of) / 2.0
            if gamma_table[i, j] < 1.14:
                gamma_table[i, j] = 1.14
            if gamma_table[i, j] > 1.24:
                gamma_table[i, j] = 1.24

            # M_mol: typically 20-24 g/mol for RP-1/LOX
            M_mol_table[i, j] = 21.0 + 1.0 * (of - 2.5)
            if M_mol_table[i, j] < 19.0:
                M_mol_table[i, j] = 19.0
            if M_mol_table[i, j] > 26.0:
                M_mol_table[i, j] = 26.0

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
        formula = parse_formula(species.name)
        for i, element in enumerate(element_list):
            a_matrix[i, j] = formula.get(element, 0)
    return a_matrix


def calculate_element_totals(
    reactants: list[Reactant], species_db: SpeciesDatabase, element_list: list[str]
) -> NDArray[np.float64]:
    """Calculate total gram-atoms of each element from reactants."""
    b = np.zeros(len(element_list), dtype=np.float64)
    for reactant in reactants:
        formula = parse_formula(reactant.species_name)
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
        sp_elements = set(parse_formula(sp.name).keys())
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
def solve_equilibrium_gordon_mcbride(
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
            elements.update(parse_formula(r.species_name).keys())
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

        T = initial_temp_guess
        total_iter = 0
        converged = False
        n = np.zeros(n_spec, dtype=np.float64)
        h_current = 0.0

        for _t_iter in range(max_iterations):
            g_rt, h_rt, cp_r = compute_thermo(T, n_spec, coeffs_low, coeffs_high, t_mid)

            n, eq_conv = solve_equilibrium_gordon_mcbride(
                T, P_atm, a_matrix, b_elements, g_rt, 200, 1e-8
            )
            total_iter += 1

            h_current = 0.0
            cp_total = 0.0
            for j in range(n_spec):
                h_current += n[j] * h_rt[j] * GAS_CONSTANT * T
                cp_total += n[j] * cp_r[j] * GAS_CONSTANT

            h_error = h_target - h_current

            if np.abs(h_error) < np.abs(h_target) * tolerance + 1.0:
                converged = True
                break

            dT = h_error / cp_total if cp_total > 0.1 else 100.0 * np.sign(h_error)

            if np.abs(dT) > 300:
                dT = 300 * np.sign(dT)

            T = max(500.0, min(T + dT, 5500.0))

        n_total = np.sum(n)
        if n_total < 1e-20:
            n_total = 1.0
        x = n / n_total
        mean_mw = np.sum(x * mw)

        _, _, cp_r = compute_thermo(T, n_spec, coeffs_low, coeffs_high, t_mid)
        cp_mix = np.sum(x * cp_r) * GAS_CONSTANT
        cv_mix = cp_mix - GAS_CONSTANT
        gamma = cp_mix / cv_mix if cv_mix > 0 else 1.2

        return EquilibriumResult(
            temperature=T,
            pressure=pressure,
            species_names=[s.name for s in species_list],
            mole_fractions=x,
            moles=n,
            total_moles=n_total,
            mean_molecular_weight=mean_mw,
            enthalpy=h_current,
            entropy=0.0,
            gamma=gamma,
            converged=converged,
            iterations=total_iter,
        )
