"""
Data types for thermochemical calculations.

Designed to separate string/metadata handling from numeric computation,
allowing Numba-compiled functions to work with raw numpy arrays.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpeciesData:
    """
    Container for NASA 7-term polynomial thermodynamic data.

    The NASA polynomial format uses two sets of 7 coefficients each:
    - High temperature range (typically 1000-6000 K)
    - Low temperature range (typically 200-1000 K)

    Coefficients a1-a5 define Cp/R, H/RT, S/R polynomials.
    Coefficients a6, a7 are integration constants for H and S.

    Attributes:
        name: Species chemical formula (e.g., "H2O", "CO2")
        molecular_weight: Molecular weight in g/mol (kg/kmol)
        phase: Phase indicator ('G' for gas, 'L' for liquid, 'S' for solid)
        temp_ranges: Temperature validity ranges [(T_low, T_mid, T_high)]
        coeffs_high: High-T coefficients array [a1, a2, a3, a4, a5, a6, a7]
        coeffs_low: Low-T coefficients array [a1, a2, a3, a4, a5, a6, a7]
        h_formation_298: Heat of formation at 298.15 K in J/mol (optional)

    Example:
        >>> h2o = SpeciesData(
        ...     name="H2O",
        ...     molecular_weight=18.01528,
        ...     phase="G",
        ...     temp_ranges=[(200.0, 1000.0, 6000.0)],
        ...     coeffs_high=np.array([...]),
        ...     coeffs_low=np.array([...])
        ... )
    """

    name: str
    molecular_weight: float
    phase: str = "G"
    temp_ranges: list[tuple[float, float, float]] = field(default_factory=list)
    coeffs_high: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(7, dtype=np.float64)
    )
    coeffs_low: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(7, dtype=np.float64)
    )
    h_formation_298: float | None = None

    @property
    def t_low(self) -> float:
        """Lowest valid temperature (K)."""
        if self.temp_ranges:
            return self.temp_ranges[0][0]
        return 200.0

    @property
    def t_mid(self) -> float:
        """Temperature where coefficient sets switch (K)."""
        if self.temp_ranges:
            return self.temp_ranges[0][1]
        return 1000.0

    @property
    def t_high(self) -> float:
        """Highest valid temperature (K)."""
        if self.temp_ranges:
            return self.temp_ranges[0][2]
        return 6000.0

    def get_coeffs_for_temp(self, T: float) -> NDArray[np.float64]:
        """
        Return appropriate coefficient set for given temperature.

        Args:
            T: Temperature in Kelvin

        Returns:
            7-element coefficient array (high or low T set)

        Raises:
            ValueError: If temperature is outside valid range
        """
        if self.t_low > T or self.t_high < T:
            raise ValueError(
                f"Temperature {T} K is outside valid range "
                f"[{self.t_low}, {self.t_high}] K for species {self.name}"
            )

        if self.t_mid <= T:
            return self.coeffs_high
        return self.coeffs_low

    def __repr__(self) -> str:
        return (
            f"SpeciesData(name='{self.name}', MW={self.molecular_weight:.4f}, "
            f"T_range=[{self.t_low:.0f}-{self.t_high:.0f}K])"
        )


@dataclass
class ThermoResult:
    """
    Results of thermodynamic property calculations.

    All properties are per mole unless otherwise specified.
    """

    temperature: float  # K
    pressure: float  # Pa

    # Non-dimensional properties
    cp_over_r: float  # Cp/R (dimensionless)
    h_over_rt: float  # H/(R*T) (dimensionless)
    s_over_r: float  # S/R (dimensionless)

    # Dimensional properties (SI units)
    cp: float  # J/(mol·K)
    h: float  # J/mol
    s: float  # J/(mol·K)
    g: float  # Gibbs free energy J/mol

    @classmethod
    def from_dimensionless(
        cls,
        T: float,
        P: float,
        cp_r: float,
        h_rt: float,
        s_r: float,
        R: float = 8.31446261815324
    ) -> "ThermoResult":
        """Create ThermoResult from dimensionless properties."""
        cp = cp_r * R
        h = h_rt * R * T
        s = s_r * R
        g = h - T * s

        return cls(
            temperature=T,
            pressure=P,
            cp_over_r=cp_r,
            h_over_rt=h_rt,
            s_over_r=s_r,
            cp=cp,
            h=h,
            s=s,
            g=g
        )


# Type alias for species database
SpeciesDatabase = dict[str, SpeciesData]


# =============================================================================
# Chemistry / Equilibrium Data Types
# =============================================================================

@dataclass(frozen=True)
class Element:
    """
    Chemical element with atomic properties.

    Frozen dataclass for immutability and hashability.

    Attributes:
        symbol: Element symbol (e.g., 'H', 'O', 'C', 'N')
        atomic_weight: Atomic weight in g/mol (kg/kmol)
    """
    symbol: str
    atomic_weight: float

    def __repr__(self) -> str:
        return f"Element({self.symbol}, {self.atomic_weight:.4f})"


# Standard atomic weights (IUPAC 2021)
ELEMENTS: dict[str, Element] = {
    'H': Element('H', 1.00794),
    'He': Element('He', 4.002602),
    'C': Element('C', 12.0107),
    'N': Element('N', 14.0067),
    'O': Element('O', 15.9994),
    'F': Element('F', 18.9984032),
    'S': Element('S', 32.065),
    'Cl': Element('Cl', 35.453),
    'Ar': Element('Ar', 39.948),
}


@dataclass
class Reactant:
    """
    Input reactant (fuel or oxidizer) for combustion problem.

    Attributes:
        species_name: Name matching species in database (e.g., 'H2', 'O2')
        moles: Number of moles of this reactant
        temperature: Initial temperature in Kelvin
        enthalpy: Total enthalpy in J (calculated from moles × molar enthalpy)
    """
    species_name: str
    moles: float
    temperature: float = 298.15
    enthalpy: float = 0.0  # Will be calculated during processing

    def __repr__(self) -> str:
        return f"Reactant({self.species_name}, n={self.moles:.4f} mol, T={self.temperature:.1f} K)"


@dataclass
class SystemState:
    """
    Current state of the equilibrium iteration.

    Stores all variables needed for Newton-Raphson iteration.

    Attributes:
        temperature: Current temperature estimate (K)
        pressure: System pressure (Pa)
        n_species: Number of species in system
        n_elements: Number of elements tracked
        moles: Mole numbers for each species (n_j)
        ln_moles: Natural log of mole numbers (for solver stability)
        lambda_i: Lagrange multipliers for element constraints
        g_rt: Dimensionless Gibbs energy (G_j / RT) for each species
        converged: Whether solution has converged
        iterations: Number of iterations performed
        max_correction: Maximum |Δln(n)| in last iteration
    """
    temperature: float
    pressure: float
    n_species: int
    n_elements: int
    moles: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    ln_moles: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    lambda_i: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    g_rt: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    converged: bool = False
    iterations: int = 0
    max_correction: float = 1.0

    @classmethod
    def create_initial(
        cls,
        T_guess: float,
        P: float,
        n_species: int,
        n_elements: int,
        initial_moles: NDArray[np.float64] | None = None
    ) -> "SystemState":
        """
        Create initial system state for iteration.

        Args:
            T_guess: Initial temperature guess (K)
            P: Pressure (Pa)
            n_species: Number of species
            n_elements: Number of elements
            initial_moles: Initial mole estimates (optional)
        """
        if initial_moles is None:
            # Default: small positive values
            moles = np.full(n_species, 1e-10, dtype=np.float64)
        else:
            moles = initial_moles.copy()

        # Clamp to minimum value
        moles = np.maximum(moles, 1e-30)

        return cls(
            temperature=T_guess,
            pressure=P,
            n_species=n_species,
            n_elements=n_elements,
            moles=moles,
            ln_moles=np.log(moles),
            lambda_i=np.zeros(n_elements, dtype=np.float64),
            g_rt=np.zeros(n_species, dtype=np.float64),
            converged=False,
            iterations=0,
            max_correction=1.0
        )


@dataclass
class EquilibriumResult:
    """
    Final result of equilibrium calculation.

    Attributes:
        temperature: Adiabatic flame temperature (K)
        pressure: System pressure (Pa)
        species_names: List of species names in order
        mole_fractions: Mole fraction of each species
        moles: Moles of each species
        total_moles: Sum of all moles
        mean_molecular_weight: Mixture molecular weight (g/mol)
        enthalpy: Total enthalpy (J)
        entropy: Total entropy (J/K)
        gamma: Ratio of specific heats (Cp/Cv)
        converged: Whether solution converged
        iterations: Number of iterations used
    """
    temperature: float
    pressure: float
    species_names: list[str]
    mole_fractions: NDArray[np.float64]
    moles: NDArray[np.float64]
    total_moles: float
    mean_molecular_weight: float
    enthalpy: float
    entropy: float
    gamma: float = 1.2  # Will be calculated
    converged: bool = True
    iterations: int = 0

    def get_mole_fraction(self, species_name: str) -> float:
        """Get mole fraction for a specific species."""
        try:
            idx = self.species_names.index(species_name)
            return float(self.mole_fractions[idx])
        except ValueError:
            return 0.0

    def __repr__(self) -> str:
        top_species = sorted(
            zip(self.species_names, self.mole_fractions, strict=False),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        species_str = ", ".join(f"{n}:{x:.4f}" for n, x in top_species)
        return (
            f"EquilibriumResult(T={self.temperature:.1f}K, "
            f"P={self.pressure/1e5:.2f}bar, [{species_str}])"
        )


class CalculationError(Exception):
    """Exception raised when equilibrium calculation fails."""
    pass


class ConvergenceError(CalculationError):
    """Exception raised when solver fails to converge."""
    pass


class SingularMatrixError(CalculationError):
    """Exception raised when stoichiometry matrix is singular."""
    pass

