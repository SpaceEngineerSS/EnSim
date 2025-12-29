"""Core physics engine - GUI independent."""

from .constants import GAS_CONSTANT, G0, NASA_R
from .types import (
    SpeciesData,
    SpeciesDatabase,
    Element,
    ELEMENTS,
    Reactant,
    SystemState,
    EquilibriumResult,
    CalculationError,
    ConvergenceError,
    SingularMatrixError,
)
from .thermodynamics import (
    cp_over_r,
    h_over_rt,
    s_over_r,
    get_thermo_properties,
)
from .chemistry import (
    CombustionProblem,
    parse_formula,
    build_stoichiometry_matrix,
)
from .propulsion import (
    NozzleConditions,
    PerformanceResult,
    calculate_performance,
    calculate_c_star,
    calculate_ideal_expansion_ratio,
)

# New modules (v2.0)
from .validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_all_inputs,
    validate_chamber_pressure,
    validate_of_ratio,
    validate_expansion_ratio,
)

from .instability import (
    InstabilityResult,
    AcousticMode,
    analyze_combustion_instability,
    quick_stability_check,
)

from .shifting_equilibrium import (
    ShiftingFlowResult,
    FlowStation,
    solve_shifting_equilibrium,
    compare_frozen_vs_shifting,
)

from .terrain import (
    TerrainAwarenessSystem,
    TerrainWarning,
    AircraftState,
    create_gpws_for_flight_sim,
)

__all__ = [
    # Constants
    "GAS_CONSTANT",
    "G0",
    "NASA_R",
    # Types
    "SpeciesData",
    "SpeciesDatabase",
    "Element",
    "ELEMENTS",
    "Reactant",
    "SystemState",
    "EquilibriumResult",
    "CalculationError",
    "ConvergenceError",
    "SingularMatrixError",
    # Thermodynamics
    "cp_over_r",
    "h_over_rt",
    "s_over_r",
    "get_thermo_properties",
    # Chemistry
    "CombustionProblem",
    "parse_formula",
    "build_stoichiometry_matrix",
    # Propulsion
    "NozzleConditions",
    "PerformanceResult",
    "calculate_performance",
    "calculate_c_star",
    "calculate_ideal_expansion_ratio",
    # Validation (new)
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_all_inputs",
    "validate_chamber_pressure",
    "validate_of_ratio",
    "validate_expansion_ratio",
    # Instability (new)
    "InstabilityResult",
    "AcousticMode",
    "analyze_combustion_instability",
    "quick_stability_check",
    # Shifting Equilibrium (new)
    "ShiftingFlowResult",
    "FlowStation",
    "solve_shifting_equilibrium",
    "compare_frozen_vs_shifting",
    # Terrain Awareness (new)
    "TerrainAwarenessSystem",
    "TerrainWarning",
    "AircraftState",
    "create_gpws_for_flight_sim",
]

