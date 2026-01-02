"""Core physics engine - GUI independent."""

from .chemistry import (
    CombustionProblem,
    build_stoichiometry_matrix,
    parse_formula,
)
from .constants import G0, GAS_CONSTANT, NASA_R
from .instability import (
    AcousticMode,
    InstabilityResult,
    analyze_combustion_instability,
    quick_stability_check,
)
from .propulsion import (
    NozzleConditions,
    PerformanceResult,
    calculate_c_star,
    calculate_ideal_expansion_ratio,
    calculate_performance,
)
from .shifting_equilibrium import (
    FlowStation,
    ShiftingFlowResult,
    compare_frozen_vs_shifting,
    solve_shifting_equilibrium,
)
from .terrain import (
    AircraftState,
    TerrainAwarenessSystem,
    TerrainWarning,
    create_gpws_for_flight_sim,
)
from .thermodynamics import (
    cp_over_r,
    get_thermo_properties,
    h_over_rt,
    s_over_r,
)
from .types import (
    ELEMENTS,
    CalculationError,
    ConvergenceError,
    Element,
    EquilibriumResult,
    Reactant,
    SingularMatrixError,
    SpeciesData,
    SpeciesDatabase,
    SystemState,
)

# New modules (v2.0)
from .validation import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_all_inputs,
    validate_chamber_pressure,
    validate_expansion_ratio,
    validate_of_ratio,
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
