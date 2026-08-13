"""Core physics engine - GUI independent."""

from .ballistics import BallisticTrajectory, propagate_dragless_wgs84
from .chemistry import (
    CombustionProblem,
    build_stoichiometry_matrix,
    parse_formula,
)
from .constants import G0, GAS_CONSTANT, NASA_R
from .cooling import (
    CoolantType,
    CoolingChannel,
    CoolingSystemDesign,
    CoolingType,
    ThermalAnalysisResult,
    analyze_cooling_system,
    bartz_heat_transfer_coefficient,
    coolant_heat_transfer_coefficient,
    smooth_channel_darcy_friction_factor,
)
from .engine_uq import (
    EngineUQAnalyzer,
    EngineUQInput,
    EngineUQResult,
    evaluate_ideal_engine_performance,
)
from .geodesy import (
    EARTH_GRAVITATIONAL_PARAMETER,
    EARTH_J2,
    EARTH_ROTATION_RATE,
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_AXIS,
    earth_fixed_acceleration,
    ecef_to_eci,
    ecef_to_eci_matrix,
    ecef_to_enu_matrix,
    ecef_to_geodetic,
    eci_to_ecef,
    eci_to_ecef_matrix,
    enu_to_ecef,
    geodetic_to_ecef,
    j2_gravity_ecef,
)
from .instability import (
    AcousticMode,
    InstabilityResult,
    analyze_combustion_instability,
    quick_stability_check,
)
from .optimization import (
    OptimizationResult,
    TrajectoryConstraints,
    optimize_engine_parameters,
    optimize_gravity_turn,
    optimize_nozzle_expansion_ratio,
    optimize_propellant_load,
    optimize_stage_mass_allocation,
)
from .propulsion import (
    NozzleConditions,
    PerformanceResult,
    calculate_c_star,
    calculate_ideal_expansion_ratio,
    calculate_performance,
)
from .staging import (
    MultiStageVehicle,
    Stage,
    StageEngine,
    StageStatus,
    StagingEvent,
    StagingTrigger,
    create_custom_vehicle,
    create_heavy_lift_reference,
    create_medium_lift_reference,
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
    "WGS84_SEMI_MAJOR_AXIS",
    "WGS84_FLATTENING",
    "EARTH_GRAVITATIONAL_PARAMETER",
    "EARTH_J2",
    "EARTH_ROTATION_RATE",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_enu_matrix",
    "enu_to_ecef",
    "ecef_to_eci",
    "eci_to_ecef",
    "ecef_to_eci_matrix",
    "eci_to_ecef_matrix",
    "j2_gravity_ecef",
    "earth_fixed_acceleration",
    # Types
    "BallisticTrajectory",
    "propagate_dragless_wgs84",
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
    # Validation
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_all_inputs",
    "validate_chamber_pressure",
    "validate_of_ratio",
    "validate_expansion_ratio",
    # Instability
    "InstabilityResult",
    "AcousticMode",
    "analyze_combustion_instability",
    "quick_stability_check",
    # Multi-stage
    "MultiStageVehicle",
    "Stage",
    "StageEngine",
    "StagingEvent",
    "StagingTrigger",
    "StageStatus",
    "create_medium_lift_reference",
    "create_heavy_lift_reference",
    "create_custom_vehicle",
    # Optimization
    "OptimizationResult",
    "TrajectoryConstraints",
    "optimize_gravity_turn",
    "optimize_nozzle_expansion_ratio",
    "optimize_stage_mass_allocation",
    "optimize_engine_parameters",
    "optimize_propellant_load",
    # Cooling
    "CoolingType",
    "CoolantType",
    "CoolingChannel",
    "CoolingSystemDesign",
    "ThermalAnalysisResult",
    "analyze_cooling_system",
    "bartz_heat_transfer_coefficient",
    "coolant_heat_transfer_coefficient",
    "smooth_channel_darcy_friction_factor",
    # Reduced-order engine uncertainty propagation
    "EngineUQInput",
    "EngineUQResult",
    "EngineUQAnalyzer",
    "evaluate_ideal_engine_performance",
]
