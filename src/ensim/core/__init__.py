"""EnSim physics models. Pure Python and NumPy; nothing here imports Qt.

Names listed below can be imported from ``ensim.core`` directly. Each one is
loaded from its submodule on first use, so importing a single model does not
pull in every other one.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, ...]] = {
    "aero": ("AeroResult", "analyze_rocket", "calculate_total_cp"),
    "ballistics": ("BallisticTrajectory", "propagate_dragless_wgs84"),
    "chemistry": ("CombustionProblem", "build_stoichiometry_matrix", "parse_formula"),
    "constants": (
        "G0",
        "GAS_CONSTANT",
        "NASA_R",
        "P_ATM",
        "P_REF",
        "EARTH_GRAVITATIONAL_PARAMETER",
        "EARTH_J2",
        "EARTH_ROTATION_RATE",
        "WGS84_FLATTENING",
        "WGS84_SEMI_MAJOR_AXIS",
    ),
    "cooling": (
        "CoolantType",
        "CoolingChannel",
        "CoolingSystemDesign",
        "CoolingType",
        "ThermalAnalysisResult",
        "analyze_cooling_system",
        "bartz_heat_transfer_coefficient",
        "coolant_heat_transfer_coefficient",
        "smooth_channel_darcy_friction_factor",
    ),
    "engine": ("EngineInputs", "EngineResult", "analyze_engine"),
    "engine_uq": (
        "EngineUQAnalyzer",
        "EngineUQInput",
        "EngineUQResult",
        "evaluate_ideal_engine_performance",
    ),
    "flight_6dof": (
        "FlightResult6DOF",
        "PhysicsViolationError",
        "SimulationCancelled",
        "simulate_flight_6dof",
    ),
    "geodesy": (
        "earth_fixed_acceleration",
        "ecef_to_eci",
        "ecef_to_eci_matrix",
        "ecef_to_enu_matrix",
        "ecef_to_geodetic",
        "eci_to_ecef",
        "eci_to_ecef_matrix",
        "enu_to_ecef",
        "geodetic_to_ecef",
        "j2_gravity_ecef",
    ),
    "instability": (
        "AcousticMode",
        "InstabilityResult",
        "analyze_combustion_instability",
        "quick_stability_check",
    ),
    "mission": ("AtmosphereState", "MissionProfile", "analyze_altitude_sweep", "get_atmosphere"),
    "monte_carlo": (
        "DispersionConfig",
        "DispersionResult",
        "SimulationResult",
        "run_monte_carlo",
    ),
    "optimization": (
        "OptimizationResult",
        "TrajectoryConstraints",
        "optimize_engine_parameters",
        "optimize_gravity_turn",
        "optimize_nozzle_expansion_ratio",
        "optimize_propellant_load",
        "optimize_stage_mass_allocation",
    ),
    "project_manager": ("ProjectFileError", "ProjectManager"),
    "propulsion": (
        "NozzleConditions",
        "PerformanceResult",
        "calculate_c_star",
        "calculate_ideal_expansion_ratio",
        "calculate_performance",
        "mach_from_area_ratio",
    ),
    "recovery": ("Parachute", "estimate_descent", "size_parachute"),
    "rocket": (
        "BodyTube",
        "EngineMount",
        "Fin",
        "FinSet",
        "NoseCone",
        "NoseShape",
        "Rocket",
        "create_default_rocket",
    ),
    "staging": (
        "MultiStageVehicle",
        "Stage",
        "StageEngine",
        "StageStatus",
        "StagingEvent",
        "StagingTrigger",
        "create_custom_vehicle",
        "create_heavy_lift_reference",
        "create_medium_lift_reference",
    ),
    "thermodynamics": ("cp_over_r", "get_thermo_properties", "h_over_rt", "s_over_r"),
    "types": (
        "ELEMENTS",
        "CalculationError",
        "ConvergenceError",
        "Element",
        "EquilibriumResult",
        "Reactant",
        "SingularMatrixError",
        "SpeciesData",
        "SpeciesDatabase",
        "SystemState",
    ),
    "validation": (
        "ValidationIssue",
        "ValidationResult",
        "ValidationSeverity",
        "validate_all_inputs",
        "validate_chamber_pressure",
        "validate_expansion_ratio",
        "validate_of_ratio",
    ),
}

_LOCATION = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = sorted(_LOCATION)


def __getattr__(name: str) -> Any:
    module = _LOCATION.get(name)
    if module is None:
        raise AttributeError(f"module 'ensim.core' has no attribute '{name}'")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
