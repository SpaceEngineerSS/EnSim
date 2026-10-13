# Other modules

## Units

::: ensim.utils.units
    options:
      members:
        - Quantity
        - UnitSystem
        - to_si
        - from_si
        - convert
        - display_unit

## Projects

::: ensim.core.project_manager
    options:
      members:
        - ProjectManager
        - ProjectData
        - ProjectFileError

## Equilibrium

::: ensim.core.chemistry
    options:
      members:
        - CombustionProblem

## Cooling

::: ensim.core.cooling
    options:
      members:
        - analyze_cooling_system
        - CoolingSystemDesign
        - CoolingChannel
        - bartz_heat_transfer_coefficient
        - coolant_heat_transfer_coefficient

## Acoustics

::: ensim.core.instability
    options:
      members:
        - analyze_combustion_instability
        - InstabilityResult
        - AcousticMode
        - calculate_chugging_frequency

## Engine uncertainty

::: ensim.core.engine_uq
    options:
      members:
        - EngineUQInput
        - EngineUQAnalyzer
        - EngineUQResult
