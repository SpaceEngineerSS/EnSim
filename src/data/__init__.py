"""Data modules - Propellant database and presets."""

from .propellant_presets import (
    PropellantCategory,
    PropellantPreset,
    PropellantSpec,
    ToxicityLevel,
    PROPELLANT_PRESETS,
    FUELS,
    OXIDIZERS,
    get_preset,
    get_presets_by_category,
    get_all_preset_names,
    find_preset_by_isp,
    find_preset_by_density,
    get_non_toxic_presets,
)

__all__ = [
    # Enums
    "PropellantCategory",
    "ToxicityLevel",
    # Data classes
    "PropellantPreset",
    "PropellantSpec",
    # Databases
    "PROPELLANT_PRESETS",
    "FUELS",
    "OXIDIZERS",
    # Functions
    "get_preset",
    "get_presets_by_category",
    "get_all_preset_names",
    "find_preset_by_isp",
    "find_preset_by_density",
    "get_non_toxic_presets",
]

