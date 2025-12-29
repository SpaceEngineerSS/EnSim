"""
Materials Database for Thermal Analysis.

Contains thermal properties of common rocket engine materials.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Material:
    """Thermal and mechanical properties of a material."""
    name: str
    thermal_conductivity: float  # W/(m·K)
    melting_point: float  # K
    density: float  # kg/m³
    specific_heat: float  # J/(kg·K)
    emissivity: float  # dimensionless (0-1)
    max_service_temp: float  # K (practical limit before degradation)


# Material database
MATERIALS: Dict[str, Material] = {
    "Copper (OFHC)": Material(
        name="Copper (OFHC)",
        thermal_conductivity=385.0,
        melting_point=1358.0,
        density=8960.0,
        specific_heat=385.0,
        emissivity=0.03,
        max_service_temp=700.0  # Loses strength at high T
    ),
    "Inconel 718": Material(
        name="Inconel 718",
        thermal_conductivity=11.2,
        melting_point=1609.0,
        density=8190.0,
        specific_heat=435.0,
        emissivity=0.3,
        max_service_temp=980.0  # Excellent high-temp strength
    ),
    "Inconel 625": Material(
        name="Inconel 625",
        thermal_conductivity=9.8,
        melting_point=1623.0,
        density=8440.0,
        specific_heat=410.0,
        emissivity=0.28,
        max_service_temp=1000.0
    ),
    "Stainless Steel 304": Material(
        name="Stainless Steel 304",
        thermal_conductivity=16.2,
        melting_point=1673.0,
        density=7900.0,
        specific_heat=500.0,
        emissivity=0.6,
        max_service_temp=870.0
    ),
    "Stainless Steel 316": Material(
        name="Stainless Steel 316",
        thermal_conductivity=16.3,
        melting_point=1673.0,
        density=8000.0,
        specific_heat=500.0,
        emissivity=0.6,
        max_service_temp=870.0
    ),
    "Graphite (ATJ)": Material(
        name="Graphite (ATJ)",
        thermal_conductivity=120.0,
        melting_point=3800.0,  # Sublimation
        density=1770.0,
        specific_heat=710.0,
        emissivity=0.85,
        max_service_temp=3000.0  # In inert atmosphere
    ),
    "Carbon-Carbon": Material(
        name="Carbon-Carbon",
        thermal_conductivity=50.0,
        melting_point=3800.0,
        density=1600.0,
        specific_heat=720.0,
        emissivity=0.9,
        max_service_temp=2500.0  # In oxidizing
    ),
    "Niobium C-103": Material(
        name="Niobium C-103",
        thermal_conductivity=42.0,
        melting_point=2623.0,
        density=8860.0,
        specific_heat=265.0,
        emissivity=0.2,
        max_service_temp=1370.0
    ),
    "Tungsten": Material(
        name="Tungsten",
        thermal_conductivity=173.0,
        melting_point=3695.0,
        density=19300.0,
        specific_heat=132.0,
        emissivity=0.04,
        max_service_temp=2500.0
    ),
    "Rhenium": Material(
        name="Rhenium",
        thermal_conductivity=48.0,
        melting_point=3459.0,
        density=21020.0,
        specific_heat=137.0,
        emissivity=0.25,
        max_service_temp=2200.0
    ),
}


def get_material(name: str) -> Material:
    """Get material by name."""
    if name in MATERIALS:
        return MATERIALS[name]
    raise ValueError(f"Unknown material: {name}")


def list_materials() -> list:
    """Get list of available material names."""
    return list(MATERIALS.keys())


def get_materials_by_temp(min_service_temp: float) -> list:
    """Get materials that can handle given temperature."""
    return [
        name for name, mat in MATERIALS.items()
        if mat.max_service_temp >= min_service_temp
    ]
