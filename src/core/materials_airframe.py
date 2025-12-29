"""
Airframe Materials Database.

Material properties for rocket airframe components
to enable automatic mass calculation from geometry.
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class AirframeMaterial(Enum):
    """Available airframe materials."""
    CARDBOARD = "cardboard"
    FIBERGLASS = "fiberglass"
    CARBON_FIBER = "carbon_fiber"
    PLYWOOD = "plywood"
    ALUMINUM = "aluminum"
    PHENOLIC = "phenolic"
    PLA = "pla"
    ABS = "abs"
    BALSA = "balsa"
    KRAFT_PAPER = "kraft_paper"


@dataclass
class MaterialProperties:
    """Physical properties of airframe materials."""
    name: str
    density: float  # kg/m³
    typical_thickness: float  # m (typical wall thickness)
    description: str = ""


# Airframe Materials Database
AIRFRAME_MATERIALS: Dict[AirframeMaterial, MaterialProperties] = {
    AirframeMaterial.CARDBOARD: MaterialProperties(
        name="Cardboard (Spiral Wound)",
        density=680.0,
        typical_thickness=0.002,
        description="Standard model rocket body tube"
    ),
    AirframeMaterial.FIBERGLASS: MaterialProperties(
        name="Fiberglass (G10/G12)",
        density=1850.0,
        typical_thickness=0.0015,
        description="High-power rockets, strong and lightweight"
    ),
    AirframeMaterial.CARBON_FIBER: MaterialProperties(
        name="Carbon Fiber Composite",
        density=1600.0,
        typical_thickness=0.001,
        description="Competition rockets, highest strength-to-weight"
    ),
    AirframeMaterial.PLYWOOD: MaterialProperties(
        name="Aircraft Plywood",
        density=600.0,
        typical_thickness=0.003,
        description="Fins and centering rings"
    ),
    AirframeMaterial.ALUMINUM: MaterialProperties(
        name="Aluminum 6061-T6",
        density=2700.0,
        typical_thickness=0.0016,
        description="Metal airframe, reusable rockets"
    ),
    AirframeMaterial.PHENOLIC: MaterialProperties(
        name="Phenolic (Paper-Resin)",
        density=1400.0,
        typical_thickness=0.002,
        description="Mid-power body tubes"
    ),
    AirframeMaterial.PLA: MaterialProperties(
        name="PLA (3D Printed)",
        density=1240.0,
        typical_thickness=0.002,
        description="3D printed components"
    ),
    AirframeMaterial.ABS: MaterialProperties(
        name="ABS (3D Printed)",
        density=1050.0,
        typical_thickness=0.002,
        description="3D printed, higher temperature resistance"
    ),
    AirframeMaterial.BALSA: MaterialProperties(
        name="Balsa Wood",
        density=160.0,
        typical_thickness=0.003,
        description="Very lightweight fins"
    ),
    AirframeMaterial.KRAFT_PAPER: MaterialProperties(
        name="Kraft Paper Tube",
        density=750.0,
        typical_thickness=0.0015,
        description="Economy body tubes"
    ),
}


def get_material(material: AirframeMaterial) -> MaterialProperties:
    """Get material properties by type."""
    return AIRFRAME_MATERIALS.get(material, AIRFRAME_MATERIALS[AirframeMaterial.CARDBOARD])


def get_material_names() -> list:
    """Get list of material names for UI dropdowns."""
    return [m.value for m in AirframeMaterial]


def calculate_tube_mass(
    length: float,
    diameter: float,
    material: AirframeMaterial,
    wall_thickness: float = None
) -> float:
    """
    Calculate mass of a cylindrical tube.
    
    Uses thin-wall approximation:
    Volume ≈ π × D × L × t
    Mass = Volume × density
    
    Args:
        length: Tube length (m)
        diameter: Outer diameter (m)
        material: Material type
        wall_thickness: Wall thickness (m), or use material default
        
    Returns:
        Mass (kg)
    """
    import numpy as np
    
    props = get_material(material)
    t = wall_thickness if wall_thickness else props.typical_thickness
    
    volume = np.pi * diameter * length * t
    mass = volume * props.density
    
    return mass


def calculate_nose_mass(
    length: float,
    diameter: float,
    material: AirframeMaterial,
    wall_thickness: float = None
) -> float:
    """
    Calculate mass of a nose cone (hollow).
    
    Approximation: half of equivalent tube
    
    Args:
        length: Nose length (m)
        diameter: Base diameter (m)
        material: Material type
        wall_thickness: Wall thickness (m)
        
    Returns:
        Mass (kg)
    """
    # Nose cone surface area is roughly half of a cylinder
    tube_mass = calculate_tube_mass(length, diameter, material, wall_thickness)
    nose_mass = tube_mass * 0.6  # Slightly more than half due to thickness
    
    return nose_mass


def calculate_fin_mass(
    root_chord: float,
    tip_chord: float,
    span: float,
    thickness: float,
    material: AirframeMaterial,
    fin_count: int = 4
) -> float:
    """
    Calculate total mass of fin set.
    
    Fin area = (root + tip) × span / 2
    Volume = area × thickness
    Mass = volume × density × fin_count
    
    Args:
        root_chord, tip_chord, span: Fin geometry (m)
        thickness: Fin thickness (m)
        material: Material type
        fin_count: Number of fins
        
    Returns:
        Total mass of all fins (kg)
    """
    props = get_material(material)
    
    area = (root_chord + tip_chord) * span / 2
    volume = area * thickness
    mass_single = volume * props.density
    
    return mass_single * fin_count
