"""
Rocket Component Definitions.

Defines the structural components of a rocket vehicle for
flight simulation and stability analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import numpy as np


class NoseShape(Enum):
    """Nose cone shape types."""
    CONICAL = "conical"
    OGIVE = "ogive"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    HAACK = "haack"


@dataclass
class NoseCone:
    """
    Nose cone component.
    
    Attributes:
        shape: Aerodynamic shape type
        length: Total length (m)
        diameter: Base diameter (m)
        mass: Component mass (kg)
        material: Material name for reference
    """
    shape: NoseShape = NoseShape.OGIVE
    length: float = 0.3  # m
    diameter: float = 0.1  # m
    mass: float = 0.5  # kg
    material: str = "Fiberglass"
    
    @property
    def fineness_ratio(self) -> float:
        """Length to diameter ratio."""
        return self.length / self.diameter if self.diameter > 0 else 0
    
    @property
    def volume(self) -> float:
        """Approximate volume (m³)."""
        if self.shape == NoseShape.CONICAL:
            return (1/3) * np.pi * (self.diameter/2)**2 * self.length
        else:  # Ogive approximation
            return 0.5 * np.pi * (self.diameter/2)**2 * self.length
    
    @property
    def position_cg(self) -> float:
        """CG distance from nose tip (m)."""
        if self.shape == NoseShape.CONICAL:
            return 0.75 * self.length  # 3/4 from tip
        else:
            return 0.6 * self.length  # Ogive


@dataclass
class BodyTube:
    """
    Cylindrical body tube component.
    
    Attributes:
        length: Tube length (m)
        diameter: Outer diameter (m)
        wall_thickness: Wall thickness (m)
        mass: Component mass (kg)
    """
    length: float = 1.0  # m
    diameter: float = 0.1  # m
    wall_thickness: float = 0.002  # m
    mass: float = 1.0  # kg
    material: str = "Fiberglass"
    
    @property
    def inner_diameter(self) -> float:
        """Inner diameter (m)."""
        return self.diameter - 2 * self.wall_thickness
    
    @property
    def cross_section_area(self) -> float:
        """Reference cross-section area (m²)."""
        return np.pi * (self.diameter / 2) ** 2
    
    @property
    def wetted_area(self) -> float:
        """External surface area (m²)."""
        return np.pi * self.diameter * self.length


@dataclass
class Fin:
    """
    Single fin definition (trapezoidal planform).
    
              Tip Chord
            ┌─────────┐
           /           \
          /             \  Span
         /               \
        └─────────────────┘
           Root Chord
    """
    root_chord: float = 0.15  # m
    tip_chord: float = 0.05  # m
    span: float = 0.08  # m (semi-span from body)
    sweep_angle: float = 30.0  # degrees (leading edge sweep)
    thickness: float = 0.003  # m
    
    @property
    def area(self) -> float:
        """Planform area of single fin (m²)."""
        return 0.5 * (self.root_chord + self.tip_chord) * self.span
    
    @property
    def mac(self) -> float:
        """Mean aerodynamic chord (m)."""
        cr, ct = self.root_chord, self.tip_chord
        return (2/3) * cr * (1 + ct/cr + (ct/cr)**2) / (1 + ct/cr)
    
    @property
    def aspect_ratio(self) -> float:
        """Fin aspect ratio."""
        return self.span ** 2 / self.area if self.area > 0 else 0


@dataclass
class FinSet:
    """
    Set of fins around the body.
    
    Attributes:
        fin: Single fin geometry
        count: Number of fins (typically 3 or 4)
        position: Distance from nose tip to fin root leading edge (m)
        mass: Total mass of all fins (kg)
    """
    fin: Fin = field(default_factory=Fin)
    count: int = 4
    position: float = 0.0  # Set dynamically
    mass: float = 0.3  # kg total
    material: str = "Plywood"
    
    @property
    def total_area(self) -> float:
        """Total planform area of all fins (m²)."""
        return self.fin.area * self.count


@dataclass
class EngineMount:
    """
    Engine mount and propellant configuration.
    
    Links the engine simulation results to the vehicle.
    """
    engine_mass_dry: float = 5.0  # kg (engine hardware)
    fuel_mass: float = 10.0  # kg
    oxidizer_mass: float = 50.0  # kg (at O/F ~ 5)
    tank_length: float = 0.5  # m (combined tank length)
    position: float = 0.0  # Distance from nose to engine CG
    
    # Engine performance (filled from simulation)
    thrust_vac: float = 0.0  # N
    isp_vac: float = 0.0  # s
    mass_flow_rate: float = 0.0  # kg/s
    burn_time: float = 0.0  # s
    
    @property
    def propellant_mass(self) -> float:
        """Total propellant mass (kg)."""
        return self.fuel_mass + self.oxidizer_mass
    
    @property
    def total_mass(self) -> float:
        """Total mass including propellant (kg)."""
        return self.engine_mass_dry + self.propellant_mass


@dataclass
class Rocket:
    """
    Complete rocket vehicle definition.
    
    Combines all components and provides aggregate properties.
    """
    name: str = "EnSim Rocket"
    nose: NoseCone = field(default_factory=NoseCone)
    body: BodyTube = field(default_factory=BodyTube)
    fins: FinSet = field(default_factory=FinSet)
    engine: EngineMount = field(default_factory=EngineMount)
    
    def __post_init__(self):
        """Calculate component positions after init."""
        self._update_positions()
    
    def _update_positions(self):
        """Update component positions based on geometry."""
        # Fins at bottom of body tube
        body_end = self.nose.length + self.body.length
        self.fins.position = body_end - self.fins.fin.root_chord
        
        # Engine at very bottom
        self.engine.position = body_end - self.engine.tank_length / 2
    
    @property
    def total_length(self) -> float:
        """Total rocket length (m)."""
        return self.nose.length + self.body.length
    
    @property
    def reference_diameter(self) -> float:
        """Reference diameter for aerodynamics (m)."""
        return self.body.diameter
    
    @property
    def reference_area(self) -> float:
        """Reference area for aerodynamics (m²)."""
        return np.pi * (self.reference_diameter / 2) ** 2
    
    @property
    def dry_mass(self) -> float:
        """Total dry mass without propellant (kg)."""
        return (self.nose.mass + 
                self.body.mass + 
                self.fins.mass + 
                self.engine.engine_mass_dry)
    
    @property
    def wet_mass(self) -> float:
        """Total mass with full propellant (kg)."""
        return self.dry_mass + self.engine.propellant_mass
    
    def get_mass_at_time(self, t: float) -> float:
        """
        Get rocket mass at time t during burn.
        
        Args:
            t: Time since ignition (s)
            
        Returns:
            Current mass (kg)
        """
        if t < 0:
            return self.wet_mass
        if t >= self.engine.burn_time:
            return self.dry_mass
        
        propellant_consumed = self.engine.mass_flow_rate * t
        remaining_propellant = max(0, self.engine.propellant_mass - propellant_consumed)
        return self.dry_mass + remaining_propellant
    
    def get_cg_at_time(self, t: float) -> float:
        """
        Calculate CG position from nose tip at time t.
        
        Args:
            t: Time since ignition (s)
            
        Returns:
            CG position from nose (m)
        """
        # Component CG positions (from nose tip)
        x_nose = self.nose.position_cg
        x_body = self.nose.length + self.body.length / 2
        x_fins = self.fins.position + self.fins.fin.root_chord / 2
        x_engine = self.engine.position
        
        # Masses
        m_nose = self.nose.mass
        m_body = self.body.mass
        m_fins = self.fins.mass
        m_engine_dry = self.engine.engine_mass_dry
        
        # Propellant mass at time t
        if t < 0:
            m_prop = self.engine.propellant_mass
        elif t >= self.engine.burn_time:
            m_prop = 0
        else:
            consumed = self.engine.mass_flow_rate * t
            m_prop = max(0, self.engine.propellant_mass - consumed)
        
        # Weighted CG
        total_moment = (m_nose * x_nose + 
                       m_body * x_body + 
                       m_fins * x_fins + 
                       (m_engine_dry + m_prop) * x_engine)
        total_mass = m_nose + m_body + m_fins + m_engine_dry + m_prop
        
        return total_moment / total_mass if total_mass > 0 else 0


def create_default_rocket() -> Rocket:
    """Create a default small rocket configuration."""
    return Rocket(
        name="Default Rocket",
        nose=NoseCone(
            shape=NoseShape.OGIVE,
            length=0.25,
            diameter=0.1,
            mass=0.3
        ),
        body=BodyTube(
            length=1.0,
            diameter=0.1,
            mass=1.0
        ),
        fins=FinSet(
            fin=Fin(
                root_chord=0.12,
                tip_chord=0.04,
                span=0.06,
                sweep_angle=35.0
            ),
            count=4,
            mass=0.2
        ),
        engine=EngineMount(
            engine_mass_dry=2.0,
            fuel_mass=5.0,
            oxidizer_mass=25.0,
            tank_length=0.4
        )
    )
