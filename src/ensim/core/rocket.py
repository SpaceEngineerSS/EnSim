"""
Rocket Component Definitions.

Defines the structural components of a rocket vehicle for
flight simulation and stability analysis.
"""

from dataclasses import dataclass, field
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
            return (1 / 3) * np.pi * (self.diameter / 2) ** 2 * self.length
        else:  # Ogive approximation
            return 0.5 * np.pi * (self.diameter / 2) ** 2 * self.length

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
    r"""
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
        return (2 / 3) * cr * (1 + ct / cr + (ct / cr) ** 2) / (1 + ct / cr)

    @property
    def aspect_ratio(self) -> float:
        """Fin aspect ratio."""
        return self.span**2 / self.area if self.area > 0 else 0


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
    axial_drag_coefficient: float = 0.45

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
        return self.nose.mass + self.body.mass + self.fins.mass + self.engine.engine_mass_dry

    @property
    def wet_mass(self) -> float:
        """Total mass with full propellant (kg)."""
        return self.dry_mass + self.engine.propellant_mass

    def propellant_at_time(self, t: float) -> float:
        """Propellant left at time t after ignition, kg.

        Uses ``engine.mass_flow_rate`` and ``engine.burn_time`` when they are
        set. Without a burn definition the tanks are reported full, which is
        the liftoff configuration stability checks need.
        """
        full = self.engine.propellant_mass
        if t <= 0.0 or self.engine.mass_flow_rate <= 0.0:
            return full
        if self.engine.burn_time > 0.0 and t >= self.engine.burn_time:
            return max(0.0, full - self.engine.mass_flow_rate * self.engine.burn_time)
        return max(0.0, full - self.engine.mass_flow_rate * t)

    def get_mass_at_time(self, t: float) -> float:
        """Total mass at time t after ignition, kg."""
        return self.get_mass_from_propellant(self.propellant_at_time(t))

    def get_cg_at_time(self, t: float) -> float:
        """CG position from the nose tip at time t after ignition, m."""
        return self.get_cg_from_propellant(self.propellant_at_time(t))

    @property
    def initial_propellant_mass(self) -> float:
        """Propellant loaded at liftoff, kg (initial value of state[13])."""
        return self.engine.propellant_mass

    def get_mass_from_propellant(self, m_prop: float) -> float:
        """Total mass for a given remaining propellant mass, kg."""
        return self.dry_mass + max(0.0, m_prop)

    def get_cg_from_propellant(self, m_prop: float) -> float:
        """CG position from the nose tip for a given remaining propellant mass, m.

        The propellant is lumped at ``engine.position`` (the tank centre).
        """
        m_prop = max(0.0, m_prop)
        x_body = self.nose.length + self.body.length / 2
        x_fins = self.fins.position + self.fins.fin.root_chord / 2
        moment = (
            self.nose.mass * self.nose.position_cg
            + self.body.mass * x_body
            + self.fins.mass * x_fins
            + (self.engine.engine_mass_dry + m_prop) * self.engine.position
        )
        total_mass = self.dry_mass + m_prop
        if total_mass <= 0.0:
            raise ValueError("Rocket mass must be positive")
        return moment / total_mass


def create_default_rocket() -> Rocket:
    """A small liquid-propellant sounding rocket, 9 kg at liftoff.

    The tank holds 3.2 kg of LOX/kerosene at O/F 3 in 0.45 m of a 0.12 m
    airframe, and the static margin is about 1.4 calibers with full tanks.
    """
    return Rocket(
        name="Default Rocket",
        nose=NoseCone(shape=NoseShape.OGIVE, length=0.45, diameter=0.12, mass=0.7),
        body=BodyTube(length=1.8, diameter=0.12, mass=2.5),
        fins=FinSet(
            fin=Fin(root_chord=0.26, tip_chord=0.10, span=0.16, sweep_angle=35.0), count=4, mass=0.6
        ),
        engine=EngineMount(engine_mass_dry=2.0, fuel_mass=0.8, oxidizer_mass=2.4, tank_length=0.45),
    )
