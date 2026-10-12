"""Unit conversion for display and data exchange.

Calculations inside EnSim are SI. This module converts values at the edges
(input fields, tables, exported files). Units are grouped by physical
quantity, so a symbol only has to be unique within its quantity: ``"g"`` is a
gram for MASS and standard gravity for ACCELERATION.

Conversion factors are exact where a definition exists (international foot
and pound, standard gravity, standard atmosphere).
"""

from __future__ import annotations

from enum import Enum

from ensim.core.constants import G0

_FOOT = 0.3048
_INCH = 0.0254
_POUND = 0.45359237
_POUND_FORCE = _POUND * G0
_PSI = _POUND_FORCE / _INCH**2


class Quantity(Enum):
    LENGTH = "length"
    AREA = "area"
    VOLUME = "volume"
    MASS = "mass"
    TIME = "time"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    FORCE = "force"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    DENSITY = "density"
    MASS_FLOW = "mass_flow"
    SPECIFIC_IMPULSE = "specific_impulse"
    ENERGY = "energy"
    POWER = "power"
    HEAT_FLUX = "heat_flux"
    ANGLE = "angle"


class UnitSystem(Enum):
    SI = "si"
    IMPERIAL = "imperial"


# value_SI = value * scale + offset
UNITS: dict[Quantity, dict[str, tuple[float, float]]] = {
    Quantity.LENGTH: {
        "m": (1.0, 0.0),
        "km": (1e3, 0.0),
        "cm": (1e-2, 0.0),
        "mm": (1e-3, 0.0),
        "in": (_INCH, 0.0),
        "ft": (_FOOT, 0.0),
        "mi": (1609.344, 0.0),
        "nmi": (1852.0, 0.0),
    },
    Quantity.AREA: {
        "m²": (1.0, 0.0),
        "cm²": (1e-4, 0.0),
        "mm²": (1e-6, 0.0),
        "in²": (_INCH**2, 0.0),
        "ft²": (_FOOT**2, 0.0),
    },
    Quantity.VOLUME: {
        "m³": (1.0, 0.0),
        "L": (1e-3, 0.0),
        "in³": (_INCH**3, 0.0),
        "ft³": (_FOOT**3, 0.0),
        "gal": (3.785411784e-3, 0.0),
    },
    Quantity.MASS: {
        "kg": (1.0, 0.0),
        "g": (1e-3, 0.0),
        "t": (1e3, 0.0),
        "lb": (_POUND, 0.0),
        "slug": (_POUND_FORCE / _FOOT, 0.0),
    },
    Quantity.TIME: {
        "s": (1.0, 0.0),
        "ms": (1e-3, 0.0),
        "min": (60.0, 0.0),
        "h": (3600.0, 0.0),
    },
    Quantity.VELOCITY: {
        "m/s": (1.0, 0.0),
        "km/h": (1.0 / 3.6, 0.0),
        "ft/s": (_FOOT, 0.0),
        "mph": (0.44704, 0.0),
        "kn": (1852.0 / 3600.0, 0.0),
    },
    Quantity.ACCELERATION: {
        "m/s²": (1.0, 0.0),
        "ft/s²": (_FOOT, 0.0),
        "g": (G0, 0.0),
    },
    Quantity.FORCE: {
        "N": (1.0, 0.0),
        "kN": (1e3, 0.0),
        "MN": (1e6, 0.0),
        "lbf": (_POUND_FORCE, 0.0),
        "klbf": (1e3 * _POUND_FORCE, 0.0),
    },
    Quantity.PRESSURE: {
        "Pa": (1.0, 0.0),
        "kPa": (1e3, 0.0),
        "MPa": (1e6, 0.0),
        "bar": (1e5, 0.0),
        "atm": (101325.0, 0.0),
        "psi": (_PSI, 0.0),
    },
    Quantity.TEMPERATURE: {
        "K": (1.0, 0.0),
        "°C": (1.0, 273.15),
        "°F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
        "°R": (5.0 / 9.0, 0.0),
    },
    Quantity.DENSITY: {
        "kg/m³": (1.0, 0.0),
        "g/cm³": (1e3, 0.0),
        "lb/ft³": (_POUND / _FOOT**3, 0.0),
        "lb/in³": (_POUND / _INCH**3, 0.0),
    },
    Quantity.MASS_FLOW: {
        "kg/s": (1.0, 0.0),
        "g/s": (1e-3, 0.0),
        "lb/s": (_POUND, 0.0),
    },
    Quantity.SPECIFIC_IMPULSE: {
        # Isp in seconds is the SI-side convention used throughout EnSim
        "s": (1.0, 0.0),
        "m/s": (1.0 / G0, 0.0),
        "N·s/kg": (1.0 / G0, 0.0),
    },
    Quantity.ENERGY: {
        "J": (1.0, 0.0),
        "kJ": (1e3, 0.0),
        "MJ": (1e6, 0.0),
        "BTU": (1055.05585262, 0.0),
    },
    Quantity.POWER: {
        "W": (1.0, 0.0),
        "kW": (1e3, 0.0),
        "MW": (1e6, 0.0),
        "hp": (745.69987158227022, 0.0),
    },
    Quantity.HEAT_FLUX: {
        "W/m²": (1.0, 0.0),
        "kW/m²": (1e3, 0.0),
        "MW/m²": (1e6, 0.0),
        "BTU/(in²·s)": (1055.05585262 / _INCH**2, 0.0),
    },
    Quantity.ANGLE: {
        "rad": (1.0, 0.0),
        "deg": (3.141592653589793 / 180.0, 0.0),
    },
}

SI_UNITS: dict[Quantity, str] = {
    Quantity.LENGTH: "m",
    Quantity.AREA: "m²",
    Quantity.VOLUME: "m³",
    Quantity.MASS: "kg",
    Quantity.TIME: "s",
    Quantity.VELOCITY: "m/s",
    Quantity.ACCELERATION: "m/s²",
    Quantity.FORCE: "N",
    Quantity.PRESSURE: "Pa",
    Quantity.TEMPERATURE: "K",
    Quantity.DENSITY: "kg/m³",
    Quantity.MASS_FLOW: "kg/s",
    Quantity.SPECIFIC_IMPULSE: "s",
    Quantity.ENERGY: "J",
    Quantity.POWER: "W",
    Quantity.HEAT_FLUX: "W/m²",
    Quantity.ANGLE: "rad",
}

# Units shown in the interface for each system
DISPLAY_UNITS: dict[UnitSystem, dict[Quantity, str]] = {
    UnitSystem.SI: {
        **SI_UNITS,
        Quantity.FORCE: "kN",
        Quantity.PRESSURE: "bar",
        Quantity.AREA: "cm²",
        Quantity.ANGLE: "deg",
        Quantity.HEAT_FLUX: "MW/m²",
    },
    UnitSystem.IMPERIAL: {
        Quantity.LENGTH: "ft",
        Quantity.AREA: "in²",
        Quantity.VOLUME: "ft³",
        Quantity.MASS: "lb",
        Quantity.TIME: "s",
        Quantity.VELOCITY: "ft/s",
        Quantity.ACCELERATION: "ft/s²",
        Quantity.FORCE: "lbf",
        Quantity.PRESSURE: "psi",
        Quantity.TEMPERATURE: "°R",
        Quantity.DENSITY: "lb/ft³",
        Quantity.MASS_FLOW: "lb/s",
        Quantity.SPECIFIC_IMPULSE: "s",
        Quantity.ENERGY: "BTU",
        Quantity.POWER: "hp",
        Quantity.HEAT_FLUX: "BTU/(in²·s)",
        Quantity.ANGLE: "deg",
    },
}


def _lookup(quantity: Quantity, unit: str) -> tuple[float, float]:
    try:
        return UNITS[quantity][unit]
    except KeyError:
        known = ", ".join(UNITS[quantity])
        raise ValueError(f"Unknown {quantity.value} unit '{unit}'. Known: {known}") from None


def to_si(value: float, quantity: Quantity, unit: str) -> float:
    """Convert ``value`` expressed in ``unit`` to the SI unit of ``quantity``."""
    scale, offset = _lookup(quantity, unit)
    return value * scale + offset


def from_si(value: float, quantity: Quantity, unit: str) -> float:
    """Convert an SI ``value`` of ``quantity`` to ``unit``."""
    scale, offset = _lookup(quantity, unit)
    return (value - offset) / scale


def convert(value: float, quantity: Quantity, from_unit: str, to_unit: str) -> float:
    """Convert between two units of the same quantity."""
    return from_si(to_si(value, quantity, from_unit), quantity, to_unit)


def display_unit(quantity: Quantity, system: UnitSystem = UnitSystem.SI) -> str:
    """Unit the interface uses for ``quantity`` in ``system``."""
    return DISPLAY_UNITS[system][quantity]
