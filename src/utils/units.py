"""
Comprehensive unit conversion system for rocket propulsion.

Supports SI (metric) and Imperial (US customary) units with
automatic conversion between systems.

All internal calculations use SI units. This module provides
conversions for user interface display and data import/export.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


class UnitSystem(Enum):
    """Unit system selection."""
    SI = auto()       # International System (metric)
    IMPERIAL = auto() # US Customary units


class UnitCategory(Enum):
    """Categories of physical quantities."""
    LENGTH = auto()
    MASS = auto()
    TIME = auto()
    FORCE = auto()
    PRESSURE = auto()
    TEMPERATURE = auto()
    VELOCITY = auto()
    ACCELERATION = auto()
    DENSITY = auto()
    MASS_FLOW = auto()
    SPECIFIC_IMPULSE = auto()
    AREA = auto()
    VOLUME = auto()
    ENERGY = auto()
    POWER = auto()
    HEAT_FLUX = auto()
    THERMAL_CONDUCTIVITY = auto()
    DYNAMIC_VISCOSITY = auto()
    SPECIFIC_HEAT = auto()
    MOLECULAR_WEIGHT = auto()


@dataclass
class UnitDefinition:
    """Definition of a unit with conversion factor."""
    name: str               # Full name
    symbol: str             # Symbol/abbreviation
    category: UnitCategory
    to_si: float            # Multiply by this to convert to SI
    from_si: float          # Multiply by this to convert from SI


# =============================================================================
# Unit Definitions
# =============================================================================

# Length units (SI base: meter)
LENGTH_UNITS = {
    "m": UnitDefinition("meter", "m", UnitCategory.LENGTH, 1.0, 1.0),
    "km": UnitDefinition("kilometer", "km", UnitCategory.LENGTH, 1000.0, 0.001),
    "cm": UnitDefinition("centimeter", "cm", UnitCategory.LENGTH, 0.01, 100.0),
    "mm": UnitDefinition("millimeter", "mm", UnitCategory.LENGTH, 0.001, 1000.0),
    "in": UnitDefinition("inch", "in", UnitCategory.LENGTH, 0.0254, 39.3701),
    "ft": UnitDefinition("foot", "ft", UnitCategory.LENGTH, 0.3048, 3.28084),
    "yd": UnitDefinition("yard", "yd", UnitCategory.LENGTH, 0.9144, 1.09361),
    "mi": UnitDefinition("mile", "mi", UnitCategory.LENGTH, 1609.344, 6.2137e-4),
    "nmi": UnitDefinition("nautical mile", "nmi", UnitCategory.LENGTH, 1852.0, 5.3996e-4),
}

# Mass units (SI base: kilogram)
MASS_UNITS = {
    "kg": UnitDefinition("kilogram", "kg", UnitCategory.MASS, 1.0, 1.0),
    "g": UnitDefinition("gram", "g", UnitCategory.MASS, 0.001, 1000.0),
    "mg": UnitDefinition("milligram", "mg", UnitCategory.MASS, 1e-6, 1e6),
    "t": UnitDefinition("metric ton", "t", UnitCategory.MASS, 1000.0, 0.001),
    "lb": UnitDefinition("pound", "lb", UnitCategory.MASS, 0.453592, 2.20462),
    "lbm": UnitDefinition("pound-mass", "lbm", UnitCategory.MASS, 0.453592, 2.20462),
    "oz": UnitDefinition("ounce", "oz", UnitCategory.MASS, 0.0283495, 35.274),
    "slug": UnitDefinition("slug", "slug", UnitCategory.MASS, 14.5939, 0.0685218),
    "st": UnitDefinition("short ton", "st", UnitCategory.MASS, 907.185, 0.00110231),
}

# Time units (SI base: second)
TIME_UNITS = {
    "s": UnitDefinition("second", "s", UnitCategory.TIME, 1.0, 1.0),
    "ms": UnitDefinition("millisecond", "ms", UnitCategory.TIME, 0.001, 1000.0),
    "min": UnitDefinition("minute", "min", UnitCategory.TIME, 60.0, 1/60),
    "h": UnitDefinition("hour", "h", UnitCategory.TIME, 3600.0, 1/3600),
    "d": UnitDefinition("day", "d", UnitCategory.TIME, 86400.0, 1/86400),
}

# Force units (SI base: Newton)
FORCE_UNITS = {
    "N": UnitDefinition("Newton", "N", UnitCategory.FORCE, 1.0, 1.0),
    "kN": UnitDefinition("kilonewton", "kN", UnitCategory.FORCE, 1000.0, 0.001),
    "MN": UnitDefinition("meganewton", "MN", UnitCategory.FORCE, 1e6, 1e-6),
    "lbf": UnitDefinition("pound-force", "lbf", UnitCategory.FORCE, 4.44822, 0.224809),
    "klbf": UnitDefinition("kilo-pound-force", "klbf", UnitCategory.FORCE, 4448.22, 2.24809e-4),
    "kgf": UnitDefinition("kilogram-force", "kgf", UnitCategory.FORCE, 9.80665, 0.101972),
    "dyn": UnitDefinition("dyne", "dyn", UnitCategory.FORCE, 1e-5, 1e5),
}

# Pressure units (SI base: Pascal)
PRESSURE_UNITS = {
    "Pa": UnitDefinition("Pascal", "Pa", UnitCategory.PRESSURE, 1.0, 1.0),
    "kPa": UnitDefinition("kilopascal", "kPa", UnitCategory.PRESSURE, 1000.0, 0.001),
    "MPa": UnitDefinition("megapascal", "MPa", UnitCategory.PRESSURE, 1e6, 1e-6),
    "bar": UnitDefinition("bar", "bar", UnitCategory.PRESSURE, 1e5, 1e-5),
    "mbar": UnitDefinition("millibar", "mbar", UnitCategory.PRESSURE, 100.0, 0.01),
    "atm": UnitDefinition("atmosphere", "atm", UnitCategory.PRESSURE, 101325.0, 9.8692e-6),
    "psi": UnitDefinition("pound per sq inch", "psi", UnitCategory.PRESSURE, 6894.76, 1.45038e-4),
    "psia": UnitDefinition("psi absolute", "psia", UnitCategory.PRESSURE, 6894.76, 1.45038e-4),
    "psig": UnitDefinition("psi gauge", "psig", UnitCategory.PRESSURE, 6894.76, 1.45038e-4),
    "torr": UnitDefinition("torr", "torr", UnitCategory.PRESSURE, 133.322, 7.5006e-3),
    "mmHg": UnitDefinition("mm mercury", "mmHg", UnitCategory.PRESSURE, 133.322, 7.5006e-3),
    "inHg": UnitDefinition("inches mercury", "inHg", UnitCategory.PRESSURE, 3386.39, 2.953e-4),
}

# Temperature units (SI base: Kelvin)
# Temperature requires offset conversion, handled specially
TEMPERATURE_UNITS = {
    "K": UnitDefinition("Kelvin", "K", UnitCategory.TEMPERATURE, 1.0, 1.0),
    "C": UnitDefinition("Celsius", "°C", UnitCategory.TEMPERATURE, 1.0, 1.0),  # +273.15 offset
    "F": UnitDefinition("Fahrenheit", "°F", UnitCategory.TEMPERATURE, 5/9, 9/5),  # Complex
    "R": UnitDefinition("Rankine", "°R", UnitCategory.TEMPERATURE, 5/9, 9/5),
}

# Velocity units (SI base: m/s)
VELOCITY_UNITS = {
    "m/s": UnitDefinition("meters/second", "m/s", UnitCategory.VELOCITY, 1.0, 1.0),
    "km/s": UnitDefinition("kilometers/second", "km/s", UnitCategory.VELOCITY, 1000.0, 0.001),
    "km/h": UnitDefinition("kilometers/hour", "km/h", UnitCategory.VELOCITY, 1/3.6, 3.6),
    "ft/s": UnitDefinition("feet/second", "ft/s", UnitCategory.VELOCITY, 0.3048, 3.28084),
    "mph": UnitDefinition("miles/hour", "mph", UnitCategory.VELOCITY, 0.44704, 2.23694),
    "kn": UnitDefinition("knots", "kn", UnitCategory.VELOCITY, 0.514444, 1.94384),
    "mach": UnitDefinition("Mach (sea level)", "M", UnitCategory.VELOCITY, 340.29, 2.9386e-3),
}

# Acceleration units (SI base: m/s²)
ACCELERATION_UNITS = {
    "m/s2": UnitDefinition("meters/second²", "m/s²", UnitCategory.ACCELERATION, 1.0, 1.0),
    "g": UnitDefinition("standard gravity", "g", UnitCategory.ACCELERATION, 9.80665, 0.101972),
    "ft/s2": UnitDefinition("feet/second²", "ft/s²", UnitCategory.ACCELERATION, 0.3048, 3.28084),
}

# Density units (SI base: kg/m³)
DENSITY_UNITS = {
    "kg/m3": UnitDefinition("kg/m³", "kg/m³", UnitCategory.DENSITY, 1.0, 1.0),
    "g/cm3": UnitDefinition("g/cm³", "g/cm³", UnitCategory.DENSITY, 1000.0, 0.001),
    "g/L": UnitDefinition("g/L", "g/L", UnitCategory.DENSITY, 1.0, 1.0),
    "lb/ft3": UnitDefinition("lb/ft³", "lb/ft³", UnitCategory.DENSITY, 16.0185, 0.062428),
    "lb/in3": UnitDefinition("lb/in³", "lb/in³", UnitCategory.DENSITY, 27679.9, 3.6127e-5),
    "slug/ft3": UnitDefinition("slug/ft³", "slug/ft³", UnitCategory.DENSITY, 515.379, 1.9403e-3),
}

# Mass flow rate units (SI base: kg/s)
MASS_FLOW_UNITS = {
    "kg/s": UnitDefinition("kg/s", "kg/s", UnitCategory.MASS_FLOW, 1.0, 1.0),
    "g/s": UnitDefinition("g/s", "g/s", UnitCategory.MASS_FLOW, 0.001, 1000.0),
    "kg/h": UnitDefinition("kg/h", "kg/h", UnitCategory.MASS_FLOW, 1/3600, 3600.0),
    "lb/s": UnitDefinition("lb/s", "lb/s", UnitCategory.MASS_FLOW, 0.453592, 2.20462),
    "lbm/s": UnitDefinition("lbm/s", "lbm/s", UnitCategory.MASS_FLOW, 0.453592, 2.20462),
    "slug/s": UnitDefinition("slug/s", "slug/s", UnitCategory.MASS_FLOW, 14.5939, 0.0685218),
}

# Specific impulse (SI base: seconds)
ISP_UNITS = {
    "s": UnitDefinition("seconds", "s", UnitCategory.SPECIFIC_IMPULSE, 1.0, 1.0),
    "m/s": UnitDefinition("m/s (exhaust vel)", "m/s", UnitCategory.SPECIFIC_IMPULSE, 1/9.80665, 9.80665),
    "ft/s": UnitDefinition("ft/s (exhaust vel)", "ft/s", UnitCategory.SPECIFIC_IMPULSE, 0.3048/9.80665, 9.80665/0.3048),
    "N·s/kg": UnitDefinition("N·s/kg", "N·s/kg", UnitCategory.SPECIFIC_IMPULSE, 1/9.80665, 9.80665),
    "lbf·s/lbm": UnitDefinition("lbf·s/lbm", "lbf·s/lbm", UnitCategory.SPECIFIC_IMPULSE, 1.0, 1.0),
}

# Area units (SI base: m²)
AREA_UNITS = {
    "m2": UnitDefinition("square meter", "m²", UnitCategory.AREA, 1.0, 1.0),
    "cm2": UnitDefinition("square centimeter", "cm²", UnitCategory.AREA, 1e-4, 1e4),
    "mm2": UnitDefinition("square millimeter", "mm²", UnitCategory.AREA, 1e-6, 1e6),
    "in2": UnitDefinition("square inch", "in²", UnitCategory.AREA, 6.4516e-4, 1550.0),
    "ft2": UnitDefinition("square foot", "ft²", UnitCategory.AREA, 0.092903, 10.7639),
}

# Volume units (SI base: m³)
VOLUME_UNITS = {
    "m3": UnitDefinition("cubic meter", "m³", UnitCategory.VOLUME, 1.0, 1.0),
    "L": UnitDefinition("liter", "L", UnitCategory.VOLUME, 0.001, 1000.0),
    "mL": UnitDefinition("milliliter", "mL", UnitCategory.VOLUME, 1e-6, 1e6),
    "cm3": UnitDefinition("cubic centimeter", "cm³", UnitCategory.VOLUME, 1e-6, 1e6),
    "gal": UnitDefinition("US gallon", "gal", UnitCategory.VOLUME, 0.00378541, 264.172),
    "ft3": UnitDefinition("cubic foot", "ft³", UnitCategory.VOLUME, 0.0283168, 35.3147),
    "in3": UnitDefinition("cubic inch", "in³", UnitCategory.VOLUME, 1.6387e-5, 61023.7),
}

# Energy units (SI base: Joule)
ENERGY_UNITS = {
    "J": UnitDefinition("Joule", "J", UnitCategory.ENERGY, 1.0, 1.0),
    "kJ": UnitDefinition("kilojoule", "kJ", UnitCategory.ENERGY, 1000.0, 0.001),
    "MJ": UnitDefinition("megajoule", "MJ", UnitCategory.ENERGY, 1e6, 1e-6),
    "cal": UnitDefinition("calorie", "cal", UnitCategory.ENERGY, 4.184, 0.239006),
    "kcal": UnitDefinition("kilocalorie", "kcal", UnitCategory.ENERGY, 4184.0, 2.39006e-4),
    "BTU": UnitDefinition("BTU", "BTU", UnitCategory.ENERGY, 1055.06, 9.4782e-4),
    "ft·lbf": UnitDefinition("foot-pound", "ft·lbf", UnitCategory.ENERGY, 1.35582, 0.737562),
}

# Power units (SI base: Watt)
POWER_UNITS = {
    "W": UnitDefinition("Watt", "W", UnitCategory.POWER, 1.0, 1.0),
    "kW": UnitDefinition("kilowatt", "kW", UnitCategory.POWER, 1000.0, 0.001),
    "MW": UnitDefinition("megawatt", "MW", UnitCategory.POWER, 1e6, 1e-6),
    "hp": UnitDefinition("horsepower", "hp", UnitCategory.POWER, 745.7, 1.341e-3),
    "BTU/s": UnitDefinition("BTU/second", "BTU/s", UnitCategory.POWER, 1055.06, 9.4782e-4),
    "BTU/h": UnitDefinition("BTU/hour", "BTU/h", UnitCategory.POWER, 0.293071, 3.41214),
}

# Heat flux units (SI base: W/m²)
HEAT_FLUX_UNITS = {
    "W/m2": UnitDefinition("W/m²", "W/m²", UnitCategory.HEAT_FLUX, 1.0, 1.0),
    "kW/m2": UnitDefinition("kW/m²", "kW/m²", UnitCategory.HEAT_FLUX, 1000.0, 0.001),
    "MW/m2": UnitDefinition("MW/m²", "MW/m²", UnitCategory.HEAT_FLUX, 1e6, 1e-6),
    "BTU/h·ft2": UnitDefinition("BTU/(h·ft²)", "BTU/(h·ft²)", UnitCategory.HEAT_FLUX, 3.15459, 0.316998),
}


# All unit dictionaries combined
ALL_UNITS = {
    **LENGTH_UNITS, **MASS_UNITS, **TIME_UNITS, **FORCE_UNITS,
    **PRESSURE_UNITS, **TEMPERATURE_UNITS, **VELOCITY_UNITS,
    **ACCELERATION_UNITS, **DENSITY_UNITS, **MASS_FLOW_UNITS,
    **ISP_UNITS, **AREA_UNITS, **VOLUME_UNITS, **ENERGY_UNITS,
    **POWER_UNITS, **HEAT_FLUX_UNITS
}


# =============================================================================
# Conversion Functions
# =============================================================================

def convert(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a value from one unit to another.

    Args:
        value: Numeric value to convert
        from_unit: Source unit symbol
        to_unit: Target unit symbol

    Returns:
        Converted value

    Raises:
        ValueError: If units are incompatible or unknown

    Example:
        >>> convert(100, "psi", "MPa")
        0.6894757
    """
    # Handle temperature specially (has offset)
    if from_unit in TEMPERATURE_UNITS and to_unit in TEMPERATURE_UNITS:
        return convert_temperature(value, from_unit, to_unit)

    if from_unit not in ALL_UNITS:
        raise ValueError(f"Unknown unit: {from_unit}")
    if to_unit not in ALL_UNITS:
        raise ValueError(f"Unknown unit: {to_unit}")

    from_def = ALL_UNITS[from_unit]
    to_def = ALL_UNITS[to_unit]

    if from_def.category != to_def.category:
        raise ValueError(f"Cannot convert {from_def.category.name} to {to_def.category.name}")

    # Convert: value -> SI -> target
    si_value = value * from_def.to_si
    return si_value * to_def.from_si


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between different scales.

    Handles the offset conversions properly.

    Args:
        value: Temperature value
        from_unit: Source unit ('K', 'C', 'F', 'R')
        to_unit: Target unit

    Returns:
        Converted temperature
    """
    # First convert to Kelvin
    if from_unit == "K":
        kelvin = value
    elif from_unit == "C":
        kelvin = value + 273.15
    elif from_unit == "F":
        kelvin = (value + 459.67) * 5/9
    elif from_unit == "R":
        kelvin = value * 5/9
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Then convert from Kelvin to target
    if to_unit == "K":
        return kelvin
    elif to_unit == "C":
        return kelvin - 273.15
    elif to_unit == "F":
        return kelvin * 9/5 - 459.67
    elif to_unit == "R":
        return kelvin * 9/5
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def to_si(value: float, unit: str) -> float:
    """Convert a value to SI units."""
    if unit in TEMPERATURE_UNITS:
        return convert_temperature(value, unit, "K")
    if unit not in ALL_UNITS:
        raise ValueError(f"Unknown unit: {unit}")
    return value * ALL_UNITS[unit].to_si


def from_si(value: float, unit: str) -> float:
    """Convert a value from SI to specified unit."""
    if unit in TEMPERATURE_UNITS:
        return convert_temperature(value, "K", unit)
    if unit not in ALL_UNITS:
        raise ValueError(f"Unknown unit: {unit}")
    return value * ALL_UNITS[unit].from_si


def get_unit_symbol(unit: str) -> str:
    """Get the display symbol for a unit."""
    if unit in ALL_UNITS:
        return ALL_UNITS[unit].symbol
    return unit


def get_units_for_category(category: UnitCategory) -> list[str]:
    """Get all unit symbols for a category."""
    return [k for k, v in ALL_UNITS.items() if v.category == category]


# =============================================================================
# Unit System Presets
# =============================================================================

SI_DEFAULTS = {
    UnitCategory.LENGTH: "m",
    UnitCategory.MASS: "kg",
    UnitCategory.TIME: "s",
    UnitCategory.FORCE: "N",
    UnitCategory.PRESSURE: "Pa",
    UnitCategory.TEMPERATURE: "K",
    UnitCategory.VELOCITY: "m/s",
    UnitCategory.ACCELERATION: "m/s2",
    UnitCategory.DENSITY: "kg/m3",
    UnitCategory.MASS_FLOW: "kg/s",
    UnitCategory.SPECIFIC_IMPULSE: "s",
    UnitCategory.AREA: "m2",
    UnitCategory.VOLUME: "m3",
    UnitCategory.ENERGY: "J",
    UnitCategory.POWER: "W",
    UnitCategory.HEAT_FLUX: "W/m2",
}

IMPERIAL_DEFAULTS = {
    UnitCategory.LENGTH: "ft",
    UnitCategory.MASS: "lb",
    UnitCategory.TIME: "s",
    UnitCategory.FORCE: "lbf",
    UnitCategory.PRESSURE: "psi",
    UnitCategory.TEMPERATURE: "R",
    UnitCategory.VELOCITY: "ft/s",
    UnitCategory.ACCELERATION: "ft/s2",
    UnitCategory.DENSITY: "lb/ft3",
    UnitCategory.MASS_FLOW: "lbm/s",
    UnitCategory.SPECIFIC_IMPULSE: "s",
    UnitCategory.AREA: "ft2",
    UnitCategory.VOLUME: "ft3",
    UnitCategory.ENERGY: "BTU",
    UnitCategory.POWER: "hp",
    UnitCategory.HEAT_FLUX: "BTU/h·ft2",
}


class UnitConverter:
    """
    Unit converter with configurable default system.

    Provides convenient methods for converting values between
    SI and Imperial units based on a selected system.
    """

    def __init__(self, system: UnitSystem = UnitSystem.SI):
        self.system = system
        self.defaults = SI_DEFAULTS if system == UnitSystem.SI else IMPERIAL_DEFAULTS

    def set_system(self, system: UnitSystem):
        """Change the default unit system."""
        self.system = system
        self.defaults = SI_DEFAULTS if system == UnitSystem.SI else IMPERIAL_DEFAULTS

    def display(self, value_si: float, category: UnitCategory, precision: int = 4) -> str:
        """
        Format a SI value for display in current unit system.

        Args:
            value_si: Value in SI units
            category: Unit category
            precision: Decimal places

        Returns:
            Formatted string with unit
        """
        target_unit = self.defaults.get(category)
        if target_unit is None:
            return f"{value_si:.{precision}g}"

        converted = from_si(value_si, target_unit)
        symbol = get_unit_symbol(target_unit)
        return f"{converted:.{precision}g} {symbol}"

    def input_to_si(self, value: float, category: UnitCategory) -> float:
        """Convert user input in current system to SI."""
        source_unit = self.defaults.get(category)
        if source_unit is None:
            return value
        return to_si(value, source_unit)

    def si_to_display(self, value_si: float, category: UnitCategory) -> float:
        """Convert SI value to current display system (numeric only)."""
        target_unit = self.defaults.get(category)
        if target_unit is None:
            return value_si
        return from_si(value_si, target_unit)

