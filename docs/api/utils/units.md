# Units Module

Comprehensive unit conversion system for rocket propulsion.

::: src.utils.units

## Overview

The units module provides complete SI/Imperial unit conversion with:

- **17 Unit Categories**: Length, mass, force, pressure, temperature, etc.
- **100+ Unit Definitions**: Comprehensive coverage of engineering units
- **Temperature Handling**: Proper offset conversions (K, °C, °F, °R)
- **Unit System Presets**: SI and Imperial default configurations

## Quick Start

```python
from src.utils.units import convert, to_si, from_si, UnitConverter, UnitSystem

# Simple conversion
thrust_lbf = 100000  # lbf
thrust_N = convert(thrust_lbf, "lbf", "N")
print(f"{thrust_lbf} lbf = {thrust_N:.0f} N")

# Temperature conversion
temp_F = 6000  # °F
temp_K = convert(temp_F, "F", "K")
print(f"{temp_F}°F = {temp_K:.0f} K")

# Using UnitConverter for display
converter = UnitConverter(UnitSystem.IMPERIAL)
from src.utils.units import UnitCategory
print(converter.display(7e6, UnitCategory.PRESSURE))  # "1015.3 psi"
```

## Unit Categories

| Category | SI Base | Imperial |
|----------|---------|----------|
| LENGTH | m | ft |
| MASS | kg | lb |
| TIME | s | s |
| FORCE | N | lbf |
| PRESSURE | Pa | psi |
| TEMPERATURE | K | °R |
| VELOCITY | m/s | ft/s |
| ACCELERATION | m/s² | ft/s² |
| DENSITY | kg/m³ | lb/ft³ |
| MASS_FLOW | kg/s | lbm/s |
| SPECIFIC_IMPULSE | s | s |
| AREA | m² | ft² |
| VOLUME | m³ | ft³ |
| ENERGY | J | BTU |
| POWER | W | hp |
| HEAT_FLUX | W/m² | BTU/(h·ft²) |

## Conversion Functions

### convert()

Convert between any compatible units:

```python
# Pressure conversion
pressure_mpa = convert(1000, "psi", "MPa")

# Force conversion
force_kn = convert(225000, "lbf", "kN")

# Velocity conversion
velocity_ms = convert(25000, "ft/s", "m/s")
```

### to_si()

Convert any unit to SI base:

```python
pressure_pa = to_si(1000, "psi")      # → Pa
temperature_k = to_si(6000, "F")       # → K
mass_kg = to_si(10000, "lb")           # → kg
```

### from_si()

Convert from SI to any unit:

```python
pressure_psi = from_si(6.89e6, "psi")  # Pa → psi
temperature_f = from_si(3500, "F")      # K → °F
mass_lb = from_si(1000, "lb")           # kg → lb
```

## Temperature Conversions

Temperature has offset conversions handled properly:

```python
# All equivalent temperatures
temp_k = 3500                           # Kelvin
temp_c = convert(temp_k, "K", "C")      # 3226.85 °C
temp_f = convert(temp_k, "K", "F")      # 5840.33 °F
temp_r = convert(temp_k, "K", "R")      # 6300 °R
```

## UnitConverter Class

For UI integration with system selection:

```python
from src.utils.units import UnitConverter, UnitSystem, UnitCategory

# Create converter for Imperial units
conv = UnitConverter(UnitSystem.IMPERIAL)

# Display values in current system
print(conv.display(7e6, UnitCategory.PRESSURE))     # "1015.3 psi"
print(conv.display(3500, UnitCategory.TEMPERATURE)) # "6300 °R"
print(conv.display(450, UnitCategory.VELOCITY))     # "1476.4 ft/s"

# Convert user input to SI
user_thrust = 225000  # lbf entered by user
thrust_si = conv.input_to_si(user_thrust, UnitCategory.FORCE)  # → N
```

## Common Unit Symbols

### Pressure
- `Pa`, `kPa`, `MPa`, `bar`, `atm`
- `psi`, `psia`, `psig`
- `torr`, `mmHg`, `inHg`

### Force
- `N`, `kN`, `MN`
- `lbf`, `klbf`
- `kgf`, `dyn`

### Mass Flow
- `kg/s`, `g/s`, `kg/h`
- `lb/s`, `lbm/s`, `slug/s`

### Energy
- `J`, `kJ`, `MJ`
- `cal`, `kcal`
- `BTU`, `ft·lbf`

## Extending Units

Add custom units:

```python
from src.utils.units import UnitDefinition, UnitCategory, ALL_UNITS

# Add custom unit
ALL_UNITS["my_unit"] = UnitDefinition(
    name="My Unit",
    symbol="mu",
    category=UnitCategory.LENGTH,
    to_si=1.234,        # my_unit → m
    from_si=1/1.234     # m → my_unit
)
```

## Best Practices

1. **Store in SI**: All internal calculations use SI units
2. **Convert at boundaries**: Convert input from user, convert output for display
3. **Use UnitConverter**: For consistent UI unit system handling
4. **Check categories**: `convert()` raises error for incompatible units

