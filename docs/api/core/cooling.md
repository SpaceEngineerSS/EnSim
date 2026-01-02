# Cooling Module

Regenerative cooling thermal analysis for rocket engines.

::: src.core.cooling

## Overview

The cooling module provides detailed thermal modeling for rocket engine cooling:

- **Heat Transfer Correlations**: Bartz (gas-side), Dittus-Boelter (coolant-side)
- **Channel Design**: Sizing of regenerative cooling channels
- **Wall Temperature**: Prediction of gas-side and coolant-side temperatures
- **Thermal Margins**: Safety margins to melting and boiling

## Quick Start

```python
from src.core.cooling import (
    CoolantType,
    design_cooling_channels,
    analyze_cooling_system
)

# Design cooling channels for an engine
design = design_cooling_channels(
    thrust=1e6,                 # 1 MN thrust
    chamber_pressure=7e6,       # 70 bar
    chamber_temp=3500,          # K
    coolant=CoolantType.RP1,
    wall_material="Inconel 718"
)

print(f"Number of channels: {design.channels.num_channels}")
print(f"Channel width: {design.channels.width*1000:.2f} mm")
print(f"Coolant mass flow: {design.coolant_mass_flow:.1f} kg/s")
```

## Key Classes

### CoolantType

Available coolant types:

| Coolant | Description |
|---------|-------------|
| `RP1` | RP-1 Kerosene |
| `LH2` | Liquid Hydrogen |
| `LOX` | Liquid Oxygen |
| `LCH4` | Liquid Methane |
| `WATER` | Water (for testing) |

### CoolingChannel

Channel geometry definition:

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | float | Channel width (m) |
| `height` | float | Channel height (m) |
| `wall_thickness` | float | Inner wall thickness (m) |
| `land_width` | float | Width between channels (m) |
| `num_channels` | int | Number of channels |
| `length` | float | Total channel length (m) |

### CoolingSystemDesign

Complete system specification:

| Parameter | Type | Description |
|-----------|------|-------------|
| `channels` | CoolingChannel | Channel geometry |
| `coolant` | CoolantType | Coolant type |
| `coolant_inlet_temp` | float | Inlet temperature (K) |
| `coolant_inlet_pressure` | float | Inlet pressure (Pa) |
| `coolant_mass_flow` | float | Mass flow rate (kg/s) |
| `wall_material` | str | Wall material name |
| `wall_thermal_conductivity` | float | k (W/(m·K)) |
| `wall_melting_point` | float | Melting point (K) |

## Heat Transfer

### Bartz Correlation

Gas-side heat transfer coefficient:

```python
from src.core.cooling import bartz_heat_transfer_coefficient

h_g = bartz_heat_transfer_coefficient(
    D_throat=0.1,           # Throat diameter (m)
    P_chamber=7e6,          # Chamber pressure (Pa)
    c_star=1800,            # Characteristic velocity (m/s)
    T_chamber=3500,         # Chamber temperature (K)
    gamma=1.2,              # Specific heat ratio
    Pr=0.7,                 # Prandtl number
    mu_ref=5e-5,            # Reference viscosity (Pa·s)
    area_ratio=10,          # Local A/At
    local_diameter=0.3      # Local diameter (m)
)
```

### Dittus-Boelter Correlation

Coolant-side heat transfer coefficient:

```python
from src.core.cooling import dittus_boelter_coefficient

h_c = dittus_boelter_coefficient(
    Re=1e6,                 # Reynolds number
    Pr=5.0,                 # Prandtl number
    k=0.13,                 # Thermal conductivity (W/(m·K))
    D_h=0.003,              # Hydraulic diameter (m)
    heating=True            # Fluid is being heated
)
```

## Thermal Analysis

### analyze_cooling_system()

Perform thermal analysis along the nozzle:

```python
# Define nozzle profile as (x, diameter) points
nozzle_profile = [
    (0.0, 0.3),      # Chamber
    (0.2, 0.1),      # Throat
    (0.8, 0.5)       # Exit
]

# Chamber conditions
chamber = {
    'T_chamber': 3500,
    'P_chamber': 7e6,
    'gamma': 1.2,
    'c_star': 1800
}

# Run analysis
results = analyze_cooling_system(
    design=design,
    nozzle_profile=nozzle_profile,
    chamber_conditions=chamber,
    num_stations=50
)

# Check results at throat (station ~10)
throat = results[10]
print(f"Throat gas-side wall: {throat.wall_temp_gas_side:.0f} K")
print(f"Heat flux: {throat.heat_flux/1e6:.1f} MW/m²")
print(f"Margin to melting: {throat.margin_to_melting:.0f} K")
```

## Wall Materials

Available wall materials:

| Material | k (W/(m·K)) | T_melt (K) |
|----------|-------------|------------|
| Inconel 718 | 11.4 | 1533 |
| OFHC Copper | 385.0 | 1356 |
| GRCop-84 | 300.0 | 1356 |
| Haynes 230 | 8.9 | 1573 |
| Monel 400 | 21.8 | 1573 |

## Design Guidelines

Typical regenerative cooling design parameters:

- **Channel velocity**: 15-30 m/s for good heat transfer
- **Wall thickness**: 1-2 mm for thermal and structural requirements
- **Channel aspect ratio**: height/width ≈ 2-4
- **Number of channels**: ~100-200 for medium engines
- **Coolant ΔT**: Keep below saturation to avoid boiling

