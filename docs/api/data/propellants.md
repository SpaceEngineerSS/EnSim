# Propellant Presets

Comprehensive propellant combination database with validated performance data.

::: src.data.propellant_presets

## Overview

The propellant presets module provides 17+ pre-configured propellant combinations with:

- **Component Properties**: Density, boiling point, heat of formation
- **Performance Data**: Isp, C*, chamber temperature, gamma
- **Mixture Ratios**: Optimal and valid range
- **Application Notes**: Historical and current usage

## Quick Start

```python
from src.data.propellant_presets import (
    get_preset,
    get_presets_by_category,
    get_all_preset_names,
    PropellantCategory
)

# Get a specific preset
lox_ch4 = get_preset("LOX_LCH4")
print(f"Name: {lox_ch4.name}")
print(f"Vacuum Isp: {lox_ch4.isp_vacuum} s")
print(f"Optimal O/F: {lox_ch4.of_ratio_optimal}")
print(f"Applications: {', '.join(lox_ch4.applications)}")

# Get all cryogenic propellants
cryo_props = get_presets_by_category(PropellantCategory.CRYOGENIC)
for p in cryo_props:
    print(f"{p.name}: Isp = {p.isp_vacuum} s")
```

## Available Presets

### Cryogenic Propellants

| Name | Isp (vac) | O/F | Density | Notes |
|------|-----------|-----|---------|-------|
| LOX/LH2 | 455 s | 6.0 | 320 kg/m³ | Highest Isp |
| LOX/LCH4 | 363 s | 3.6 | 828 kg/m³ | Modern choice |
| LOX/LNG | 358 s | 3.4 | 810 kg/m³ | Cost-effective |

### Hydrocarbon Propellants

| Name | Isp (vac) | O/F | Density | Notes |
|------|-----------|-----|---------|-------|
| LOX/RP-1 | 338 s | 2.72 | 1030 kg/m³ | Workhorse |
| LOX/Ethanol | 308 s | 1.8 | 990 kg/m³ | Educational |
| LOX/Propane | 350 s | 3.2 | 860 kg/m³ | Alternative |

### Storable Propellants

| Name | Isp (vac) | O/F | Density | Notes |
|------|-----------|-----|---------|-------|
| N2O4/UDMH | 318 s | 2.6 | 1150 kg/m³ | Classic hypergolic |
| N2O4/MMH | 326 s | 2.2 | 1190 kg/m³ | Standard spacecraft |
| N2O4/N2H4 | 315 s | 1.3 | 1210 kg/m³ | High density |
| IRFNA/UDMH | 310 s | 3.0 | 1200 kg/m³ | Military |

### Green Propellants

| Name | Isp (vac) | O/F | Density | Notes |
|------|-----------|-----|---------|-------|
| LOX/75% Ethanol | 285 s | 1.4 | 950 kg/m³ | Low toxicity |
| N2O/HTPB | 250 s | 7.0 | 1080 kg/m³ | Hybrid |
| H2O2/RP-1 | 310 s | 7.5 | 1130 kg/m³ | Non-toxic storable |

### Monopropellants

| Name | Isp (vac) | Notes |
|------|-----------|-------|
| Hydrazine | 235 s | Standard RCS |
| HTP (90%) | 185 s | Green option |

### Exotic (Research Only)

| Name | Isp (vac) | Notes |
|------|-----------|-------|
| LF2/LH2 | 479 s | Maximum theoretical |
| ClF5/N2H4 | 350 s | Extreme hypergolic |

## Propellant Categories

```python
class PropellantCategory(Enum):
    CRYOGENIC = auto()       # LOX/LH2, LOX/LCH4
    HYDROCARBON = auto()     # LOX/RP-1, LOX/Ethanol
    STORABLE = auto()        # N2O4/UDMH, N2O4/MMH
    GREEN = auto()           # Non-toxic alternatives
    MONOPROPELLANT = auto()  # Hydrazine, HAN
    HYBRID = auto()          # LOX/HTPB, N2O/HTPB
    EXOTIC = auto()          # High-energy (F2, Be)
```

## Search Functions

### find_preset_by_isp()

Find presets within an Isp range:

```python
high_isp = find_preset_by_isp(min_isp=350, max_isp=500)
for p in high_isp:
    print(f"{p.name}: {p.isp_vacuum} s")
```

### find_preset_by_density()

Find high-density presets:

```python
dense = find_preset_by_density(min_density=1000)
for p in dense:
    print(f"{p.name}: {p.density_bulk} kg/m³")
```

### get_non_toxic_presets()

Find environmentally friendly options:

```python
green = get_non_toxic_presets()
for p in green:
    print(f"{p.name}: {p.fuel.toxicity.name} / {p.oxidizer.toxicity.name}")
```

## Toxicity Levels

```python
class ToxicityLevel(Enum):
    BENIGN = auto()      # Water, N2
    LOW = auto()         # Kerosene, ethanol
    MODERATE = auto()    # LOX, LH2
    HIGH = auto()        # MMH, N2H4
    EXTREME = auto()     # N2O4, F2, Be
```

## Data Sources

Performance data validated against:

- NASA CEA (Chemical Equilibrium with Applications)
- CPIA/M5 Liquid Propellant Manual
- Sutton & Biblarz, "Rocket Propulsion Elements"
- Historical engine specifications

