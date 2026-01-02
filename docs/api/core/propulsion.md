# propulsion

Nozzle flow physics and rocket performance calculations.

## Overview

The `propulsion` module provides functions for calculating rocket engine 
performance using 1-D isentropic compressible flow theory.

## Key Functions

### calculate_c_star

Calculate characteristic velocity (C*).

```python
from src.core.propulsion import calculate_c_star

c_star = calculate_c_star(
    T_chamber=3500,    # Chamber temperature (K)
    gamma=1.15,        # Specific heat ratio
    M_mol=0.018        # Molecular weight (kg/mol)
)
print(f"C* = {c_star:.1f} m/s")
```

**Formula:**

$$
C^* = \frac{\sqrt{\gamma R T_c}}{\Gamma(\gamma)}
$$

where the gamma function is:

$$
\Gamma(\gamma) = \sqrt{\gamma \left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}}}
$$

### calculate_exit_velocity

Calculate nozzle exit velocity.

```python
from src.core.propulsion import calculate_exit_velocity

Ve = calculate_exit_velocity(
    T_chamber=3500,    # K
    gamma=1.15,
    M_mol=0.018,       # kg/mol
    pressure_ratio=0.01  # Pe/Pc
)
print(f"Ve = {Ve:.1f} m/s")
```

**Formula:**

$$
V_e = \sqrt{\frac{2\gamma}{\gamma-1} \frac{R T_c}{M} \left[1 - \left(\frac{P_e}{P_c}\right)^{\frac{\gamma-1}{\gamma}}\right]}
$$

### calculate_thrust_coefficient

Calculate thrust coefficient (Cf).

```python
from src.core.propulsion import calculate_thrust_coefficient

# Vacuum conditions
cf_vac = calculate_thrust_coefficient(
    gamma=1.15,
    expansion_ratio=50,
    Pc=10e6,        # Pa
    Pa=0            # Vacuum
)

# Sea level
cf_sl = calculate_thrust_coefficient(
    gamma=1.15,
    expansion_ratio=50,
    Pc=10e6,
    Pa=101325       # 1 atm
)
```

**Formula:**

$$
C_F = \sqrt{\frac{2\gamma^2}{\gamma-1}\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}} \left[1-\left(\frac{P_e}{P_c}\right)^{\frac{\gamma-1}{\gamma}}\right]} + \varepsilon \frac{P_e - P_a}{P_c}
$$

### solve_mach_from_area_ratio_supersonic

Solve for Mach number from area ratio (supersonic branch).

```python
from src.core.propulsion import solve_mach_from_area_ratio_supersonic

M_exit = solve_mach_from_area_ratio_supersonic(
    area_ratio=50,
    gamma=1.15
)
print(f"M_exit = {M_exit:.2f}")
```

Uses Newton-Raphson iteration on the area-Mach relation:

$$
\frac{A}{A^*} = \frac{1}{M}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}
$$

### calculate_ideal_expansion_ratio

Calculate the ideal expansion ratio for a given exit pressure.

```python
from src.core.propulsion import calculate_ideal_expansion_ratio

epsilon = calculate_ideal_expansion_ratio(
    gamma=1.15,
    Pc=10e6,
    Pe=101325
)
print(f"Optimal ε = {epsilon:.1f}")
```

### check_flow_separation

Check for flow separation using Summerfield criterion.

```python
from src.core.propulsion import check_flow_separation

separated, P_sep = check_flow_separation(
    Pc=1e6,
    Pa=101325,
    Pe=5000,
    expansion_ratio=50
)

if separated:
    print(f"Flow separates at P = {P_sep/1000:.1f} kPa")
```

### get_nozzle_profile

Generate nozzle contour coordinates for visualization.

```python
from src.core.propulsion import get_nozzle_profile

x, r = get_nozzle_profile(
    expansion_ratio=50,
    throat_radius=0.05,  # m
    n_points=100
)

import matplotlib.pyplot as plt
plt.plot(x, r, 'b-')
plt.plot(x, -r, 'b-')
plt.axis('equal')
plt.show()
```

## Performance Calculator

### calculate_performance

Complete engine performance calculation.

```python
from src.core.propulsion import calculate_performance

result = calculate_performance(
    T_chamber=3500,       # K
    gamma=1.15,
    M_mol=18.0,           # g/mol
    Pc=10e6,              # Pa
    expansion_ratio=50,
    Pa=0,                 # Vacuum
    eta_c_star=0.97,      # Combustion efficiency
    eta_cf=0.98,          # Nozzle efficiency
    divergence_half_angle=15  # degrees
)

print(f"C* = {result.c_star:.1f} m/s")
print(f"Ve = {result.exit_velocity:.1f} m/s")
print(f"Cf = {result.thrust_coefficient:.3f}")
print(f"Isp_vac = {result.isp_vacuum:.1f} s")
```

#### PerformanceResult

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| `c_star` | `float` | m/s | Characteristic velocity |
| `exit_velocity` | `float` | m/s | Nozzle exit velocity |
| `thrust_coefficient` | `float` | - | Thrust coefficient |
| `isp_vacuum` | `float` | s | Vacuum specific impulse |
| `isp_sea_level` | `float` | s | Sea-level specific impulse |
| `exit_mach` | `float` | - | Exit Mach number |
| `exit_pressure` | `float` | Pa | Exit pressure |
| `exit_temperature` | `float` | K | Exit temperature |

## Efficiency Factors

### Combustion Efficiency (η_c*)

Accounts for incomplete combustion:

$$
C^*_{actual} = \eta_{C^*} \cdot C^*_{ideal}
$$

Typical values: 0.95-0.99

### Nozzle Efficiency (η_Cf)

Accounts for boundary layer losses:

$$
C_{F,actual} = \eta_{C_F} \cdot C_{F,ideal}
$$

Typical values: 0.95-0.99

### Divergence Loss

For conical nozzle with half-angle α:

$$
\lambda = \frac{1 + \cos\alpha}{2}
$$

## Example: Complete Engine Analysis

```python
from src.core.chemistry import CombustionProblem
from src.core.propulsion import calculate_performance
from src.utils.nasa_parser import create_sample_database

# Step 1: Combustion analysis
db = create_sample_database()
combustion = CombustionProblem(db)
combustion.add_fuel('CH4', moles=1.0)
combustion.add_oxidizer('O2', moles=2.0)
eq = combustion.solve(pressure=10e6)

# Step 2: Performance calculation
perf = calculate_performance(
    T_chamber=eq.temperature,
    gamma=eq.gamma,
    M_mol=eq.mean_molecular_weight,
    Pc=10e6,
    expansion_ratio=40,
    Pa=0,
    eta_c_star=0.97,
    eta_cf=0.98
)

print(f"=== LOX/Methane Engine ===")
print(f"T_chamber: {eq.temperature:.0f} K")
print(f"Gamma: {eq.gamma:.3f}")
print(f"C*: {perf.c_star:.1f} m/s")
print(f"Isp_vac: {perf.isp_vacuum:.1f} s")
```

## References

1. Sutton, G.P. & Biblarz, O. (2017). *Rocket Propulsion Elements*, 9th Ed.
2. Anderson, J.D. (2003). *Modern Compressible Flow*, 3rd Ed.

