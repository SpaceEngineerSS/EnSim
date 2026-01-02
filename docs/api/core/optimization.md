# Optimization Module

Trajectory and engine design optimization algorithms.

::: src.core.optimization

## Overview

The optimization module provides algorithms for:

- **Gravity Turn Optimization**: Find optimal pitch program parameters
- **Nozzle Design**: Optimize expansion ratio for mission profile
- **Stage Mass Allocation**: Optimal propellant distribution between stages
- **Engine Design**: Optimize chamber conditions for target performance
- **Propellant Load**: Determine minimum propellant for mission requirements

## Quick Start

```python
from src.core.optimization import (
    optimize_gravity_turn,
    optimize_nozzle_expansion_ratio,
    optimize_stage_mass_allocation,
    optimize_engine_parameters,
    TrajectoryConstraints
)

# Optimize gravity turn trajectory
result = optimize_gravity_turn(
    vehicle_mass=50000,
    thrust=500000,
    isp=310,
    propellant_mass=45000
)

print(f"Optimal kickoff altitude: {result.optimal_params['kickoff_altitude']:.0f} m")
print(f"Optimal kickoff angle: {result.optimal_params['kickoff_angle']:.1f}°")
```

## Functions

### optimize_gravity_turn()

Finds optimal gravity turn parameters to maximize payload to orbit.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vehicle_mass` | float | Initial vehicle mass (kg) |
| `thrust` | float | Engine thrust (N) |
| `isp` | float | Specific impulse (s) |
| `propellant_mass` | float | Available propellant (kg) |
| `constraints` | TrajectoryConstraints | Trajectory constraints |
| `method` | str | Optimization method |

**Returns:** `OptimizationResult` with optimal parameters.

### optimize_nozzle_expansion_ratio()

Optimizes nozzle expansion ratio for mission profile.

```python
result = optimize_nozzle_expansion_ratio(
    chamber_pressure=7e6,       # 70 bar
    ambient_pressure=101325,    # 1 atm
    gamma=1.2,
    weight_vacuum=0.7,          # 70% vacuum operation
    weight_sealevel=0.3         # 30% sea-level
)

print(f"Optimal ε: {result.optimal_params['area_ratio']:.1f}")
print(f"Exit Mach: {result.optimal_params['exit_mach']:.2f}")
```

### optimize_stage_mass_allocation()

Finds optimal propellant distribution between stages.

```python
result = optimize_stage_mass_allocation(
    total_propellant=100000,    # kg
    num_stages=2,
    payload_mass=5000,          # kg
    stage_isps=[310, 340],      # s
    structural_coefficients=[0.1, 0.08]
)

print(f"Total ΔV: {result.optimal_params['total_delta_v']:.0f} m/s")
for i, mass in enumerate(result.optimal_params['propellant_masses']):
    print(f"Stage {i+1}: {mass:.0f} kg")
```

### optimize_engine_parameters()

Optimizes engine chamber conditions for target performance.

```python
result = optimize_engine_parameters(
    target_thrust=1e6,          # 1 MN
    target_isp=350,             # s
    propellant_type="LOX/CH4",
    chamber_pressure_range=(5e6, 25e6),
    mixture_ratio_range=(2.5, 4.0)
)

print(f"Chamber pressure: {result.optimal_params['chamber_pressure']/1e6:.1f} MPa")
print(f"O/F ratio: {result.optimal_params['mixture_ratio']:.2f}")
```

### optimize_propellant_load()

Determines minimum propellant load for mission requirements.

```python
result = optimize_propellant_load(
    dry_mass=5000,              # kg
    tank_volume=100,            # m³
    propellant_density=1000,    # kg/m³
    target_delta_v=3000,        # m/s
    isp=320,                    # s
    payload_mass=1000           # kg
)

if result.success:
    print(f"Required propellant: {result.optimal_params['propellant_mass']:.0f} kg")
    print(f"Tank utilization: {result.optimal_params['tank_utilization']*100:.1f}%")
```

## Constraints

The `TrajectoryConstraints` class defines trajectory limits:

```python
constraints = TrajectoryConstraints(
    max_dynamic_pressure=35000,     # Pa (~35 kPa max Q)
    max_acceleration=6.0,           # g's
    min_altitude=0.0,               # m
    target_altitude=200_000,        # m (orbit)
    target_velocity=7800,           # m/s
    target_flight_path_angle=0.0    # rad (horizontal at insertion)
)
```

## Optimization Methods

Available methods for `optimize_gravity_turn()`:

- `"SLSQP"`: Sequential Least Squares Programming (default)
- `"L-BFGS-B"`: Limited-memory BFGS with bounds
- `"differential_evolution"`: Global evolutionary optimization

