# Staging Module

Multi-stage rocket vehicle model with comprehensive staging logic.

::: src.core.staging

## Overview

The staging module provides support for multi-stage launch vehicle simulation, including:

- **Stage Definition**: Complete stage specifications with engines, masses, and propellant
- **Staging Logic**: Automatic staging based on propellant depletion, time, altitude, or velocity
- **Mass Tracking**: Real-time mass calculation including payload and fairings
- **Delta-V Analysis**: Ideal delta-v calculation using Tsiolkovsky equation

## Quick Start

```python
from src.core.staging import (
    Stage, StageEngine, MultiStageVehicle,
    StagingTrigger, create_falcon_9_like
)

# Create a Falcon 9-like vehicle
vehicle = create_falcon_9_like()
vehicle.payload_mass = 22_800  # kg to LEO

# Get total delta-v
total_dv = vehicle.get_total_delta_v()
print(f"Total ΔV: {total_dv:.0f} m/s")

# Get stage breakdown
for stage_name, dv in vehicle.get_stage_delta_v_breakdown():
    print(f"  {stage_name}: {dv:.0f} m/s")
```

## Key Classes

### StageEngine

Engine parameters for a stage:

| Parameter | Type | Description |
|-----------|------|-------------|
| `thrust_sl` | float | Sea-level thrust (N) |
| `thrust_vac` | float | Vacuum thrust (N) |
| `isp_sl` | float | Sea-level Isp (s) |
| `isp_vac` | float | Vacuum Isp (s) |
| `num_engines` | int | Number of engines |
| `throttle_min` | float | Minimum throttle (0-1) |
| `throttle_max` | float | Maximum throttle (0-1) |

### Stage

Complete stage definition:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Stage identifier |
| `dry_mass` | float | Dry mass without propellant (kg) |
| `propellant_mass` | float | Total propellant mass (kg) |
| `engine` | StageEngine | Engine parameters |
| `staging_trigger` | StagingTrigger | What triggers separation |
| `fairing_mass` | float | Payload fairing mass (kg) |
| `fairing_jettison_alt` | float | Altitude to jettison fairing (m) |

### MultiStageVehicle

Complete vehicle model:

```python
vehicle = MultiStageVehicle(
    name="My Rocket",
    stages=[stage1, stage2],
    payload_mass=1000.0,
    launch_site_altitude=0.0
)
```

## Staging Triggers

Available triggers for stage separation:

- `PROPELLANT_DEPLETION`: When propellant runs out (default)
- `TIME_BASED`: At specific mission time
- `ALTITUDE_BASED`: At specific altitude
- `VELOCITY_BASED`: At specific velocity
- `MANUAL`: Manual trigger

## Factory Functions

### create_falcon_9_like()

Creates a Falcon 9 Full Thrust-like two-stage vehicle.

### create_saturn_v_like()

Creates a Saturn V-like three-stage vehicle.

### create_custom_vehicle()

Creates a custom vehicle from configuration dictionaries:

```python
config = [
    {
        'name': 'Stage 1',
        'dry_mass': 5000,
        'propellant_mass': 50000,
        'thrust_sl': 500000,
        'thrust_vac': 550000,
        'isp_sl': 280,
        'isp_vac': 310
    },
    {
        'name': 'Stage 2',
        'dry_mass': 1000,
        'propellant_mass': 10000,
        'thrust_vac': 100000,
        'isp_vac': 340
    }
]

vehicle = create_custom_vehicle(config, payload_mass=500)
```

## Simulation Integration

The `step()` method advances the vehicle state:

```python
# Simulation loop
dt = 0.1  # timestep
while vehicle.remaining_stages > 0:
    events = vehicle.step(
        dt=dt,
        altitude=altitude,
        velocity=velocity,
        throttle=1.0
    )
    
    for event in events:
        print(f"Event: {event.event_type} at t={event.time:.1f}s")
```

