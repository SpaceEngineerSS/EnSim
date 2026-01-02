# API Reference

This section provides detailed documentation for EnSim's Python API.

## Core Modules

The `src.core` package contains the physics engine:

| Module | Description |
|--------|-------------|
| [`chemistry`](core/chemistry.md) | Chemical equilibrium solver (Gordon-McBride) |
| [`propulsion`](core/propulsion.md) | Nozzle flow and performance calculations |
| [`thermodynamics`](core/thermodynamics.md) | NASA polynomial property evaluation |
| [`flight_6dof`](core/flight_6dof.md) | 6-DOF rigid body dynamics |
| [`monte_carlo`](core/monte_carlo.md) | Statistical dispersion analysis |

## Utility Modules

The `src.utils` package provides support functions:

| Module | Description |
|--------|-------------|
| [`nasa_parser`](utils/nasa_parser.md) | NASA thermodynamic data parser |

## Quick Reference

### Combustion Analysis

```python
from src.core.chemistry import CombustionProblem
from src.utils.nasa_parser import create_sample_database

# Create solver
db = create_sample_database()
problem = CombustionProblem(db)

# Add reactants
problem.add_fuel('H2', moles=2.0)
problem.add_oxidizer('O2', moles=1.0)

# Solve equilibrium
result = problem.solve(pressure=10e6)
```

### Performance Calculation

```python
from src.core.propulsion import calculate_c_star, calculate_thrust_coefficient

# Characteristic velocity
c_star = calculate_c_star(
    T_chamber=3500,  # K
    gamma=1.15,
    M_mol=0.018  # kg/mol
)

# Thrust coefficient
cf = calculate_thrust_coefficient(
    gamma=1.15,
    expansion_ratio=50,
    Pc=10e6,  # Pa
    Pa=0  # Vacuum
)
```

### Flight Simulation

```python
from src.core.flight_6dof import simulate_flight_6dof
from src.core.rocket import Rocket

# Configure rocket
rocket = Rocket(...)

# Run simulation
result = simulate_flight_6dof(
    rocket=rocket,
    dt=0.01,
    max_time=300
)
```

## Type Definitions

Key data types are defined in `src.core.types`:

- `SpeciesData` - Thermodynamic species data
- `EquilibriumResult` - Combustion equilibrium results
- `Reactant` - Fuel/oxidizer specification

## Constants

Physical constants are in `src.core.constants`:

```python
from src.core.constants import GAS_CONSTANT, G0

print(f"R = {GAS_CONSTANT} J/(mol·K)")
print(f"g₀ = {G0} m/s²")
```

