# chemistry

Chemical equilibrium solver using Gibbs free energy minimization.

## Overview

The `chemistry` module implements the Gordon-McBride method for calculating 
chemical equilibrium compositions in rocket combustion chambers. It uses
NASA 7-term polynomials for thermodynamic properties.

## Main Classes

### CombustionProblem

The primary interface for combustion equilibrium calculations.

```python
from src.core.chemistry import CombustionProblem
from src.utils.nasa_parser import create_sample_database

# Initialize with species database
db = create_sample_database()
problem = CombustionProblem(db)

# Add reactants
problem.add_fuel('H2', moles=2.0, temperature=298.15)
problem.add_oxidizer('O2', moles=1.0, temperature=298.15)

# Solve for adiabatic equilibrium
result = problem.solve(
    pressure=10e6,           # Chamber pressure (Pa)
    initial_temp_guess=3000, # Initial temperature guess (K)
    max_iterations=50,       # Maximum iterations
    tolerance=1e-5           # Convergence tolerance
)
```

#### Methods

##### `add_fuel(species_name, moles, temperature)`

Add a fuel species to the reactants.

| Parameter | Type | Description |
|-----------|------|-------------|
| `species_name` | `str` | Species name (must exist in database) |
| `moles` | `float` | Number of moles |
| `temperature` | `float` | Initial temperature (K), default 298.15 |

##### `add_oxidizer(species_name, moles, temperature)`

Add an oxidizer species to the reactants.

Parameters same as `add_fuel`.

##### `solve(pressure, initial_temp_guess, max_iterations, tolerance)`

Solve for equilibrium composition.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pressure` | `float` | 101325.0 | Chamber pressure (Pa) |
| `initial_temp_guess` | `float` | 3000.0 | Starting temperature (K) |
| `max_iterations` | `int` | 50 | Maximum iterations |
| `tolerance` | `float` | 1e-5 | Convergence tolerance |

**Returns**: `EquilibriumResult` object

## Core Functions

### nasa_get_cp_r

```python
@jit(nopython=True, cache=True)
def nasa_get_cp_r(T, coeffs_low, coeffs_high, T_mid) -> float
```

Calculate dimensionless heat capacity Cp/R.

**Formula:**

$$
\frac{C_p}{R} = a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4
$$

### nasa_get_h_rt

```python
@jit(nopython=True, cache=True)
def nasa_get_h_rt(T, coeffs_low, coeffs_high, T_mid) -> float
```

Calculate dimensionless enthalpy H/(RT).

**Formula:**

$$
\frac{H}{RT} = a_1 + \frac{a_2}{2}T + \frac{a_3}{3}T^2 + \frac{a_4}{4}T^3 + \frac{a_5}{5}T^4 + \frac{a_6}{T}
$$

### nasa_get_s_r

```python
@jit(nopython=True, cache=True)  
def nasa_get_s_r(T, coeffs_low, coeffs_high, T_mid) -> float
```

Calculate dimensionless entropy S/R.

**Formula:**

$$
\frac{S}{R} = a_1 \ln T + a_2 T + \frac{a_3}{2}T^2 + \frac{a_4}{3}T^3 + \frac{a_5}{4}T^4 + a_7
$$

### solve_equilibrium_gordon_mcbride

```python
@jit(nopython=True, cache=True)
def solve_equilibrium_gordon_mcbride(
    T: float,
    P_atm: float,
    a_ij: np.ndarray,
    b_i: np.ndarray,
    g_rt: np.ndarray,
    max_iter: int,
    tol: float
) -> Tuple[np.ndarray, bool]
```

Solve equilibrium using Gordon-McBride iteration (NASA RP-1311 method).

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` | `float` | Temperature (K) |
| `P_atm` | `float` | Pressure in atmospheres |
| `a_ij` | `ndarray` | Stoichiometry matrix (n_elem × n_spec) |
| `b_i` | `ndarray` | Element totals |
| `g_rt` | `ndarray` | Dimensionless Gibbs energy |
| `max_iter` | `int` | Maximum iterations |
| `tol` | `float` | Convergence tolerance |

**Returns**: Tuple of (mole numbers array, convergence flag)

## Data Types

### EquilibriumResult

Result from equilibrium calculation.

| Attribute | Type | Description |
|-----------|------|-------------|
| `temperature` | `float` | Adiabatic flame temperature (K) |
| `pressure` | `float` | System pressure (Pa) |
| `species_names` | `List[str]` | Species in order |
| `mole_fractions` | `ndarray` | Mole fractions |
| `moles` | `ndarray` | Mole numbers |
| `mean_molecular_weight` | `float` | Mixture MW (g/mol) |
| `gamma` | `float` | Specific heat ratio (Cp/Cv) |
| `converged` | `bool` | Whether solver converged |

#### Methods

##### `get_mole_fraction(species_name)`

Get mole fraction for a specific species.

```python
x_h2o = result.get_mole_fraction('H2O')
```

## Example Usage

### LOX/LH2 Analysis

```python
from src.core.chemistry import CombustionProblem
from src.utils.nasa_parser import create_sample_database

db = create_sample_database()
problem = CombustionProblem(db)

# Stoichiometric H2/O2
problem.add_fuel('H2', moles=2.0)
problem.add_oxidizer('O2', moles=1.0)

# High-pressure combustion
result = problem.solve(pressure=20e6)  # 200 bar

print(f"T_chamber = {result.temperature:.0f} K")
print(f"γ = {result.gamma:.3f}")
print(f"M̄ = {result.mean_molecular_weight:.2f} g/mol")

# Check dissociation
print(f"\nSpecies composition:")
for name in ['H2O', 'OH', 'H', 'O', 'H2', 'O2']:
    x = result.get_mole_fraction(name)
    if x > 1e-4:
        print(f"  {name}: {x:.4f}")
```

## References

1. Gordon, S. & McBride, B.J. (1994). NASA RP-1311
2. McBride, B.J., Zehe, M.J., & Gordon, S. (2002). NASA/TP-2002-211556

