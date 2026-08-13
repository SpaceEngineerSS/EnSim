# Chemical equilibrium API

```python
from ensim.core.chemistry import CombustionProblem
from ensim.utils.nasa_parser import load_default_database

database = load_default_database()
problem = CombustionProblem(database)
problem.add_fuel("H2", moles=2.0, temperature=298.15)
problem.add_oxidizer("O2", moles=1.0, temperature=298.15)
result = problem.solve(
    pressure=6.89e6,
    initial_temp_guess=3500.0,
    max_iterations=100,
    tolerance=1e-6,
)

if not result.converged:
    raise RuntimeError("equilibrium did not converge")
```

`CombustionProblem` enforces species lookup and builds elemental constraints from
the supplied reactants. `solve` returns temperature, product moles/mole fractions,
mixture molar mass, frozen `Cp/Cv`, iteration count and convergence state.

Pressures are Pa, temperatures K and reactant amounts mol. The result's
`mean_molecular_weight` is g/mol. The available species define the equilibrium
problem; use `load_default_database()` for the same packaged data as the GUI.

See [theory](../../THEORY.md) and [validation](../../VALIDATION.md).
