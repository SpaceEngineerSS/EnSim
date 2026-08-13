# Python API

The installed namespace is `ensim`. Core numerical modules do not require a Qt
event loop.

```python
from ensim.core.chemistry import CombustionProblem
from ensim.core.propulsion import NozzleConditions, calculate_performance
from ensim.utils.nasa_parser import load_default_database
```

## Principal modules

| Module | Responsibility |
|---|---|
| `ensim.core.chemistry` | equilibrium and adiabatic chamber solution |
| `ensim.core.thermodynamics` | NASA polynomial properties |
| `ensim.core.propulsion` | ideal nozzle relations and station profiles |
| `ensim.core.cooling` | preliminary regenerative-cooling correlations |
| `ensim.core.flight_6dof` | rigid-body flight propagation |
| `ensim.core.geodesy` | WGS-84, ECEF/ECI transforms and J2 gravity |
| `ensim.core.monte_carlo` | flight dispersion |
| `ensim.core.engine_uq` | reduced-order engine UQ |
| `ensim.core.optimization` | reduced-order engineering objectives |

## Unit contract

Unless a parameter name says otherwise, the API uses SI. The established
thermochemistry boundary reports mean molecular weight in g/mol. Read each
dataclass/function docstring and validate domains before composing models.

See [chemistry](core/chemistry.md) and [propulsion](core/propulsion.md) for short
examples. Public code should not import `src.core` or manipulate the source-tree
layout.
