# Moving to 4.0

Most users only need the first section. The rest is for code that imports
EnSim.

## Installing

The desktop application is now an extra:

```console
$ pip install --upgrade "ensim[gui]"
```

Plain `pip install ensim` gives the library and the command line without Qt,
which is what servers, notebooks and CI usually want.

## Your projects

3.x project files open without changes and are saved as format 4.0 the next
time you save. 1.x files open too.

Expect different numbers. Two changes move almost every result:

- **Propellants are liquids now.** 3.x burned every propellant as a gas at
  298 K. Cryogenic propellants carry less energy than that: LH2/LOX at O/F 6
  loses about 100 K of chamber temperature and 2 % of Isp. Names are migrated
  on load (`H2` → `LH2`, `O2` → `LOX`, `CH4` → `LCH4`, `RP1` → `RP-1`,
  `N2O4` → `NTO`, `H2O2` → `HTP`).
- **The nozzle keeps $c_p(T)$.** Constant-$\gamma$ expansion overstated vacuum
  Isp by about 1 %.

Flights change for different reasons: the atmosphere was too thin at altitude
(5 % low in pressure at 50 km),
forces were frozen across each integration step, and the default nozzle exit
area of 0.01 m² took away up to 1 kN of thrust at sea level. Small rockets now
fly higher; staged vehicles carry the thrust they were meant to.

## Code

### Engine analysis

3.x code assembled the chamber problem and the nozzle relations by hand. In 4.0
one call does it:

```python
from ensim.core.engine import EngineInputs, analyze_engine

result = analyze_engine(EngineInputs("LH2", "LOX", 6.0, 70e5, 60, throat_area=0.01))
result.vacuum.isp, result.sea_level, result.performance_at(20e3)
```

`CombustionProblem` and `ensim.core.propulsion` are still there for lower-level
work. `CombustionProblem.solve` no longer falls back to the old solver; check
`converged` on its result, or let `analyze_engine` raise `ConvergenceError`.

### Renamed or moved

| 3.x | 4.0 |
|---|---|
| `StageEngine(thrust_sl=…, thrust_vac=…)` | `StageEngine(thrust_sl_per_engine=…, thrust_vac_per_engine=…)` |
| `chemistry.parse_formula` | `ensim.core.formula.parse_formula` |
| `flight_6dof.SimulationCancelled` (new) | also `ensim.core.types.SimulationCancelled` |
| `units.convert(value, "kN", "lbf")` | `units.convert(value, Quantity.FORCE, "kN", "lbf")` |
| `units.UnitConverter`, `UnitCategory` | `units.Quantity`, `to_si`, `from_si`, `display_unit` |
| `monte_carlo.plot_dispersion`, `print_dispersion_summary` | removed; plot `DispersionResult` yourself |
| `mission.get_atmosphere_table` | `mission.atmosphere_properties(z)` per altitude |
| `ensim --test` | `ensim check` |
| `examples/` in the repository | `ensim/examples/` inside the package |

### Changed defaults

| Function | Argument | 3.x | 4.0 |
|---|---|---|---|
| `simulate_flight_6dof` | `exit_area` | 0.01 m² | 0.0 |
| `run_monte_carlo` | `exit_area` | 0.01 m² | 0.0 |
| `run_monte_carlo` | `verbose` | `True` | removed; use `progress=` |

### Flight results

`FlightResult6DOF` gains `landing_x`, `landing_y`, `drogue_deploy_time`,
`parachute_deploy_time` and `forced_min_steps`. The landing point is
interpolated to the ground; in 3.x it was the last output sample.

`DispersionResult` counts landings separately (`n_landed`). Flights that hit the
time limit in the air no longer pull the landing statistics toward the pad.
