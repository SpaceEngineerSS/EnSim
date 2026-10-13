# Python API

`pip install ensim` gives you the models without any GUI packages. Everything
below runs in a plain interpreter or a notebook; all inputs and outputs are SI.

## An engine

```python
from ensim.core.engine import EngineInputs, analyze_engine

inputs = EngineInputs(
    fuel="LCH4", oxidizer="LOX", of_ratio=3.4,
    chamber_pressure=100e5,      # Pa
    expansion_ratio=35,
    throat_area=0.02,            # m², optional: without it there is no thrust or mass flow
)
engine = analyze_engine(inputs)

engine.chamber_temperature       # 3598.0 K
engine.gamma, engine.molecular_weight   # 1.1955, 21.770 g/mol
engine.vacuum.c_star             # 1809.2 m/s
engine.vacuum.isp, engine.sea_level.isp  # 337.7 s, 272.2 s
engine.vacuum.thrust             # 366.0e3 N
engine.vacuum.mass_flow_rate     # 110.55 kg/s
engine.composition               # {'H2O': 0.4931, 'CO': 0.1782, 'CO2': 0.1302, ...}
```

`fuel` and `oxidizer` name [propellants](../reference/propellants.md) in their
storage state: `LH2`, `LCH4`, `RP-1`, `N2H4`, `MMH`, `UDMH`, `ETHANOL`, `LOX`,
`NTO`, `HTP`, `N2O`. Any other name is looked up in the thermodynamic database
and treated as a gas at 298.15 K (or at `fuel_temperature`).

The result carries three operating points, `vacuum`, `sea_level` (or `None`
when the nozzle cannot run choked at one atmosphere) and `operating` at
`inputs.ambient_pressure`. `engine.to_dict()` gives plain JSON.

### Other back pressures

```python
from ensim.core.mission import atmosphere_properties

for h in (0, 5_000, 10_000, 20_000):
    p = engine.performance_at(atmosphere_properties(h)[1])
    print(f"{h/1000:5.0f} km  {p.thrust/1e3:7.1f} kN  {p.isp:6.1f} s")
```

```text
    0 km    295.1 kN   272.2 s
    5 km    328.2 kN   302.8 s
   10 km    347.5 kN   320.5 s
   20 km    362.2 kN   334.1 s
```

### A sweep

```python
import numpy as np
from ensim.core.analysis import sweep_engine

sweep = sweep_engine(inputs, "of_ratio", np.linspace(2.6, 4.2, 9))
sweep.values[np.argmax(sweep.isp_vacuum)]            # 2.8, highest Isp
sweep.values[np.argmax(sweep.chamber_temperature)]   # 3.8, hottest
```

Peak Isp sits on the fuel-rich side of peak temperature because the lighter
exhaust gains more than the cooler flame loses.

## A flight

```python
from ensim.core.engine import EngineInputs, analyze_engine
from ensim.core.flight_6dof import simulate_flight_6dof
from ensim.core.rocket import create_default_rocket

motor = analyze_engine(EngineInputs(
    "RP-1", "LOX", 2.3, 20e5, 4.0, throat_area=1.4e-4, eta_cstar=0.95, eta_cf=0.97,
))
rocket = create_default_rocket()            # 9 kg sounding rocket, 3.2 kg propellant
burn = rocket.engine.propellant_mass / motor.vacuum.mass_flow_rate   # 18.9 s

flight = simulate_flight_6dof(
    rocket,
    thrust_vac=motor.vacuum.thrust,
    isp_vac=motor.vacuum.isp,
    burn_time=burn,
    exit_area=1.4e-4 * 4,                   # for the ambient-pressure thrust loss
    launch_angle_deg=87,
    wind_speed=4, wind_direction_deg=270,
    drogue_diameter=0.45, chute_diameter=1.5,
    deploy_at_apogee=False, main_deploy_altitude_m=300,
    max_time=600,
)
flight.apogee_altitude, flight.apogee_time   # 6112 m, 36.8 s
flight.max_mach                              # 1.14
flight.landing_x, flight.landing_y           # -1112 m east, 1165 m north
```

The result holds full time histories (`time`, `position_z`, `mach`,
`angle_of_attack`, `thrust`, …) at `output_dt` spacing. Pass
`cancel=lambda: flag.is_set()` to stop a long run from another thread, and
`progress=` to receive the fraction completed.

## A dispersion study

```python
from ensim.core.monte_carlo import DispersionConfig, run_monte_carlo

study = run_monte_carlo(
    rocket, motor.vacuum.thrust, motor.vacuum.isp, burn,
    exit_area=1.4e-4 * 4,
    config=DispersionConfig(num_simulations=50, seed=1),
    launch_angle_deg=87, chute_diameter=1.5, deploy_at_apogee=False,
    flight_options={"drogue_diameter": 0.45, "main_deploy_altitude_m": 300},
    max_time=600,
)
study.n_landed, study.cep_radius            # 50, 1840 m
study.apogee_mean, study.apogee_std         # 5690 m, 991 m
```

`n_workers` in the configuration sets the number of processes; `1` runs in the
current one. On Windows and macOS, call the study from inside
`if __name__ == "__main__":` when more than one process is used.

## Stages

```python
from ensim.core.staging import create_medium_lift_reference

vehicle = create_medium_lift_reference()
vehicle.payload_mass = 10_000
vehicle.get_total_delta_v()     # 11492 m/s, rocket equation only
```

## Units

The library is SI throughout. For display or for data from elsewhere:

```python
from ensim.utils.units import Quantity, convert

convert(1.0, Quantity.FORCE, "kN", "lbf")    # 224.809
convert(3000.0, Quantity.PRESSURE, "psi", "bar")
```

## Lower level

`analyze_engine` is built from parts you can use on their own:

- `ensim.core.chemistry.CombustionProblem` for chamber equilibrium alone, with
  any reactant set, including your own formula and enthalpy;
- `ensim.core.frozen_flow.frozen_nozzle` and `frozen_profile` for the nozzle;
- `ensim.core.geometry.chamber_contour` for the wall;
- `ensim.core.cooling.analyze_cooling_system`, `ensim.core.instability`,
  `ensim.core.optimization` and `ensim.core.engine_uq` for the other studies.

The [reference](../reference/engine.md) pages list every public function.
