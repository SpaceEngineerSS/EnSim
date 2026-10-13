# Project files

An `.ensim` file is JSON with one object per section. Every field is optional:
missing fields take their defaults, and fields EnSim does not know are ignored,
so a file written by a newer version still opens.

```json
{
  "meta": {"version": "4.0", "created": "2026-10-13T09:00:00", "modified": "…"},
  "engine": {
    "fuel": "RP-1", "oxidizer": "LOX", "of_ratio": 2.3,
    "chamber_pressure_bar": 20.0, "throat_area_cm2": 1.4, "expansion_ratio": 4.0,
    "ambient_pressure_bar": 1.01325, "eta_cstar": 0.95, "eta_cf": 0.97, "alpha_deg": 15.0,
    "contraction_ratio": 4.0, "characteristic_length": 1.0
  },
  "rocket": {
    "name": "Kerolox sounding rocket", "nose_shape": "haack", "nose_length": 0.45,
    "body_length": 1.8, "body_diameter": 0.12, "axial_drag_coefficient": 0.42,
    "fin_count": 4, "fin_root_chord": 0.26, "fin_tip_chord": 0.10, "fin_span": 0.16,
    "engine_mass_dry": 2.0, "tank_length": 0.45, "fuel_mass": 0.97, "oxidizer_mass": 2.23
  },
  "recovery": {"dual_deploy": true, "drogue_diameter": 0.45, "main_diameter": 1.5, "main_deploy_altitude": 300.0},
  "environment": {"wind_speed": 4.0, "wind_direction": 270.0, "rail_length": 3.0, "launch_angle": 87.0},
  "results": {"isp_vacuum": 257.3, "thrust": 427.3, "apogee": 6365.0}
}
```

Units are given in the field names where they are not SI (`_bar`, `_cm2`,
`_deg`); everything else is SI. `results` caches the last numbers for quick
listing and is recomputed when the project runs.

## Older files

Files from EnSim 3.x and 1.x load without changes:

- the flat 1.x layout is mapped to the `engine` section;
- the 3.x ambient text (`"Sea Level (1.01 bar)"`) becomes `ambient_pressure_bar`;
- propellant names change to the storage-state keys: `H2` → `LH2`, `O2` → `LOX`,
  `CH4` → `LCH4`, `RP1` → `RP-1`, `N2O4` → `NTO`, `H2O2` → `HTP`.

The last point changes results: 3.x evaluated every reactant as a gas at 298 K.
See [moving to 4.0](../migrating.md).

## From Python

```python
from ensim.core.project_manager import ProjectManager

project = ProjectManager()
data = project.load("kerolox_sounding_rocket.ensim")
rocket = project.build_rocket()
main, drogue = project.build_recovery()
data.engine.chamber_pressure_bar = 25.0
project.save()                     # raises ProjectFileError with a readable message on failure
```
