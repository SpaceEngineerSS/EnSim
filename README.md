# EnSim

Liquid rocket engine analysis and six-degree-of-freedom flight simulation, as a
desktop application and a Python library.

[![CI](https://github.com/SpaceEngineerSS/EnSim/actions/workflows/ci.yml/badge.svg)](https://github.com/SpaceEngineerSS/EnSim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ensim.svg)](https://pypi.org/project/ensim/)
[![Python](https://img.shields.io/pypi/pyversions/ensim.svg)](https://pypi.org/project/ensim/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![EnSim, engine performance](https://raw.githubusercontent.com/SpaceEngineerSS/EnSim/main/docs/assets/screens/engine_performance.png)

Pick two propellants, a mixture ratio and a chamber pressure. EnSim finds the
chamber equilibrium, expands the gas through the nozzle and gives you Isp,
thrust and the composition of the exhaust, matching NASA CEA to within 0.05 %.
From there you can draw and cool the chamber, put the engine in a rocket, fly
it from the rail to landing and scatter a few hundred flights to see where it
comes down.

**Documentation:** <https://spaceengineerss.github.io/EnSim/>

## Install

The desktop application:

```console
$ pip install "ensim[gui]"
$ ensim
```

or a self-contained download for Windows, macOS and Linux from the
[releases page](https://github.com/SpaceEngineerSS/EnSim/releases/latest).

The library and the command line only, without Qt:

```console
$ pip install ensim
$ ensim engine --fuel LCH4 --of 3.4 --pc 100 --eps 35 --throat-area 200
```

Python 3.10 or newer.

## In Python

```python
from ensim.core.engine import EngineInputs, analyze_engine

engine = analyze_engine(EngineInputs(
    fuel="LCH4", oxidizer="LOX", of_ratio=3.4,
    chamber_pressure=100e5, expansion_ratio=35, throat_area=0.02,
))
engine.chamber_temperature                 # 3598.0 K
engine.vacuum.isp, engine.sea_level.isp    # 337.7 s, 272.2 s
engine.vacuum.thrust                       # 366.0 kN
```

Flights, dispersion studies, staging and the lower-level solvers are covered in
the [library guide](https://spaceengineerss.github.io/EnSim/library/python/).

## What is inside

| | |
|---|---|
| Chamber | Gibbs minimisation over the NASA Glenn species set, propellants in their storage state |
| Nozzle | Frozen isentrope with temperature-dependent heat capacity, divergence and efficiency losses, off-design back pressure |
| Chamber contour | Cylinder from L\*, conical convergence, 80 % bell; CSV and STL export |
| Cooling | Bartz gas side, Gnielinski coolant side, wall temperature and pressure drop along the contour |
| Acoustics | Longitudinal, tangential and radial chamber modes; chug estimate |
| Vehicle | Barrowman centre of pressure, stability margin through the burn, multi-stage sizing |
| Flight | 6-DOF rigid body, RK4 or Dormand–Prince, US Standard Atmosphere 1976, wind profile, optional WGS-84/J2 rotating Earth, dual-deploy recovery |
| Studies | Parametric sweeps, nozzle and design-point optimisation, gravity-turn ascent, engine uncertainty, Monte Carlo landing dispersion, planar MOC nozzle |

## How far to trust it

Engine numbers are checked against NASA CEA 3.3.2 on every commit, and the
flight model against two NASA NESC check cases. Neither the cooling model nor
the aerodynamics has been compared with hardware. The
[validation](https://spaceengineerss.github.io/EnSim/validation/) and
[model limitations](https://spaceengineerss.github.io/EnSim/model-limitations/)
pages say exactly what has and has not been checked. EnSim is for design studies
and teaching, not for certifying hardware.

## Moving from 3.x

Projects open unchanged, but propellants now enter the chamber as liquids and
several flight and staging errors were fixed, so results move. See
[Moving to 4.0](https://spaceengineerss.github.io/EnSim/migrating/) and the
[changelog](CHANGELOG.md).

## Development

```console
$ git clone https://github.com/SpaceEngineerSS/EnSim.git
$ cd EnSim
$ pip install -e ".[all]"
$ pytest
$ ruff check . && ruff format --check .
$ mkdocs serve
```

[CONTRIBUTING.md](CONTRIBUTING.md) explains the layout of the code and what a
change needs before it is merged; [ARCHITECTURE.md](ARCHITECTURE.md) describes
how the pieces fit.

Bug reports and challenges to the physics are welcome in the
[issue tracker](https://github.com/SpaceEngineerSS/EnSim/issues). Please include
the inputs, units and, for a physics question, the reference you compare with.

## Licence

MIT. The bundled IBM Plex fonts are under the SIL Open Font License
(`src/ensim/assets/fonts/OFL.txt`).
