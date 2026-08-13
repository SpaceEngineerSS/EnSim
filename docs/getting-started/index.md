# Getting started

## Install from PyPI

EnSim requires Python 3.10 or newer.

```bash
python -m pip install ensim
ensim --test
ensim
```

The smoke test loads the packaged thermodynamic database and runs one coupled
H2/O2 equilibrium and ideal-nozzle calculation.

## Install a source checkout

```bash
git clone https://github.com/SpaceEngineerSS/EnSim.git
cd EnSim
python -m venv .venv
python -m pip install -e ".[dev,docs]"
python -m pytest
python main.py
```

Activate the environment using `.venv\Scripts\Activate.ps1` on PowerShell or
`source .venv/bin/activate` on POSIX shells.

## First desktop calculation

1. Select fuel and oxidizer. Visible labels map to packaged species identifiers.
2. Enter mass O/F ratio, chamber pressure, nozzle area ratio, throat diameter and
   ambient pressure in the displayed SI units.
3. Run the engine calculation.
4. Check convergence and the reported model notes before reading performance.
5. Use Results for station profiles, Engine for specialist analyses and Vehicle
   for the configured rocket and flight simulation.

The sea-level and vacuum figures use the same selected nozzle geometry. An
overexpanded attached-flow result does not include an empirical separation loss.

## 3-D troubleshooting

Interactive rendering requires PyVista, PyVistaQt, Qt and a working OpenGL/VTK
environment. EnSim falls back to a nonfatal placeholder in headless sessions.

## Next reading

Read [model limitations](../MODEL_LIMITATIONS.md) and
[validation evidence](../VALIDATION.md) before interpreting a result as more than
a preliminary engineering estimate.
