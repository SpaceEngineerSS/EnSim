# Install

EnSim needs Python 3.10 or newer. The package comes in two sizes.

| You want | Command | What it adds |
|---|---|---|
| The library and the command line | `pip install ensim` | NumPy, SciPy, Numba |
| The desktop application as well | `pip install "ensim[gui]"` | PyQt6, Matplotlib, PyVista |

The library install has no graphical dependencies, so it is safe on servers,
in CI and in notebooks. `ensim check` confirms an installation in a few seconds:

```console
$ ensim check
EnSim 4.0.0
  species                      43
  H2O Cp(1000 K)               41.295 J/(mol K)
  LH2/LOX Tc, O/F 6, 68.9 bar  3483.5 K
  Isp vacuum, eps 40           429.3 s
OK
```

The chamber temperature in the third line is compared with NASA CEA 3.3.2,
which gives 3483.4 K for the same case.

## Desktop bundles

Each release on [GitHub](https://github.com/SpaceEngineerSS/EnSim/releases)
carries ready-to-run builds that need no Python:

- Windows: a single `EnSim-<version>-Windows-x64.exe`
- macOS: `.dmg` images for Apple silicon and for Intel
- Linux: a `tar.gz` with an `EnSim` executable

Every file has a `.sha256` next to it.

!!! note "First start"
    The numerical kernels are compiled the first time they run and cached in
    your user cache folder (`%LOCALAPPDATA%\EnSim\Cache` on Windows,
    `~/Library/Caches/EnSim` on macOS, `~/.cache/ensim` on Linux). The first
    calculation after an install takes a second or two longer; later sessions
    start straight away.

## From a checkout

```console
$ git clone https://github.com/SpaceEngineerSS/EnSim.git
$ cd EnSim
$ python -m venv .venv
$ .venv/bin/pip install -e ".[gui,dev]"     # .venv\Scripts\pip on Windows
$ python -m pytest
```

`python main.py` starts the desktop application from the source tree.

## Optional pieces

- The 3D nozzle viewer uses PyVista and needs working OpenGL. Everything else
  works without it, and STL export of the nozzle wall does not need it either.
- `ENSIM_DISABLE_3D=1` turns the viewer off, for remote desktops whose OpenGL
  drivers crash.
- Without Numba (for example on a Python version it does not support yet) the
  same code runs as plain Python, correct but several times slower.
