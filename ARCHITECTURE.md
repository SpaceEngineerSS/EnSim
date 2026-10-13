# Architecture

EnSim is two things in one package: a numerical library (`ensim.core`,
`ensim.utils`) that needs NumPy, SciPy and optionally Numba, and a desktop
application (`ensim.ui`) on top of it that needs PyQt6 and Matplotlib. The
library never imports the application, and CI tests it on all three operating
systems with Qt not installed.

## Layout

```text
src/ensim/
  core/        models and result types; no Qt, no Matplotlib, no printing
  utils/       NASA data parser, units, export, Numba shim, user paths
  data/        nasa_thermo.dat, the NASA Glenn species set
  examples/    .ensim projects opened from the application
  ui/          desktop application
    app.py         start-up: splash, fonts, theme, main window
    theme.py       colour tokens, Qt stylesheet and Matplotlib style
    tasks.py       TaskRunner: background work with progress and cancel
    workspace.py   the open project and the last engine result
    shell.py       navigation rail, header, status strip
    sections/      Engine, Vehicle, Flight, Analysis, Log
    widgets/       form controls, plot canvas, 3-D view
    windows/       main window: menus, shortcuts, project lifecycle
  assets/      IBM Plex fonts, SVG icons, application icon
  cli.py       ensim engine | run | check | gui
tests/
  unit/        equations, invariants, edge cases
  validation/  comparisons with NASA CEA and NASA NESC check cases
  reference/   reference data with provenance
  ui/          headless end-to-end tests of the application
scripts/       reference generation, figures, screenshots, start-up benchmark
docs/          MkDocs site
```

## Engine path

```text
EngineInputs
  -> propellants.get_propellant      storage state, formula, enthalpy
  -> CombustionProblem.solve         Gibbs minimum + enthalpy balance -> Tc, x_j
  -> frozen_flow.frozen_nozzle       isentrope s(T,p) = s_c, throat at max(rho u),
                                     exit at the area ratio
  -> engine._performance             divergence, eta_c*, eta_CF, back pressure
  -> EngineResult                    vacuum, sea_level, operating, composition
```

`analyze_engine` is the only entry point the application uses. The chamber
contour (`geometry.chamber_contour`), the frozen-flow profile along it
(`frozen_flow.frozen_profile`) and the cooling study are computed from an
`EngineResult` afterwards and are not needed for performance.

## Flight path

`simulate_flight_6dof` packs the rocket, engine and environment into one
parameter vector and integrates a 14-element state with a Numba-compiled
derivative (`_flight_evaluate`) that recomputes thrust, aerodynamics, mass
properties and gravity at every Runge–Kutta stage. Events (rail exit, apogee,
parachute openings, impact) are handled between steps in Python, with Hermite
interpolation inside the step for apogee and impact. `run_monte_carlo` calls it
in a process pool, or in the current process when `n_workers=1`.

## Numba

`ensim.utils.numba_support.jit` is `numba.njit` when Numba is installed and
the identity otherwise, so every kernel also runs as plain Python. Compiled
kernels are cached in the user's cache directory, which also works in the
packaged application where the install directory is read-only. The first
engine run of a session pays the compilation; the application does it in the
background while the window is already usable.

## Application

- **Start-up.** `app.run()` shows a frameless splash, loads fonts and theme and
  builds the main window. Sections are constructed the first time they are
  shown; inside them, heavy tabs (cooling, stability, 3-D view) are `Lazy`
  placeholders. The window is usable in about half a second;
  `scripts/bench_startup.py` measures it.
- **Work off the GUI thread.** Every calculation goes through `TaskRunner`, a
  `QThreadPool` wrapper. A task receives a context with `progress()` and
  `cancelled()`; the core functions accept `progress` and `cancel` callbacks and
  stop cooperatively. Esc cancels the running task. Tests use
  `TaskRunner(synchronous=True)`.
- **One source of state.** `Workspace` owns the `ProjectManager`, the last
  engine run and its contour, and emits signals when they change. Sections
  read from it and write their inputs back to the project data.
- **Units.** Values are SI everywhere inside. `NumberField` converts at the edge
  using `ensim.utils.units`, so switching to Imperial never changes a stored
  number.
- **Theme.** `theme.py` defines the console and paper palettes once and
  generates both the Qt stylesheet and the Matplotlib style from them. Icons
  are SVG drawn with `currentColor` and tinted at load time.

## Project files

JSON with `meta`, `engine`, `rocket`, `recovery`, `environment` and
`results` sections; see the [documentation](https://spaceengineerss.github.io/EnSim/library/project-files/).
Readers are tolerant: unknown keys are ignored, missing keys default, and 1.x
and 3.x layouts are migrated on load. Files never contain code.

## Errors

Invalid inputs raise `ValueError` or `PhysicsViolationError`; solver failures
raise the `CalculationError` family from `ensim.core.types`. Nothing falls back
silently to another species, coolant, correlation or solver. The application
catches these at the task boundary and shows the message in the status strip
and the log.

## Adding a model

1. Put it in `ensim.core` with SI inputs, a dataclass result and no I/O.
2. Test the equations, the limits and the invalid inputs in `tests/unit`.
3. If an independent reference exists, add it to `tests/reference` with its
   source and a test in `tests/validation`.
4. Document the model in `docs/theory` and its gaps in
   `docs/model-limitations.md`.
5. Only then add it to the application, as a section tab that builds lazily
   and runs through `TaskRunner`.
