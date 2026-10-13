# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [4.0.0] - 2026-10-13

A rebuild. The physics was checked line by line against its sources, the
desktop application was redrawn from scratch, and the library can now be
installed without any GUI packages. Projects from 3.x open unchanged, but
results differ: see [Moving to 4.0](https://spaceengineerss.github.io/EnSim/migrating/).

### Breaking changes

- `pip install ensim` installs the library only. The desktop application
  needs `pip install "ensim[gui]"`.
- Propellants are named by their storage state (`LH2`, `LOX`, `LCH4`,
  `RP-1`, `NTO`, `HTP`, …) and enter the chamber with CEA's liquid
  enthalpies. 3.x names still load from project files.
- `StageEngine.thrust_sl` and `thrust_vac` are now `thrust_sl_per_engine`
  and `thrust_vac_per_engine`. 3.x multiplied an already clustered thrust by
  the engine count.
- `simulate_flight_6dof` takes `exit_area=0.0` by default (was 0.01 m²),
  which removed most of the thrust of small rockets at sea level.
- `run_monte_carlo` no longer prints or plots; `plot_dispersion` and
  `print_dispersion_summary` are gone. Landing statistics count only flights
  that reached the ground (`n_landed`).
- `ensim.utils.units` was rewritten around `Quantity`; `UnitCategory`,
  `UnitConverter` and `get_unit_symbol` were removed.
- The command line has subcommands: `ensim engine`, `ensim run`,
  `ensim check`, `ensim gui`.
- Project files are written as format 4.0.
- The legacy equilibrium solver, `get_atmosphere_table` and the 3.x GUI
  modules were removed.

### Physics

- Engine performance uses a frozen isentrope with temperature-dependent heat
  capacity, as CEA does. Against NASA CEA 3.3.2 for five liquid propellant
  pairs: chamber temperature within 0.01 %, vacuum Isp within 0.05 %.
- Standard state is 1 bar, matching the NASA Glenn data.
- Staging thrust was overstated by the engine count (9 and 5 times for the
  reference vehicles).
- The US Standard Atmosphere now converts geometric to geopotential altitude.
  3.x skipped the conversion and underestimated pressure by 1 % at 20 km,
  5 % at 50 km and 16 % at 80 km. Above 86 km the atmosphere continues
  smoothly.
- The 6-DOF derivative is evaluated from the full state at every Runge-Kutta
  stage, so thrust, drag, mass and inertia change within a step. The
  adaptive integrator keeps its step controller between steps and reports
  steps forced at the minimum size.
- Apogee and impact are interpolated within the step; landing points no
  longer depend on the output interval.
- Dual deploy with drogue and main, and a canopy fill time.
- Liftoff waits until thrust exceeds weight along the rail, and the run says
  so when it never does. Steps on the rail are a tenth of the normal step.
- The stability margin at liftoff used the dry centre of gravity.
- Monte Carlo ignored the user's drag coefficient and parachute.
- Pitzer departure enthalpy was wrong by a factor of 1/Tr.
- Haack nose cones had no centre-of-pressure coefficient.
- Cooling: wall temperature converges to 0.01 K, Bartz uses Eucken's Prandtl
  number and Bartz's viscosity estimate by default, N2H4 and MMH coolant data,
  and invalid inputs raise instead of returning numbers.

### Desktop application

- New interface: a test-stand console with one accent colour, IBM Plex type
  and its own icon set, in dark and light versions.
- Four sections (Engine, Vehicle, Flight, Analysis) instead of three levels of
  tabs. Every long calculation runs in the background and can be cancelled
  with Esc.
- The window is usable in about half a second; heavy parts load when first
  shown.
- SI and Imperial units throughout.
- Chamber contour from L\*, contraction ratio and an 80 % bell, exported as
  CSV or STL.
- A Markdown report and a JSON export of the whole workspace; every plot
  saves as PNG, SVG or CSV.
- The first start opens an example project.

### Library and tools

- `analyze_engine` returns vacuum, sea-level and operating-point
  performance in one call; `performance_at` evaluates any back pressure.
- `sweep_engine` for parametric studies.
- `ensim engine` and `ensim run` work without Qt and write JSON.
- Example projects ship inside the package.
- Documentation rebuilt as a site with theory, validation and API reference.


## [3.0.1] - 2026-08-20

### Distribution

- Added self-contained desktop downloads for Windows x64, Linux x86-64, and
  macOS on both Apple Silicon and Intel.
- Added release-time smoke tests and SHA-256 checksum files for every desktop
  bundle.

## [3.0.0] - 2026-08-13

### Scientific models

- Reworked packaged thermochemistry and NASA-format parsing with explicit
  conservation and CEA 3.3.2 reference comparisons.
- Corrected frozen ideal-nozzle operating-point reporting and removed hidden
  geometry changes and empirical separation penalties.
- Made thrust, mass flow, `c*` efficiency and specific impulse mutually
  consistent and removed the single-threshold flow-separation verdict.
- Removed the heuristic shifting-equilibrium API; equilibrium nozzle flow remains
  out of scope until energy, entropy, chemistry and element conservation are
  solved together at each station.
- Corrected Bartz inputs, added Gnielinski coolant convection, counterflow energy
  balance, pressure loss and explicit correlation-domain validation.
- Replaced heuristic cooling-channel auto-sizing with explicit geometry, mass
  flow and coolant inlet-state inputs.
- Replaced geometry-only combustion-stability verdicts with acoustic modes and
  user-supplied growth/damping classification.
- Added WGS-84 geodesy, ECEF/ECI transforms, J2 gravity and NASA NESC case 1/2
  reference subsets.
- Removed the undocumented synthetic transonic drag curve; trajectory drag now
  uses an explicit user-supplied axial coefficient and RK stages use depleted mass.
- Replaced proxy optimization objectives with reproducible reduced-order physics.
- Added positive-support uncertainty sampling, explicit seeds, failure accounting
  and correctly labelled confidence ellipses.

### Application and package

- Migrated the installed namespace to `ensim` and packaged data/assets with the
  wheel.
- Rebuilt the advanced panel around MOC design and engine uncertainty analysis.
- Fixed propellant identifier mapping and mass-to-mole conversion in the GUI.
- Removed nonfunctional replay, unit-toggle and duplicate optimization controls.
- Replaced unsourced hardware replicas and static propellant-performance previews
  with three reproducible, neutral engine input cases.
- Added safe headless 3-D fallback and GUI workflow/smoke tests.
- Rewrote scientific documentation around reproducible evidence and limitations.

## [2.0.0] - 2026-01-02

- Added the initial multi-stage, cooling, optimization, materials, mission,
  propellant-preset and desktop analysis modules.
- Published the `ensim` package and trusted-publishing workflow.

## [1.0.0] - 2026-01-02

- Initial open-source release with thermochemistry, ideal nozzle performance,
  PyQt6 interface, plotting and project export.

[Unreleased]: https://github.com/SpaceEngineerSS/EnSim/compare/v4.0.0...HEAD
[4.0.0]: https://github.com/SpaceEngineerSS/EnSim/compare/v3.0.1...v4.0.0
[3.0.1]: https://github.com/SpaceEngineerSS/EnSim/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/SpaceEngineerSS/EnSim/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/SpaceEngineerSS/EnSim/releases/tag/v2.0.0
[1.0.0]: https://github.com/SpaceEngineerSS/EnSim/releases/tag/v1.0.0
