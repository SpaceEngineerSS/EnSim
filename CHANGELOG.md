# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/SpaceEngineerSS/EnSim/compare/v3.0.1...HEAD
[3.0.1]: https://github.com/SpaceEngineerSS/EnSim/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/SpaceEngineerSS/EnSim/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/SpaceEngineerSS/EnSim/releases/tag/v2.0.0
[1.0.0]: https://github.com/SpaceEngineerSS/EnSim/releases/tag/v1.0.0
