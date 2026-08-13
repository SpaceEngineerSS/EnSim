# EnSim architecture

## Design boundaries

EnSim separates numerical models from presentation and file I/O. Modules under
`src/ensim/core` do not import Qt. The desktop layer converts user input to SI,
runs expensive calculations in worker threads and renders immutable result
objects. This boundary makes the physics usable from tests, scripts and the GUI.

```text
PyQt6 interface
    | validated SI inputs
    v
worker orchestration -------- project/export services
    |
    +-- thermochemistry --> frozen nozzle performance --> engine result
    +-- vehicle + environment + integrator -----------> flight result
    +-- cooling / MOC / optimization / UQ ------------> specialist results
```

## Package layout

```text
src/ensim/
  core/          numerical models and typed result records
  data/          packaged thermodynamic data resources
  ui/            windows, widgets and background workers
  utils/         NASA-data parsing, units and export
  visualization/ plotting support
tests/
  unit/          equations, invariants, edge cases and UI smoke tests
  validation/    external-reference comparisons
  reference/     provenance and immutable NESC reference subsets
```

The import namespace is `ensim`; the physical source layout is an implementation
detail. Public examples must never import from `src`.

## Engine calculation path

1. Resolve visible propellant choices to packaged species identifiers.
2. Convert the mass O/F ratio to the reactant mole basis using database molar
   masses.
3. Solve adiabatic, constant-pressure equilibrium using element constraints and
   an enthalpy balance.
4. Derive mixture molar mass and frozen `Cp/Cv` from the converged composition.
5. solve the supersonic area-Mach relation for the selected area ratio;
6. calculate exit pressure, thrust coefficient, characteristic velocity and
   specific impulse for the selected ambient pressure.

The same geometry is retained when reporting vacuum and sea-level operating
points. EnSim does not silently cap the expansion ratio. It reports the ideal
one-dimensional attached-flow result and identifies overexpansion as a model
limitation; it does not invent a separation penalty.

## Flight calculation path

The vehicle model owns mass properties, engine data, aerodynamic geometry and
staging information. The local model propagates position and velocity in an ENU
frame. The optional WGS-84 model propagates ECI position, velocity and attitude,
uses ellipsoidal geodesy and J2 gravity, and transforms the rotating atmosphere
through ECEF. Quaternion normalization and nonnegative mass are maintained as
explicit numerical invariants.

Aerodynamic forces use air-relative velocity. The current coefficient model is
appropriate for preliminary slender-vehicle studies, not general CFD-equivalent
prediction. See `docs/FLIGHT_VALIDATION.md`.

## Concurrency and determinism

Qt workers own each long-running task. The GUI prevents overlapping engine runs
and never mutates a result while it is being displayed. Stochastic analyzers
accept an explicit seed. Engine UQ preserves sample order and records failed
samples rather than replacing them with nominal values.

## Data and units

- internal calculations use SI;
- thermodynamic molar mass is exposed as g/mol at the current API boundary;
- the packaged NASA-format database is loaded with `importlib.resources`;
- project files contain input parameters, not executable code;
- reference test data include source URLs and SHA-256 hashes.

## Failure policy

Inputs outside mathematical domains raise `ValueError`; numerical failures use
the calculation exceptions in `ensim.core.types`. Material, coolant and species
lookups do not silently fall back to a different physical substance. Monte Carlo
results expose attempted, valid and failed sample counts.

## Extending the project

A new scientific model should include:

1. an explicit domain of applicability and unit contract;
2. a typed input/result boundary;
3. unit tests for equations, invariants and invalid inputs;
4. an external comparison when credible reference data exist;
5. documentation that distinguishes correlation, verification and validation;
6. no hidden empirical margins or undocumented fallback constants.
