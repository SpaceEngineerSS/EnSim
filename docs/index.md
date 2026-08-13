# EnSim documentation

EnSim combines a Python numerical package with a PyQt6 desktop interface for
preliminary rocket-propulsion and flight studies.

## Start here

- [Installation and first calculation](getting-started/index.md)
- [Desktop workflow](user-guide/index.md)
- [Governing equations](THEORY.md)
- [Verification and validation evidence](VALIDATION.md)
- [Model limitations](MODEL_LIMITATIONS.md)
- [Python API](api/index.md)

## Evidence at a glance

Two ideal-gas HP equilibrium cases are cross-compared with NASA CEA 3.3.2.
Selected dragless translation and torque-free rotation histories are compared
with public NASA NESC check cases. Correlation-based cooling, acoustics,
optimization and uncertainty modules have equation/invariant tests but no broad
experimental validation claim.

EnSim is suitable for concept screening, reproducible trade studies and
education. It is not a certification, range-safety or hardware-release tool.

## Scientific modules

| Area | Main model | Evidence |
|---|---|---|
| Thermochemistry | ideal-gas Gibbs equilibrium and enthalpy closure | analytical checks and NASA CEA comparison |
| Nozzle | frozen, calorically perfect, 1-D attached flow | analytical relations |
| Cooling | Bartz, Gnielinski and resistance network | equation/domain/trend tests |
| Instability | cylindrical acoustics; supplied rate balance | eigenfrequency and classification tests |
| Flight | local ENU or ECI/WGS-84/J2 rigid body | invariants and NESC subsets |
| UQ | seeded sampling and covariance statistics | deterministic/statistical tests |

Report defects or scientific challenges through
[GitHub Issues](https://github.com/SpaceEngineerSS/EnSim/issues).
