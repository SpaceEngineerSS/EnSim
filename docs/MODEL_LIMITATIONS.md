# Model limitations and appropriate use

EnSim is a preliminary engineering simulator. Numerical convergence is not proof
that a result represents a buildable or safe engine.

## Thermochemistry

- ideal-gas mixture; no fugacity correction or real-fluid feed-system state;
- optional Pitzer-Curl second-virial and Redlich-Kwong utilities are preliminary
  gas-phase corrections, not a multiphase propellant-property package; mixture
  estimates use Kay pseudo-critical mixing and reject missing critical data;
- only packaged gas species participate;
- no condensed carbon, metal oxides, ions or plasma chemistry;
- bundled engine and staging cases are editable numerical inputs, not replicas,
  performance specifications or validation records;
- chamber equilibrium does not model injector mixing, ignition or finite-rate
  combustion.

## Nozzle and thrust

- one-dimensional, steady, adiabatic and attached flow;
- chamber composition and gamma are frozen for the standard performance result;
- equilibrium-shifting nozzle expansion is not implemented; EnSim does not infer
  recombination from temperature thresholds or apply an empirical Isp increment;
- conical momentum divergence uses only `(1 + cos(alpha))/2`; no
  boundary-layer, erosion, two-phase or separated-flow loss;
- reported ideal thrust must not be used as an acceptance prediction without
  independently justified efficiencies.
- the MOC contour is a two-dimensional planar, sharp-corner minimum-length
  solution; it is not an axisymmetric bell nozzle and contains no viscous,
  boundary-layer, variable-gamma or separated-flow correction;
- in the planar MOC result, `throat_radius` is a backward-compatible API name for
  throat half-height and the area ratio is the exit-to-throat height ratio;

## Cooling and structures

- Bartz and Gnielinski are engineering correlations with restricted domains;
- geometry is reduced to local hydraulic and wall dimensions;
- no conjugate finite-element conduction, film cooling, boiling, coking,
  roughness evolution, fatigue, creep or transient start/shutdown analysis;
- material limits are screening values, not allowables for certification.

## Combustion stability

Acoustic eigenfrequencies identify possible resonances. EnSim cannot infer a
universal stable/unstable verdict from geometry alone. Growth and damping inputs
must come from a defensible model or experiment.

## Flight

- rigid body with preliminary aerodynamic coefficients;
- no flexible modes, propellant slosh, actuator dynamics, sensor model or closed
  loop guidance/navigation/control unless explicitly supplied by another model;
- atmosphere and wind are engineering profiles, not a launch-day forecast;
- the implemented 1976 Standard Atmosphere layers end at 84.852 km; above that
  boundary pressure and density are explicitly set to zero for aerodynamic
  propagation, without claiming an upper-atmosphere composition model;
- the general trajectory is not validated against telemetry.

## Uncertainty and optimization

Outputs are conditional on the selected distributions, bounds and objective.
Monte Carlo frequency is not evidence that the assumed distributions are true.
Reduced-order optimizers do not enforce manufacturing, combustion stability,
structural dynamics, range safety or all thermal constraints.

## Decision policy

Use EnSim to understand trends, screen concepts and reproduce documented trade
studies. For hardware decisions, establish an independent model-validation plan,
use traceable test data, quantify model-form discrepancy and obtain review by a
qualified propulsion/flight-safety team.
