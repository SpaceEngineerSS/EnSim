# Model limitations

EnSim is built for the conceptual and preliminary stages of a design: comparing
propellants, sizing a chamber and its channels, checking that a sounding rocket
is stable and roughly where it will land. This page lists what the models leave
out, so you know when a result needs a better tool or a test.

A converged answer is not a safe design.

## Thermochemistry

- Ideal-gas products. No condensed phases (soot, metal oxides), no ions.
- Only species in the packaged NASA Glenn set can form. That covers the C, H, O,
  N chemistry of the listed propellants.
- The chamber reaches equilibrium. Injector mixing, atomisation and
  finite-rate combustion are summarised by the c\* efficiency you enter.
- Reactant enthalpies are fixed at the storage temperatures in the propellant
  table. A propellant conditioned to another temperature can be entered
  through the library with its own enthalpy.
- The real-gas departure functions in `thermodynamics` (Pitzer, Redlich–Kwong)
  are single-phase gas corrections for feed-system estimates, not a property
  package for liquids near their critical point.

## Nozzle

- One-dimensional, steady, adiabatic and attached flow.
- Frozen composition. For hot hydrogen-rich mixtures shifting equilibrium gives
  1–2 % more Isp; EnSim does not add it.
- Losses are the conical divergence factor and the two efficiencies. Boundary
  layer, two-phase flow, erosion and kinetic losses are not computed.
- No flow separation model. An over-expanded nozzle is reported with
  attached-flow thrust and a warning below $p_e/p_a=0.4$.
- The drawn bell uses fixed entry and exit angles; it is not an optimised Rao
  contour. The MOC tool is planar, sharp-cornered and inviscid.

## Cooling

- Steady, one-dimensional, single-phase coolant with properties fixed at the
  inlet state.
- Bartz and Gnielinski are correlations; Bartz in particular can be off by tens
  of percent away from the throat. The gas transport properties are estimates
  unless supplied.
- Not modelled: film cooling, boiling, supercritical property swings, rough
  channels, coking, axial conduction, radiation, and thermal or mechanical
  stress. *Melt margin* is not a structural allowable.

## Combustion stability

- Mode frequencies of an ideal cylinder at the chamber sound speed.
- No growth prediction without driving and damping rates from elsewhere.
- Chug is a lumped single-frequency estimate.

## Flight

- One rigid body: no flexible modes, propellant slosh, roll from canted fins, or
  guidance and control.
- Constant axial drag coefficient; it does not vary with Mach or Reynolds number.
- Barrowman normal force and centre of pressure, valid at small angles of
  attack. No pitch damping derivative is modelled, so oscillations after a gust
  decay more slowly than in reality.
- Moments of inertia of a uniform cylinder.
- US Standard Atmosphere 1976 up to 86 km. Above, the temperature is held and
  pressure decays exponentially from the 86 km value, which keeps drag
  continuous but is not a model of the thermosphere. `get_atmosphere` marks
  such states with `within_model = False`.
- Wind is steady with a power-law profile up to 1 km. No gusts and no
  turbulence; a forecast profile cannot be loaded yet.
- Staged vehicles are sized with the rocket equation and not flown in six
  degrees of freedom.

## Uncertainty and optimisation

- Inputs are independent. Correlated scatter is not represented.
- The engine uncertainty study uses constant-$\gamma$ nozzle relations around
  the last result, not the full equilibrium solution.
- Optimisers search the objective they are given. They know nothing about
  manufacturability, stability, structures or range safety.

## Using results

Use EnSim to rank options, find trends and size first iterations. Before
hardware is built or flown, check the decisive numbers with an independent
method, add margin for what this page lists, and have a qualified person
review the result.
