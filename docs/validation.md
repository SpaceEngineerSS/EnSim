# Verification and validation evidence

Last reviewed: 2026-08-13

## Terminology

- **Unit verification** checks that code implements an equation or invariant.
- **Reference comparison** compares output with an independent implementation.
- **Validation** compares a model with physical observations for its intended use.

EnSim has extensive unit verification and several external reference comparisons.
It does not yet have enough hot-fire and flight-test evidence to claim general
hardware-level validation.

## Thermochemical equilibrium comparison

The committed comparison values were generated with the official NASA `cea`
3.3.2 Python distribution. Both cases are ideal-gas, adiabatic, constant-pressure
(`HP`) equilibrium with gas-phase reactants at 298.15 K. The mole basis and
pressure are explicit in `tests/validation/test_cea_comparison.py`.

| Case | Pressure | Quantity | NASA CEA 3.3.2 | EnSim | Relative difference |
|---|---:|---|---:|---:|---:|
| 2 H2 + O2 | 6.89 MPa | chamber temperature | 3674.145 K | 3672.064 K | -0.0566% |
| | | frozen Cp/Cv | 1.193434 | 1.193300 | -0.0112% |
| | | mean molar mass | 15.76768 g/mol | 15.76440 g/mol | -0.0208% |
| CH4 + 2 O2 | 10.0 MPa | chamber temperature | 3680.166 K | 3677.934 K | -0.0607% |
| | | frozen Cp/Cv | 1.196242 | 1.196273 | +0.0027% |
| | | mean molar mass | 22.83704 g/mol | 22.83243 g/mol | -0.0202% |

The automated acceptance limit for these three quantities is 2%. The observed
agreement above applies only to the listed species sets, reactant phases,
temperatures, pressures and equilibrium constraint. It must not be generalized
to every supported reactant pair or to liquid injection thermodynamics.

Additional tests enforce nonnegative finite composition, element conservation,
enthalpy closure, dissociation trends, NASA-polynomial interval selection and
analytical characteristic-velocity/thrust-coefficient relations.

## Flight dynamics reference comparisons

EnSim includes immutable subsets of two public NASA Engineering and Safety Center
atmospheric check cases:

- case 1: a dragless sphere over rotating WGS-84 Earth, compared at seven points
  over 30 seconds;
- case 2: a dragless torque-free tumbling brick, compared at seven points over
  30 seconds.

The exact source URLs, selected columns, units and SHA-256 hashes are in
`tests/reference/README.md`. Tolerances and remaining gaps are documented in
[Flight verification](FLIGHT_VALIDATION.md). These are software cross-checks, not
flight-test validation.

## Cooling, acoustics, optimization and UQ

These modules currently have equation, trend, domain and reproducibility tests:

- Bartz gas-side and Gnielinski coolant-side correlations;
- counterflow energy balance, pressure loss and invalid-domain rejection;
- cylindrical acoustic eigenfrequencies and explicit growth-minus-damping logic;
- deterministic ideal-nozzle and stage-allocation objectives;
- seeded uncertainty samples, positive-support inputs, failure accounting and
  confidence-ellipse geometry.

No claim is made that these reduced-order models reproduce a particular engine
or launch vehicle without independent calibration data.

## Running the evidence

```bash
python -m pytest tests/validation -v
python -m pytest tests/unit -v
```

The short `ensim --test` command checks installation and one coupled H2/O2
calculation only.

## Known evidence gaps

1. hot-fire wall-temperature and coolant-pressure histories for a fully defined
   chamber/channel geometry;
2. finite-rate or shifting-equilibrium nozzle comparison across multiple area
   ratios;
3. aerodynamically coupled NASA NESC case 3 or equivalent;
4. telemetry comparison for a documented vehicle;
5. experimental combustion-stability growth and damping data.

Until those gaps are closed, EnSim should be described as a preliminary
engineering simulator with documented verification evidence.

## Primary sources

- Gordon and McBride, [NASA RP-1311 Part I](https://ntrs.nasa.gov/citations/19950013764), 1994.
- McBride, Zehe and Gordon, [NASA/TP-2002-211556](https://ntrs.nasa.gov/citations/20020085330), 2002.
- Jackson, Murri and Shelton, [NASA/TM-2015-218675](https://ntrs.nasa.gov/citations/20150001263), 2015.
- NASA, [Technical Bulletin 24-04](https://ntrs.nasa.gov/citations/20240013467), 2024.
