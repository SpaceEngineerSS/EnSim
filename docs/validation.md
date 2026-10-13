# Validation

Every number on this page comes from a test in `tests/validation` that runs on
each commit. When a result changes, a test fails before the page goes stale.

```console
$ python -m pytest tests/validation -v
```

Three words are used carefully below. *Verification* checks that the code
solves its equations. A *reference comparison* checks the result against an
independent program solving the same problem. *Validation* would compare with
measurements from hardware. EnSim has a lot of the first two and little of the
third, so read the comparisons as evidence that the software is right, not that
the model is.

## Engine against NASA CEA

The reference is NASA's own `cea` 3.3.2 Python package, run by
`scripts/generate_cea_references.py` on the same problem EnSim solves: reactants
in their storage state, adiabatic equilibrium in the chamber, composition frozen
from there to the exit. The divergence loss is switched off for the comparison
because CEA has none.

<figure class="figure-svg">
--8<-- "docs/assets/figures/cea_deviation.svg"
<figcaption>Relative deviation of EnSim from CEA for chamber temperature, molar mass, characteristic velocity and vacuum Isp. Axis ±0.15 %. Drawn by <code>scripts/make_doc_figures.py</code>.</figcaption>
</figure>

| Case | O/F | pc, bar | ε | Tc, K (CEA) | Tc | c\* | Isp vac, s (CEA) | Isp |
|---|---|---|---|---|---|---|---|---|
| LH2/LOX | 6.0 | 68.95 | 40 | 3483.4 | +0.004 % | +0.007 % | 430.44 | −0.007 % |
| LCH4/LOX | 3.4 | 100 | 35 | 3598.3 | −0.007 % | −0.001 % | 343.25 | +0.000 % |
| RP-1/LOX | 2.6 | 100 | 25 | 3723.6 | +0.010 % | −0.049 % | 326.96 | −0.042 % |
| N2H4/NTO | 1.3 | 20 | 60 | 3143.4 | −0.001 % | +0.003 % | 327.75 | −0.002 % |
| MMH/NTO | 2.0 | 10 | 100 | 3170.4 | −0.004 % | +0.002 % | 322.62 | +0.000 % |

The test tolerances are 0.05 % on chamber temperature and 0.1 % on c\* and Isp. The RP-1 case is the loosest because CEA's
kerosene is a different mixture of the same formula, which shows as a 0.1 %
difference in molar mass.

Two gas-phase cases check the equilibrium solver alone, with reactants at
298.15 K:

| Case | pc | Tc, K (CEA) | Tc | γ frozen | Molar mass |
|---|---|---|---|---|---|
| 2 H2 + O2 | 68.9 bar | 3674.15 | −0.003 % | −0.016 % | −0.001 % |
| CH4 + 2 O2 | 100 bar | 3680.17 | −0.008 % | −0.002 % | −0.001 % |

Other tests in the same folder check element conservation, the enthalpy
balance, non-negative composition, the dissociation trend with pressure and the
choice of polynomial range.

## Flight against NASA check cases

NASA's Engineering and Safety Center published a set of six-degree-of-freedom
check cases, each flown by several independent simulation tools
(Jackson et al., NASA/TM-2015-218675). EnSim carries unchanged extracts of two
of them; the source files, columns and SHA-256 hashes are listed in
`tests/reference/README.md`.

| Case | What it tests | Compared | Tolerance |
|---|---|---|---|
| 1, dragless sphere | Translation over a rotating WGS-84 Earth with J2 gravity, from 30 000 ft | ECI position and velocity, altitude, longitude, latitude at 5 s intervals for 30 s | 0.002 ft, 0.0002 ft/s, 2·10⁻¹⁰ deg |
| 2, tumbling brick | Torque-free rotation of an asymmetric body | Euler angles and body rates at 5 s intervals for 30 s | 2·10⁻⁸ deg, deg/s |

Both pass. They exercise the Earth model, gravity, frames and rotational
dynamics; neither has aerodynamics or thrust.

## Flight model checks

Unit tests in `tests/unit` cover what the check cases do not:

- free fall against the closed form, and inverse-square gravity with altitude;
- the US Standard Atmosphere 1976 table at 0, 11, 20, 50 and 80 km, which
  only matches with the geopotential altitude conversion, and continuity above
  the 86 km model top;
- quaternion norm and energy and angular momentum in torque-free motion;
- RK4 and Dormand–Prince both converging to a fine-step reference through a
  powered flight, with the adaptive one at least as accurate;
- monotonic propellant depletion and the centre of gravity at liftoff;
- the restoring direction of the aerodynamic moment, and a stable vehicle
  turning into the wind;
- landing point, drogue and main deployment events, and the canopy fill time
  limiting the opening load.

## Other models

Cooling, acoustics, optimisation and the uncertainty studies have equation-level
tests: Bartz and Gnielinski against hand calculations, energy balance along the
channel, Bessel-root mode frequencies, growth only when rates are given, seeded
reproducibility, lognormal moments and the geometry of the confidence ellipse.
None of these models has been compared with hardware data.

## What is missing

1. Wall temperatures and coolant pressure drop from a hot-fire test of a fully
   documented chamber.
2. Shifting-equilibrium or finite-rate nozzle results across area ratios.
3. NESC case 3 or another aerodynamically coupled reference trajectory.
4. Telemetry from a documented flight.
5. Measured growth and damping rates for combustion stability.

If you have any of these and can share them, please open an issue.

## Sources

- Gordon and McBride, [NASA RP-1311 Part I](https://ntrs.nasa.gov/citations/19950013764), 1994.
- McBride, Zehe and Gordon, [NASA/TP-2002-211556](https://ntrs.nasa.gov/citations/20020085330), 2002.
- Jackson, Murri and Shelton, [NASA/TM-2015-218675](https://ntrs.nasa.gov/citations/20150001263), 2015.
- NASA Engineering and Safety Center, [Technical Bulletin 24-04](https://ntrs.nasa.gov/citations/20240013467), 2024.
