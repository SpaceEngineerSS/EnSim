# Combustion-instability analysis

## What the module computes

For an ideal cylindrical chamber, EnSim calculates longitudinal, tangential and
radial acoustic eigenfrequencies. Tangential/radial modes use roots of the
appropriate Bessel-function derivative; longitudinal modes use chamber length
and frozen sound speed.

This is an acoustic mode finder. A frequency is not, by itself, an instability.

## Stability classification

When compatible modal growth and damping rates are supplied, EnSim reports the
net rate

$$
\alpha_{net}=\alpha_{drive}-\alpha_{damp}.
$$

Positive net rate indicates exponential growth within that supplied linear
model; negative net rate indicates decay. If rates are absent, the result remains
“not assessed.” The software does not fabricate a universal threshold from
chamber pressure, L-star or injector count.

The low-frequency chug estimate uses the explicitly documented feed/compliance
inputs. It should be treated as a lumped-system screening result.

## Missing physics

Injector admittance, time lag, distributed combustion response, nonlinear limit
cycles, entropy/vorticity waves, nozzle damping, baffles, acoustic liners and
multiphase propellant dynamics are not derived from geometry automatically.

## References

- Harrje, D. T. and Reardon, F. H., [Liquid Propellant Rocket Combustion
  Instability](https://ntrs.nasa.gov/citations/19720015260), NASA SP-194, 1972.
- Culick, F. E. C., “Combustion Instabilities in Liquid-Fueled Propulsion
  Systems,” AGARD Conference Proceedings No. 450, 1988.
