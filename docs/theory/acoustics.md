# Chamber acoustics

High-frequency combustion instability grows when heat release couples to an
acoustic mode of the chamber. EnSim finds the modes. Whether a mode will grow
depends on the injector and the flame, which no geometry-only model can
predict, so EnSim only says *stable* or *unstable* when you supply the driving
and damping rates.

## Modes of a cylinder

The chamber is a closed cylinder of length $L$ and diameter $D$ filled with gas
at the chamber sound speed

$$
a=\sqrt{\gamma_f\,\frac{R}{M}\,T_c}.
$$

Longitudinal modes are those of a pipe closed at both ends,

$$
f_{nL}=\frac{n\,a}{2L},
$$

and transverse (tangential and radial) modes follow from the zeros of the
Bessel derivative $J_m'$:

$$
f_{mn}=\frac{\alpha_{mn}\,a}{\pi D},\qquad \alpha_{10}=1.8412\ (1T),\ \alpha_{20}=3.0542\ (2T),\ \alpha_{01}=3.8317\ (1R).
$$

For the 366 kN methalox engine on the [Python API](../library/python.md) page,
the contour chamber is 0.331 m long and 0.319 m across, the sound speed is
1282 m/s, and the lowest modes are 1L at 1936 Hz and 1T at 2354 Hz. The first
tangential mode is the one that has destroyed the most engines, and it is the
one baffles and acoustic cavities are tuned against.

## Growth

When you supply a driving rate $\alpha_d$ and a damping rate $\alpha_\delta$
for a mode (from test data or a separate model, in 1/s), EnSim reports

$$
\alpha_\text{net}=\alpha_d-\alpha_\delta.
$$

A positive value means the mode grows in that linear model, a negative one that
it decays. Without the rates the result stays *not assessed*. EnSim does not
derive a threshold from chamber pressure, L\* or injector count.

## Chugging

Low-frequency chug couples the feed system to the chamber. A lumped
inertance–compliance estimate gives its frequency:

$$
f_\text{chug}=\frac{1}{2\pi}\sqrt{\frac{A}{\rho\,L\,C}}
$$

with feed-line area $A$, length $L$, propellant density $\rho$ and the
combined compliance $C$ of feed system and chamber in m³/Pa. Treat it as a
screening number for where to look in test data.

## Not modelled

Injector admittance and time lag, the flame's response function, nozzle
damping, baffles and acoustic liners, non-linear limit cycles, and the
entropy and vorticity waves that matter for low-frequency coupling.

## Sources

- D. T. Harrje and F. H. Reardon (eds.), *Liquid Propellant Rocket Combustion
  Instability*, NASA SP-194, 1972.
- F. E. C. Culick, "Combustion Instabilities in Liquid-Fueled Propulsion
  Systems", AGARD CP-450, 1988.
