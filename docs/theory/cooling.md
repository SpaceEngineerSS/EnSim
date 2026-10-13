# Regenerative cooling

The cooling study answers an early design question: with these channels and
this much coolant, how hot does the wall run, and how much pressure does the
coolant lose? It is one-dimensional and steady, and meant for sizing channels.
It is not a thermal certification of a chamber.

<figure class="figure-svg">
--8<-- "docs/assets/figures/cooling.svg"
<figcaption>One station of the wall. Heat crosses three resistances in series; the coolant warms as it moves from station to station.</figcaption>
</figure>

## Along the wall

The wall is the chamber contour from the last engine run, divided into
stations. At each station the local area ratio gives a Mach number from the
one-dimensional relation with the chamber $\gamma_f$, subsonic upstream of the
throat and supersonic downstream. The gas drives heat into the wall at its
recovery temperature,

$$
T_{aw}=T\left(1+r\,\frac{\gamma-1}{2}M^2\right),\qquad r=Pr^{1/3},
$$

which sits just below the chamber temperature everywhere.

## Gas side: Bartz

$$
h_g=\frac{0.026}{D_t^{0.2}}\left(\frac{\mu^{0.2}c_p}{Pr^{0.6}}\right)
\left(\frac{p_c}{c^*}\right)^{0.8}\left(\frac{D_t}{R_c}\right)^{0.1}
\left(\frac{A_t}{A}\right)^{0.9}\sigma
$$

with the property correction

$$
\sigma=\left[\frac12\frac{T_{wg}}{T_c}\left(1+\frac{\gamma-1}{2}M^2\right)+\frac12\right]^{-0.68}
\left(1+\frac{\gamma-1}{2}M^2\right)^{-0.12}.
$$

$R_c$ is the throat radius of curvature, $1.5R_t$ for the EnSim contour. The
transport properties are the estimates Bartz recommends when nothing better is
known: Eucken's Prandtl number $Pr=4\gamma/(9\gamma-5)$ and
$\mu=1.184\times10^{-7}M^{0.5}T_c^{0.6}$ Pa·s with $M$ in g/mol. Both can be
supplied directly through the library.

$\sigma$ depends on the wall temperature, which depends on the heat flux, which
depends on $h_g$. Each station therefore iterates until the gas-side wall
temperature changes by less than 0.01 K; it stops with an error after 50 passes
rather than returning an unconverged number.

## Coolant side: Gnielinski

Coolant properties are taken at the inlet state. The hydraulic diameter comes
from the channel width and height, and

$$
Nu=\frac{(f/8)(Re-1000)Pr}{1+12.7\sqrt{f/8}\,(Pr^{2/3}-1)},\qquad f=(0.79\ln Re-1.64)^{-2}
$$

for $Re\ge3000$. Below 2300 the flow is laminar and $Nu=3.66$; in between the
two are blended linearly. Laminar coolant is almost always a design error: the
coefficient drops by an order of magnitude and the wall overheats. The desktop
study warns when it happens, and its default channel count is as many channels
as fit around the throat at the chosen width and land, which keeps the flow
turbulent for engines of any size.

Pressure loss uses the smooth-channel Darcy factor
$f=[-1.8\log_{10}(6.9/Re)]^{-2}$ (Haaland) and $64/Re$ in laminar flow.

## Wall and coolant march

The three resistances add:

$$
q=\frac{T_{aw}-T_{cool}}{1/h_g+t_w/k+1/h_c}.
$$

The gas-side and coolant-side wall temperatures follow from $q$. Moving to the
next station, the coolant absorbs $q$ over the wetted perimeter and the station
length,

$$
\Delta T_{cool}=\frac{q\,\pi D\,\Delta x}{\dot m_c\,c_{p,c}},
$$

and loses $f\,(\Delta x/D_h)\,\tfrac12\rho v^2$ of pressure. The coolant can
enter at the exit and run toward the injector (the usual arrangement) or the
other way.

## Reading the result

*Melt margin* is the difference between the material's melting point and
the hottest gas-side wall temperature. *Margin to critical temperature* compares
the coolant with its critical temperature. Both are temperature differences
for orientation. They are not safety factors: the wall loses strength long
before it melts, and a coolant near its critical point changes its properties
far more than the constant-property model assumes.

## Not modelled

Boiling and supercritical property variation, film cooling, rough or ribbed
channels, curvature effects on the coolant side, manifold maldistribution,
coking, axial conduction in the wall, radiation, and thermal stress.

## Sources

- D. R. Bartz, "A Simple Equation for Rapid Estimation of Rocket Nozzle
  Convective Heat Transfer Coefficients", *Jet Propulsion* 27(1), 1957.
- V. Gnielinski, "New Equations for Heat and Mass Transfer in Turbulent Pipe
  and Channel Flow", *International Chemical Engineering* 16, 1976.
- T. D. Smith, *A Comparison of Techniques for Predicting Local Heat-Transfer
  Coefficients for Rocket Engines*, NASA TM X-71817, 1975.
- Huzel and Huang, *Modern Engineering for Design of Liquid-Propellant Rocket
  Engines*, AIAA, 1992, chapter 4.
