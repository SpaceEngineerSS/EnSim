# Theory and governing equations

This document describes the equations represented in EnSim and, equally
important, the assumptions that close them.

## Thermodynamic properties

For the seven-coefficient NASA form used by the packaged data,

$$
\frac{C_p^\circ}{R}=a_1+a_2T+a_3T^2+a_4T^3+a_5T^4,
$$

$$
\frac{H^\circ}{RT}=a_1+\frac{a_2T}{2}+\frac{a_3T^2}{3}
+\frac{a_4T^3}{4}+\frac{a_5T^4}{5}+\frac{a_6}{T},
$$

$$
\frac{S^\circ}{R}=a_1\ln T+a_2T+\frac{a_3T^2}{2}
+\frac{a_4T^3}{3}+\frac{a_5T^4}{4}+a_7.
$$

The parser selects the coefficient interval declared for each species and
rejects unavailable species. The model is ideal gas and does not calculate
fugacity or high-pressure real-fluid injection states.

## Chemical equilibrium

At fixed temperature and pressure, the gas composition minimizes

$$
G=\sum_j n_j\left[g_j^\circ(T)+RT\ln\left(\frac{n_j}{n}
\frac{P}{P^\circ}\right)\right]
$$

subject to elemental conservation

$$
\sum_j a_{ij}n_j=b_i,\qquad n_j\ge 0.
$$

The chamber problem is adiabatic. EnSim therefore iterates temperature until
reactant and product enthalpy agree while resolving equilibrium at every trial
temperature. Species selection bounds the physical solution: an omitted species
cannot appear, even if thermodynamically favorable.

## Frozen ideal-nozzle performance

The chamber mixture is treated as calorically perfect during nozzle expansion;
`gamma` and molar mass are frozen at the chamber solution. The characteristic
velocity is

$$
c^*=\frac{\sqrt{R_sT_c}}
{\sqrt{\gamma}\left(2/(\gamma+1)\right)^{(\gamma+1)/(2(\gamma-1))}}.
$$

The supersonic exit Mach number is obtained from

$$
\frac{A_e}{A_t}=\frac{1}{M_e}
\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M_e^2\right)
\right]^{(\gamma+1)/(2(\gamma-1))}.
$$

Then

$$
\frac{P_e}{P_c}=\left(1+\frac{\gamma-1}{2}M_e^2\right)^{-\gamma/(\gamma-1)},
$$

$$
C_F=\sqrt{\frac{2\gamma^2}{\gamma-1}
\left(\frac{2}{\gamma+1}\right)^{(\gamma+1)/(\gamma-1)}
\left[1-\left(\frac{P_e}{P_c}\right)^{(\gamma-1)/\gamma}\right]}
+\frac{(P_e-P_a)A_e}{P_cA_t},
$$

and `Isp = C_F c* / g0`. For a conical nozzle, the momentum component of
`C_F` is multiplied by `(1 + cos(alpha))/2`; the user-supplied nozzle-efficiency
factor is then applied to the complete coefficient. Boundary-layer loss,
finite-rate chemistry, two-phase flow and separated-flow thrust are outside this
calculation.

## Planar minimum-length nozzle

The advanced contour tool solves the two-dimensional compatibility relations
for an irrotational, calorically perfect supersonic flow. Along the two
characteristic families,

$$K_- = \theta + \nu = \mathrm{constant},\qquad
K_+ = \theta - \nu = \mathrm{constant},$$

where $\nu(M)$ is the Prandtl-Meyer function and
$\mu=\sin^{-1}(1/M)$ is the Mach angle. A centered expansion begins at a sharp
throat corner with $\theta_{w,\max}=\nu(M_e)/2$; reflected characteristics then
turn the wall back to axial flow. The plotted transverse coordinate is a planar
half-height. Consequently, $A_e/A_t=y_e/y_t$, not the squared radius ratio used
for an axisymmetric nozzle. Characteristic count controls discretization only;
the model does not include axisymmetric source terms, throat rounding, viscosity
or variable thermochemistry.

## Regenerative cooling

Gas-side convection uses the standard Bartz correlation with throat diameter,
local area ratio, chamber pressure, characteristic velocity, gas properties and
the Bartz property-variation factor. Coolant convection uses Gnielinski in its
turbulent validity range; smooth-channel Darcy friction uses a corresponding
correlation. Wall conduction and the two convective resistances form a local
thermal-resistance network. See [Cooling model](COOLING.md).

## Chamber acoustics

Longitudinal, tangential and radial acoustic frequencies use ideal cylindrical
cavity eigenvalues. Frequency proximity alone is not a growth model. EnSim only
classifies a mode when the user supplies compatible driving and damping rates.
See [Combustion instability](COMBUSTION_INSTABILITY.md).

## Flight dynamics

The rigid-body translational and rotational equations are

$$
\dot{\mathbf r}=\mathbf v,\qquad
\dot{\mathbf v}=\frac{\mathbf F}{m}+\mathbf g,
$$

$$
\mathbf I\dot{\boldsymbol\omega}
=\mathbf M-\boldsymbol\omega\times(\mathbf I\boldsymbol\omega).
$$

A unit quaternion maps body coordinates to the propagation frame. The WGS-84
path evaluates gravity in ECEF, including axisymmetric J2, and accounts for Earth
rotation when converting atmospheric velocity. The standard aerodynamic model
uses dynamic pressure, air-relative velocity and preliminary Barrowman
normal-force and center-of-pressure derivatives. Axial drag uses the explicit
coefficient supplied by the user, referenced to body frontal area; EnSim does
not synthesize a Mach-dependent drag polar.

## Uncertainty propagation

Positive quantities such as pressure and area ratio use lognormal factors;
bounded quantities use explicitly truncated samples. Sample standard deviations
use `N-1`. A two-dimensional normal confidence ellipse is scaled with the
chi-square quantile for two degrees of freedom. These assumptions describe the
chosen input model, not epistemic certainty about a real engine or vehicle.

## Primary sources

- Gordon and McBride, NASA RP-1311 Part I (1994) and Part II (1996).
- McBride, Zehe and Gordon, NASA/TP-2002-211556 (2002).
- Bartz, *Jet Propulsion*, Vol. 27, No. 1 (1957), pp. 49-51.
- Gnielinski, *International Chemical Engineering*, Vol. 16 (1976), pp. 359-368.
- Jackson, Murri and Shelton, NASA/TM-2015-218675 (2015).
- Barrowman and Barrowman, *The Theoretical Prediction of the Center of Pressure*,
  NARAM-8, 1966.
- Shames and Seashore, *Design Data for Graphical Construction of
  Two-Dimensional Sharp-Edge-Throat Supersonic Nozzles*, NACA RM E8J12 (1948).
- Goldman and Vanco, *Computer Program for Design of Two Dimensional Supersonic
  Nozzle with Sharp Edged Throat*, NASA TM X-1502 (1968).
