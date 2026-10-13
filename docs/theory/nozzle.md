# Nozzle expansion

The gas leaves the chamber with the composition it reached there and keeps it
to the exit plane. This is CEA's *frozen* rocket problem. It is one percent or
two pessimistic for hot, dissociated mixtures, where real recombination in the
nozzle returns some energy, and close to exact for cooler ones.

<figure class="figure-svg">
--8<-- "docs/assets/figures/nozzle.svg"
<figcaption>Stations used throughout: chamber (c), throat (t) and exit (e).</figcaption>
</figure>

## The isentrope

With the mole fractions $x_j$ fixed, entropy per unit mass is

$$
\frac{s(T,p)}{R/M}=\sum_j x_j\frac{s_j^\circ(T)}{R}-\sum_j x_j\ln x_j-\ln\frac{p}{p^\circ}.
$$

Holding it at the chamber value $s_c$ gives the pressure on the isentrope in
closed form:

$$
\ln\frac{p}{p^\circ}=\sum_j x_j\frac{s_j^\circ(T)}{R}-\sum_j x_j\ln x_j-\frac{s_c}{R/M}.
$$

So every state along the expansion is a function of temperature alone. The
velocity follows from the energy equation, $u=\sqrt{2\,[h(T_c)-h(T)]}$, and the
density from the ideal-gas law. Both $h$ and $c_p$ keep their full temperature
dependence; nothing is frozen except the composition.

## Throat and exit

The throat is where the mass flux $\rho u$ peaks along the isentrope, found by
a bounded one-dimensional search. Continuity then fixes the exit:

$$
\frac{A_e}{A_t}=\frac{(\rho u)_t}{(\rho u)_e},
$$

solved for the supersonic root. From the throat state,

$$
c^*=\frac{p_c}{(\rho u)_t},\qquad
C_F=\frac{u_e}{c^*}+\frac{(p_e-p_a)}{p_c}\frac{A_e}{A_t},\qquad
I_{sp}=\frac{C_F\,c^*}{g_0}.
$$

The difference from the textbook constant-$\gamma$ relations is not small. For
LCH4/LOX at O/F 3.4, 100 bar and area ratio 35:

| | Isp vacuum | Exit pressure |
|---|---|---|
| Constant $\gamma_f = 1.1955$ | 346.5 s | 25.2 kPa |
| EnSim, frozen with $c_p(T)$ | 343.25 s | 22.6 kPa |
| NASA CEA 3.3.2, frozen | 343.25 s | |

The gas cools from 3598 K to 1210 K in that nozzle and its $c_p$ falls with it;
a constant $\gamma$ misses that.

## Losses

Three corrections are applied, in this order:

- **Divergence.** A conical nozzle of half-angle $\alpha$ wastes the radial part
  of the momentum: the momentum term of $C_F$ is multiplied by
  $\lambda=(1+\cos\alpha)/2$. Set $\alpha=0$ for a contoured nozzle whose
  losses you prefer to put in $\eta_{C_F}$.
- **Combustion efficiency** $\eta_{c^*}$ scales $c^*$ and therefore the mass
  flow at a given throat area and chamber pressure.
- **Nozzle efficiency** $\eta_{C_F}$ scales the complete thrust coefficient,
  pressure term included.

Thrust is $F=C_F\,p_c\,A_t$ and mass flow $\dot m=p_cA_t/c^*$, both after the
efficiencies.

## Back pressure

Performance at an ambient pressure $p_a$ changes only the pressure term, so
`performance_at` re-uses the solved nozzle. EnSim does not model flow
separation: a strongly over-expanded nozzle is reported with the attached-flow
thrust, and the Engine page warns when $p_e/p_a$ falls below the Summerfield
estimate of 0.4. If $p_a$ exceeds the throat pressure the nozzle cannot run
choked and no result is returned.

## Wall contour

The drawn and exported wall is a cylinder, a 30° cone, an upstream throat arc
of $1.5R_t$, a downstream arc of $0.382R_t$ and a parabolic Rao bell at 80 % of
the length of a 15° cone, with entry and exit angles of 35° and 8°. The
cylinder length makes the volume up to the throat equal to $L^* A_t$. The bell
is a geometric approximation: its angles are fixed rather than optimised for
the area ratio, and the performance calculation does not depend on it.

## Planar MOC contour

The *MOC contour* tool designs a minimum-length nozzle by the method of
characteristics for a calorically perfect gas. Along the two families of
characteristics

$$
\theta+\nu=\text{const},\qquad \theta-\nu=\text{const},
$$

where $\nu(M)$ is the Prandtl–Meyer function. A centred expansion at a sharp
throat corner turns the wall to $\theta_\text{max}=\nu(M_e)/2$, and the
reflected characteristics bring it back to axial flow. The geometry is planar:
the reported coordinate is a half-height and $A_e/A_t=y_e/y_t$.

## Sources

- Gordon and McBride, NASA RP-1311 Part I (1994), section 6.
- G. V. R. Rao, "Approximation of Optimum Thrust Nozzle Contour",
  *ARS Journal* 30(6), 1960.
- D. K. Huzel and D. H. Huang, *Modern Engineering for Design of
  Liquid-Propellant Rocket Engines*, AIAA, 1992.
- Shames and Seashore, NACA RM E8J12 (1948); Goldman and Vanco,
  NASA TM X-1502 (1968).
