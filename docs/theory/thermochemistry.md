# Chamber thermochemistry

The chamber is treated the way NASA CEA treats it: an adiabatic, constant-pressure
reactor that reaches chemical equilibrium before the gas enters the nozzle.
Two numbers come out of it that everything downstream depends on, the flame
temperature and the product composition.

<figure class="figure-svg">
--8<-- "docs/assets/figures/solver.svg"
<figcaption>From propellants to performance. The chamber problem (left) is solved once; the nozzle (right) reuses its composition.</figcaption>
</figure>

## Species data

Each species carries NASA Glenn seven-coefficient polynomials in two
temperature ranges:

$$
\frac{c_p^\circ}{R}=a_1+a_2T+a_3T^2+a_4T^3+a_5T^4
$$

$$
\frac{h^\circ}{RT}=a_1+\frac{a_2T}{2}+\frac{a_3T^2}{3}+\frac{a_4T^3}{4}+\frac{a_5T^4}{5}+\frac{a_6}{T}
$$

$$
\frac{s^\circ}{R}=a_1\ln T+a_2T+\frac{a_3T^2}{2}+\frac{a_4T^3}{3}+\frac{a_5T^4}{4}+a_7
$$

The standard state is 1 bar, as in the Glenn database. The parser reads the
range limits from each record and refuses a malformed one rather than guessing.

## Reactants

Propellants enter at their storage state. Liquid hydrogen at 20.27 K carries
−4.47 MJ/kg relative to the gas at 298 K; that energy is spent warming and
evaporating the propellant and never reaches the flame. The packaged table uses
the enthalpies CEA assigns to its liquid reactants:

| Key | Formula | Storage temperature | Enthalpy, kJ/kg |
|---|---|---|---|
| LH2 | H2 | 20.27 K | −4470.5 |
| LCH4 | CH4 | 111.64 K | −5562.3 |
| RP-1 | CH1.9423 | 298.15 K | −1768.6 |
| N2H4 | N2H4 | 298.15 K | +1572.2 |
| MMH | CH6N2 | 298.15 K | +1176.4 |
| UDMH | C2H8N2 | 298.15 K | +813.7 |
| ETHANOL | C2H6O | 298.15 K | −6023.9 |
| LOX | O2 | 90.17 K | −405.6 |
| NTO | N2O4 | 298.15 K | −190.7 |
| HTP | H2O2 | 298.15 K | −5520.6 |
| N2O | N2O (vapour) | 298.15 K | +1854.0 |

The difference matters. LH2/LOX at O/F 6, 70 bar and area ratio 60 burns at
3485 K with a vacuum Isp of 436.9 s. Treated as gases at room temperature, the
same mixture gives 3591 K and 445.9 s, two percent too optimistic. EnSim 3.x did
exactly that.

## Equilibrium

At temperature $T$ and pressure $p$ the product mixture minimises the Gibbs
energy

$$
G=\sum_j n_j\left[\frac{g_j^\circ(T)}{RT}+\ln\frac{n_j}{n}+\ln\frac{p}{p^\circ}\right]
$$

subject to the element balance $\sum_j a_{ij}n_j=b_i$ and $n_j\ge 0$. The
candidate products are every gas in the database made only of the elements the
reactants bring in.

The minimum is found in two passes. SLSQP works on the mole numbers with the
element balance as a linear constraint; a least-squares polish then solves the
optimality conditions in element potentials, which drives trace species to the
precision CEA reports. A linear program supplies a feasible starting point.

## Flame temperature

The chamber is adiabatic, so the products must carry the reactants' enthalpy:

$$
H_\text{products}(T_c)=H_\text{reactants}.
$$

The residual is evaluated with a full equilibrium at each trial temperature. The
search starts from an estimate, steps outward by 400 K until the residual
changes sign and then closes the bracket with Brent's method. Every step solves
a new composition, so dissociation is accounted for at the temperature it
happens.

## What the nozzle receives

The chamber result carries the temperature, the mole fractions, the molar mass
and the frozen ratio of specific heats,

$$
\gamma_f=\frac{c_p}{c_p-R/M},\qquad c_p=\sum_j x_j c_{p,j}^\circ(T_c)/M.
$$

$\gamma_f$ is reported for reference and used by the simpler studies (sweeps of
nozzle shape, uncertainty, the MOC contour). The engine result itself does not
use a constant $\gamma$: see [nozzle expansion](nozzle.md).

## Agreement with CEA

For the five liquid propellant pairs in the validation set (hydrolox,
methalox, kerolox and two hypergolic pairs), chamber temperature agrees with
NASA CEA to within 0.01 % and vacuum Isp to within 0.05 %. The comparison and how to repeat it are on the
[validation](../validation.md) page.

## Sources

- S. Gordon and B. J. McBride, *Computer Program for Calculation of Complex
  Chemical Equilibrium Compositions and Applications*, NASA RP-1311, Part I
  (1994) and Part II (1996).
- B. J. McBride, M. J. Zehe and S. Gordon, *NASA Glenn Coefficients for
  Calculating Thermodynamic Properties of Individual Species*,
  NASA/TP-2002-211556 (2002).
