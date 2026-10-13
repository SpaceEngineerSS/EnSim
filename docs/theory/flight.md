# Flight dynamics

The vehicle is one rigid body with a varying mass. Its state has fourteen
numbers: position, velocity, an attitude quaternion, body angular rates and the
propellant left on board.

<figure class="figure-svg">
--8<-- "docs/assets/figures/frames.svg"
<figcaption>Body axes and the stability lever arm. A centre of pressure behind the centre of gravity gives a restoring moment.</figcaption>
</figure>

## Equations of motion

$$
\dot{\mathbf r}=\mathbf v,\qquad
\dot{\mathbf v}=\frac{R(q)\,(\mathbf F_T+\mathbf F_A)}{m}+\mathbf g(\mathbf r),
$$

$$
\dot q=\tfrac12\,q\otimes(0,\boldsymbol\omega),\qquad
I\dot{\boldsymbol\omega}=\mathbf M-\boldsymbol\omega\times I\boldsymbol\omega,\qquad
\dot m_p=-\dot m.
$$

$R(q)$ turns body axes into the integration frame. Every stage of the
Runge–Kutta step re-evaluates thrust, aerodynamics, mass, centre of gravity,
inertia and gravity from that stage's own state, so the integrator delivers
the order it promises. Two integrators are available: classical RK4 with a
fixed step, and Dormand–Prince 5(4) with step-size control.

## Frames and gravity

By default the Earth is flat and still, the frame is local east–north–up at
the pad, and gravity falls off as $g_0\,[R_E/(R_E+h)]^2$. For a sounding rocket
that lands a few kilometres away this is accurate to well below the other
uncertainties.

*Round, rotating Earth* integrates in an Earth-centred inertial frame instead.
Gravity includes the $J_2$ oblateness term, the pad sits on the WGS-84
ellipsoid at the given latitude and longitude, and the air moves with the
rotating Earth, so the vehicle feels the correct relative wind and the
Coriolis effect appears without being added by hand. Results are reported back
in the pad's east–north–up axes.

## Mass properties

Structure, engine and tanks are fixed masses at their stations. Propellant
drains from the tank's centre, so the centre of gravity moves forward during
the burn. The moments of inertia are those of a uniform solid cylinder with the
vehicle's length, radius and current mass, which is enough for the slow
attitude motion of a stable rocket.

## Thrust

While propellant remains, thrust acts along the body axis:

$$
F_T=F_\text{vac}-p_a(h)\,A_e,
$$

with mass flow $\dot m=F_\text{vac}/(I_{sp,\text{vac}}g_0)$. The vehicle stays on
the pad until thrust exceeds the weight component along the rail; if it never
does, the run stops with the two numbers in the message.

## Aerodynamics

Forces come from the air-relative velocity, including wind, and the US
Standard Atmosphere 1976:

- **Axial drag** $D=\bar q\,A_\text{ref}\,C_D$ with the constant $C_D$ from the
  Vehicle page, plus the drag area of any open parachute.
- **Normal force** from Barrowman's method: nose and fin normal-force
  derivatives summed to $C_{N\alpha}$, acting at the combined centre of pressure.
  It produces the restoring moment about the centre of gravity,
  $\mathbf M=(\mathbf x_{cg}-\mathbf x_{cp})\times\mathbf F_N$.
- **Fin misalignment** adds a small constant normal force at the fins'
  centre of pressure, a trim moment that bends the trajectory slightly.

$C_D$ does not change with Mach number. For transonic flights, use the
average of the drag curve over the speeds the vehicle spends most time at,
or tune it to a flown trajectory.

## Wind

Wind is given at 10 m and grows with height as a one-seventh power law up to
1 km, constant above that. The direction is meteorological, the direction the
wind blows from.

## Launch rail

On the rail the attitude is fixed and the velocity is projected onto the rail
direction after every step. Because that constraint acts between steps, steps on
the rail are limited to a tenth of the normal step. The vehicle leaves the rail
when it has travelled the rail length.

## Events and recovery

Apogee and impact are located inside a step by Hermite interpolation of the
solution, not by stepping until the sign changes, so their times do not depend
on the step size.

Recovery is single or dual deploy. In dual deploy the drogue opens at apogee
and the main at a set height above the pad; with single deploy the main opens
at apogee or at the set height. A canopy does not open instantly: its drag area
grows linearly over the fill time (1 s by default), which keeps the opening
load finite.

<figure class="figure-svg">
--8<-- "docs/assets/figures/timeline.svg"
<figcaption>Flight phases and the events between them.</figcaption>
</figure>

## Stages

The Stages view on the Vehicle page computes ideal delta-v with the rocket
equation and burn times from per-engine thrust, scaled between sea level and
vacuum with the US76 pressure. It is a sizing tool. Staged vehicles are not
flown in six degrees of freedom.

## Sources

- B. L. Stevens, F. L. Lewis and E. N. Johnson, *Aircraft Control and
  Simulation*, 3rd ed., Wiley, 2016.
- P. H. Zipfel, *Modeling and Simulation of Aerospace Vehicle Dynamics*,
  3rd ed., AIAA, 2014.
- J. S. Barrowman and J. A. Barrowman, *The Theoretical Prediction of the
  Center of Pressure*, NARAM-8, 1966.
- *U.S. Standard Atmosphere, 1976*, NOAA/NASA/USAF.
- E. B. Jackson et al., *Assessment of Flight Simulation Accuracy and
  Consistency*, NASA/TM-2015-218675.
