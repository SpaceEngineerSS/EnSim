# Theory & Mathematics

This section covers the theoretical foundations of EnSim's physics engine.

## Overview

EnSim implements rigorous mathematical models based on established aerospace engineering principles:

1. **Thermodynamics**: NASA 7-term polynomial formulation
2. **Chemical Equilibrium**: Gibbs free energy minimization (Gordon-McBride)
3. **Nozzle Flow**: 1-D isentropic compressible flow
4. **Flight Dynamics**: 6-DOF rigid body equations with quaternion orientation

## Topics

<div class="grid cards" markdown>

-   :material-fire:{ .lg .middle } __Thermodynamics__

    ---

    NASA polynomial property evaluation and mixture calculations.

    [:octicons-arrow-right-24: Thermodynamics](thermodynamics.md)

-   :material-atom:{ .lg .middle } __Chemical Equilibrium__

    ---

    Gibbs minimization and the Gordon-McBride method.

    [:octicons-arrow-right-24: Equilibrium](equilibrium.md)

-   :material-waves:{ .lg .middle } __Nozzle Flow__

    ---

    Isentropic flow relations and performance equations.

    [:octicons-arrow-right-24: Nozzle Flow](nozzle-flow.md)

-   :material-rocket:{ .lg .middle } __Flight Dynamics__

    ---

    6-DOF equations of motion and numerical integration.

    [:octicons-arrow-right-24: Flight Dynamics](flight-dynamics.md)

</div>

## Key Equations

### Characteristic Velocity

$$
C^* = \frac{\sqrt{\gamma R_u T_c / M_w}}{\sqrt{\gamma} \left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{2(\gamma-1)}}}
$$

### Specific Impulse

$$
I_{sp} = \frac{C^* \cdot C_F}{g_0}
$$

### Adiabatic Flame Temperature

Solved iteratively from energy balance:

$$
H_{reactants}(T_{in}) = H_{products}(T_{ad})
$$

## References

- Gordon, S. & McBride, B.J. (1994). NASA RP-1311
- Sutton, G.P. & Biblarz, O. (2017). Rocket Propulsion Elements, 9th Ed.
- Anderson, J.D. (2003). Modern Compressible Flow, 3rd Ed.

