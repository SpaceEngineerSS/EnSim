# EnSim Theory and Mathematical Background

This document provides the theoretical foundation for the EnSim rocket engine simulation.

## Table of Contents
1. [Thermodynamics](#thermodynamics)
2. [Chemical Equilibrium](#chemical-equilibrium)
3. [Propulsion Performance](#propulsion-performance)
4. [References](#references)

---

## Thermodynamics

### NASA 7-Term Polynomials

EnSim uses NASA Glenn thermodynamic coefficients to calculate species properties. The polynomial form is:

$$\frac{C_p}{R} = a_1 + a_2T + a_3T^2 + a_4T^3 + a_5T^4$$

$$\frac{H}{RT} = a_1 + \frac{a_2T}{2} + \frac{a_3T^2}{3} + \frac{a_4T^3}{4} + \frac{a_5T^4}{5} + \frac{a_6}{T}$$

$$\frac{S}{R} = a_1\ln T + a_2T + \frac{a_3T^2}{2} + \frac{a_4T^3}{3} + \frac{a_5T^4}{4} + a_7$$

Where:
- $R$ = Universal gas constant (8.3144621 J/mol·K)
- $a_1...a_7$ = NASA polynomial coefficients
- Two sets of coefficients: low-T (200-1000K) and high-T (1000-6000K)

---

## Chemical Equilibrium

### Gibbs Free Energy Minimization

The equilibrium composition is found by minimizing the total Gibbs free energy:

$$G = \sum_j n_j \left( \mu_j^{0} + RT \ln \frac{n_j P}{n_{total} P^{0}} \right)$$

Subject to element conservation:

$$\sum_j a_{ij} n_j = b_i \quad \forall \text{ elements } i$$

### Gordon-McBride Method

EnSim uses the Newton-Raphson iteration method described by Gordon & McBride (NASA RP-1311):

1. **Initialize**: Estimate species moles from stoichiometry
2. **Iterate**: Solve the linearized system:

$$\begin{bmatrix} \nabla^{2} G & A^{T} \\ A & 0 \end{bmatrix} \begin{bmatrix} \Delta n \\ \lambda \end{bmatrix} = \begin{bmatrix} -\nabla G \\ b - An \end{bmatrix}$$

3. **Convergence**: Check $\|\Delta n / n\| < 10^{-7}$

### Adiabatic Flame Temperature

Energy balance for adiabatic combustion:

$$\sum_j n_j H_j(T_{ad}) = \sum_i n_i^{0} H_i(T_0)$$

Solved iteratively with equilibrium composition at each temperature.

---

## Propulsion Performance

### Characteristic Velocity (C*)

$$C^{*} = \frac{\sqrt{\gamma R T_c}}{\Gamma}$$

where the vandenkerckhove function:

$$\Gamma = \sqrt{\gamma} \left( \frac{2}{\gamma + 1} \right)^{\frac{\gamma + 1}{2(\gamma - 1)}}$$

### Exit Velocity

From isentropic expansion:

$$V_e = \sqrt{\frac{2\gamma}{\gamma - 1} R T_c \left[ 1 - \left( \frac{P_e}{P_c} \right)^{\frac{\gamma - 1}{\gamma}} \right]}$$

### Thrust Coefficient

$$C_F = \Gamma \sqrt{\frac{2\gamma^2}{\gamma - 1} \left[ 1 - \left( \frac{P_e}{P_c} \right)^{\frac{\gamma - 1}{\gamma}} \right]} + \epsilon \frac{P_e - P_a}{P_c}$$

### Specific Impulse

$$I_{sp} = \frac{C^{*} \cdot C_F}{g_0}$$

### Area-Mach Relation

$$\frac{A}{A^{*}} = \frac{1}{M} \left[ \frac{2}{\gamma + 1} \left( 1 + \frac{\gamma - 1}{2} M^{2} \right) \right]^{\frac{\gamma + 1}{2(\gamma - 1)}}$$

### Efficiency Corrections

Real-world performance includes losses:

$$C^{*}_{actual} = \eta_{C^{*}} \cdot C^{*}_{ideal}$$

$$C_{F,actual} = \eta_{C_F} \cdot C_{F,ideal}$$

Typical values:
- $\eta_{C^{*}}$ = 0.94 - 0.99 (combustion efficiency)
- $\eta_{C_F}$ = 0.96 - 0.99 (nozzle efficiency)

---

## References

1. **Gordon, S. & McBride, B.J.** (1994). *Computer Program for Calculation of Complex Chemical Equilibrium Compositions and Applications - Part I: Analysis*. NASA Reference Publication 1311.

2. **McBride, B.J. & Gordon, S.** (1996). *Computer Program for Calculation of Complex Chemical Equilibrium Compositions and Applications - Part II: Users Manual*. NASA RP-1311.

3. **Sutton, G.P. & Biblarz, O.** (2016). *Rocket Propulsion Elements*, 9th Edition. John Wiley & Sons.

4. **NASA Glenn Research Center**. *Thermodynamic Data*. https://cearun.grc.nasa.gov/

5. **NIST Chemistry WebBook**. *Thermophysical Properties*. https://webbook.nist.gov/chemistry/
