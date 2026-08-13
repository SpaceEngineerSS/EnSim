# Regenerative-cooling model

## Scope

The cooling module estimates steady one-dimensional wall heat transfer and
coolant pressure loss along a supplied nozzle contour. It is intended for early
channel sizing and sensitivity studies.

Channel count, width, height, land width, hot-wall thickness, cooled length,
coolant mass flow and inlet state are explicit inputs. EnSim does not infer a
channel design from thrust or impose an undocumented heat-flux target. The GUI
uses a stated three-point converging-diverging contour for this standalone
screening analysis; importing a detailed contour is not yet supported.

## Gas-side convection

EnSim implements the standard Bartz correlation using chamber pressure,
characteristic velocity, throat diameter, local area ratio, gas viscosity,
specific heat, Prandtl number and the property-variation factor. Molecular weight
and gamma come from the engine calculation when invoked through the GUI.

The correlation was derived for rocket thrust chambers and is sensitive to the
choice of reference properties. NASA comparisons have documented substantial
variation between Bartz-type estimates and measurements, particularly away from
the throat; it is a correlation, not a universal heat-flux law.

## Coolant side and pressure loss

The hydraulic diameter is computed from channel geometry. For turbulent flow,
the coolant coefficient uses Gnielinski and smooth-channel Darcy friction uses a
logarithmic correlation. Invalid Reynolds/Prandtl domains are rejected rather
than silently switching fluids or correlations. Coolant bulk temperature is
marched counter to the gas flow and the energy rise uses the actual channel mass
flow.

The lower-level reduced thermal-profile API accepts contraction ratio and both
conical half-angles explicitly. It is retained for equation-level studies; the
desktop Cooling workspace uses the channel-resolved analysis described above.

## Thermal resistance

At each station, gas convection, wall conduction and coolant convection are
combined in series. Reported hot-wall and coolant-wall temperatures are the
steady solution of that local network. Differences from the nominal material
melting point and coolant critical temperature are diagnostic temperature
differences, not certified safety margins or phase-stability predictions.

## Not represented

Boiling, supercritical property tables, film cooling, rib roughness, manifold
maldistribution, coking, conjugate axial conduction, radiation, transient thermal
stress, creep and fatigue are outside the current model.

## References

- Bartz, D. R., “A Simple Equation for Rapid Estimation of Rocket Nozzle
  Convective Heat Transfer Coefficients,” *Jet Propulsion*, 27(1), 1957.
- Smith, T. D., [A Comparison of Techniques for Predicting Local Heat-Transfer
  Coefficients for Rocket Engines](https://ntrs.nasa.gov/api/citations/19750020175/downloads/19750020175.pdf), NASA TM X-71817, 1975.
- Brown, A. M. et al., [RL10A-3-3A Rocket Engine Modeling Project](https://ntrs.nasa.gov/api/citations/19970010379/downloads/19970010379.pdf), NASA/CR-198538, 1996.
- Gnielinski, V., “New Equations for Heat and Mass Transfer in Turbulent Pipe and
  Channel Flow,” *International Chemical Engineering*, 16, 1976.
