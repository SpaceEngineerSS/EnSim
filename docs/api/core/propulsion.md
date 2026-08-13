# Propulsion API

```python
from ensim.core.propulsion import NozzleConditions, calculate_performance

nozzle = NozzleConditions(
    area_ratio=40.0,
    chamber_pressure=6.89e6,
    ambient_pressure=0.0,
)
result = calculate_performance(
    T_chamber=3672.0,
    P_chamber=6.89e6,
    gamma=1.1933,
    mean_molecular_weight=15.7644,
    nozzle=nozzle,
)

print(result.c_star, result.c_f, result.isp, result.exit_pressure)
```

The solver obtains the supersonic exit Mach number from the area-Mach relation
and evaluates the ideal attached-flow pressure-thrust term at the requested
ambient pressure. `mean_molecular_weight` is g/mol; other inputs use SI.

The result is frozen-composition and calorically perfect. It does not include
boundary-layer, separated-flow, divergence, two-phase or finite-rate chemistry
losses. See [model limitations](../../MODEL_LIMITATIONS.md).
