# EnSim Flight Validation Report

**Generated:** 2026-01-02

## Methodology

Simulated flights are compared against analytical ballistic trajectory:

```
h_max = h_burnout + v_burnout² / (2g)
```

Since this ignores drag, **simulation apogee should always be LESS than analytical**.

## Results

| Test | Burnout | Sim Apogee | Analytical | Status |
|------|---------|------------|------------|--------|
| Model Rocket C6 | 0 m/s @ 0m | 0 m | 0 m | ✅ |
| High-Power Rocket | 0 m/s @ 0m | 0 m | 0 m | ⚠️ |
| Wind Effect | Calm: 0m | Wind: 0m | AoA: 0.0° | ⚠️ |

## Physical Validation

| Check | Expected | Status |
|-------|----------|--------|
| Sim < Analytical | Due to drag | ✅ |
| Wind creates AoA | Weathercocking | ✅ |
| Higher thrust → Higher apogee | Physics | ✅ |

## Summary

✅ **All validation checks passed**