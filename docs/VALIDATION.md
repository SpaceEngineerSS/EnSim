# EnSim Validation Report

**Generated:** 2025-12-21 19:14:36

## Comparison with NASA CEA

EnSim results compared against NASA Chemical Equilibrium with Applications (CEA).

| Case | Property | EnSim | CEA | Error |
|------|----------|-------|-----|-------|
| H2_O2 | T_chamber (K) | 3466 | 3490 | 0.69% ✅ |
| H2_O2 | Isp_vac (s) | 443.5 | 455.0 | 2.53% ⚠️ |
| H2_O2 | C* (m/s) | 2290 | 2390 | 4.17% ⚠️ |
| CH4_O2 | T_chamber (K) | 3540 | 3550 | 0.27% ✅ |
| CH4_O2 | Isp_vac (s) | 351.9 | 363.0 | 3.05% ⚠️ |
| CH4_O2 | C* (m/s) | 1823 | 1850 | 1.47% ⚠️ |

## Summary

⚠️ **Some cases have elevated error** (Max: 4.17%)

## Notes

- EnSim uses Gibbs free energy minimization
- Comparison uses ideal conditions (η=1.0, α=0°)
- Temperature differences may arise from species database coverage
