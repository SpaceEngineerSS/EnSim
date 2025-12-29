# EnSim Documentation

## Scientific References

### Thermodynamics
- NASA Glenn Coefficients Database
- NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/)
- JANAF Thermochemical Tables

### Chemical Equilibrium
- Gordon, S. & McBride, B.J. (1994). *NASA RP-1311: Computer Program for Calculation of Complex Chemical Equilibrium Compositions and Applications.*
- McBride, B.J. & Gordon, S. (1996). *NASA RP-1311 Part II: Users Manual and Program Description.*

### Rocket Propulsion
- Sutton, G.P. & Biblarz, O. (2016). *Rocket Propulsion Elements*, 9th Edition. Wiley.
- Humble, R.W., Henry, G.N., & Larson, W.J. (1995). *Space Propulsion Analysis and Design*. McGraw-Hill.

## Validation Reports

### H2O Thermodynamic Properties
- Temperature: 1000 K
- Cp (EnSim): 41.2947 J/(mol·K)
- Cp (NIST): 41.294 J/(mol·K)
- Error: <0.01%

### H2/O2 Combustion Products (10 atm, stoichiometric)
| Species | EnSim | NASA CEA |
|---------|-------|----------|
| T_ad | 3600 K | 3516 K |
| H2O | 68.1% | 56.7% |
| OH | 9.9% | 12.2% |
| H | 4.1% | 6.7% |

### Vacuum Isp (ε=50, Pc=68 bar)
- EnSim: 414.6 s
- Theoretical: ~420 s
- Error: 1.3%
