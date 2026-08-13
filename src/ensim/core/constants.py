"""
Physical constants for rocket propulsion calculations.

All constants are defined with high precision to match NASA CEA outputs.
References:
    - CODATA 2018 recommended values
    - NASA CEA source code (cea2.f)
    - NIST Chemistry WebBook
"""

from typing import Final

# Universal Gas Constant (J/(mol·K))
# CODATA 2018 exact value
# Reference: https://physics.nist.gov/cgi-bin/cuu/Value?r
GAS_CONSTANT: Final[float] = 8.31446261815324

# NASA CEA uses this value in J/(kmol·K) internally
# R_CEA = 8314.46261815324 J/(kmol·K)
NASA_R: Final[float] = 8314.46261815324

# Standard gravitational acceleration (m/s²)
# Used for Isp calculations: Isp = F / (m_dot * g0)
# Reference: NIST standard gravity
G0: Final[float] = 9.80665

# Standard temperature (K) for thermodynamic reference
T_REF: Final[float] = 298.15

# Standard pressure (Pa) for thermodynamic reference
P_REF: Final[float] = 101325.0

# Boltzmann constant (J/K)
# Reference: CODATA 2018
K_BOLTZMANN: Final[float] = 1.380649e-23

# Avogadro's number (1/mol)
# Reference: CODATA 2018
N_AVOGADRO: Final[float] = 6.02214076e23

# Planck constant (J·s)
# Reference: CODATA 2018
H_PLANCK: Final[float] = 6.62607015e-34
