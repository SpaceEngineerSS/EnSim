"""Physical constants used across EnSim.

Every module takes its constants from here so a value only has to be
checked once. Sources: CODATA 2018, NIMA TR8350.2 (WGS-84) and
NASA/TP-2002-211556 for the thermodynamic reference state.
"""

from typing import Final

# Thermochemistry
GAS_CONSTANT: Final[float] = 8.31446261815324
"""Universal gas constant, J/(mol K). Exact since the 2019 SI redefinition."""

NASA_R: Final[float] = 8314.46261815324
"""Universal gas constant per kmol, J/(kmol K). Divide by molar mass in kg/kmol to get R_specific."""

T_REF: Final[float] = 298.15
"""Thermodynamic reference temperature, K."""

P_REF: Final[float] = 1.0e5
"""Standard-state pressure, Pa. The NASA Glenn coefficients are referenced to 1 bar."""

P_ATM: Final[float] = 101325.0
"""Standard atmosphere at sea level, Pa. Used for ambient conditions, not the thermo reference."""

K_BOLTZMANN: Final[float] = 1.380649e-23
N_AVOGADRO: Final[float] = 6.02214076e23
H_PLANCK: Final[float] = 6.62607015e-34

# Gravity and Earth model
G0: Final[float] = 9.80665
"""Standard gravity, m/s^2. Also the conversion factor between Isp in s and exhaust velocity."""

EARTH_RADIUS_MEAN: Final[float] = 6_371_000.0
"""Mean Earth radius, m. Used by the flat-Earth flight model for the inverse-square gravity law."""

EARTH_RADIUS_US76: Final[float] = 6_356_766.0
"""Effective Earth radius used by US Standard Atmosphere 1976 for geopotential altitude, m."""

WGS84_SEMI_MAJOR_AXIS: Final[float] = 6_378_137.0
WGS84_FLATTENING: Final[float] = 1.0 / 298.257_223_563
EARTH_GRAVITATIONAL_PARAMETER: Final[float] = 3.986_004_418e14
EARTH_J2: Final[float] = 1.082_626_68e-3
EARTH_ROTATION_RATE: Final[float] = 7.292_115e-5
