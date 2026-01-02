"""
NASA Glenn thermodynamic polynomial data parser.

Parses NASA-format thermodynamic data files containing 7-term polynomial
coefficients for species properties (Cp, H, S).

Supported formats:
- NASA Glenn .thermo files
- CEA2 .inp format (thermo section)
- JANAF-style fixed-width format

References:
    - McBride, B.J., Zehe, M.J., & Gordon, S. (2002). "NASA Glenn Coefficients
      for Calculating Thermodynamic Properties of Individual Species"
      NASA/TP-2002-211556.
    - NASA Glenn thermodynamic database format specification
"""

import re
from pathlib import Path

import numpy as np

from ..core.types import SpeciesData, SpeciesDatabase


class NASAParserError(Exception):
    """Exception raised for errors during NASA data parsing."""
    pass


def parse_nasa_file(filepath: str | Path) -> SpeciesDatabase:
    """
    Parse a NASA-format thermodynamic data file.

    This function auto-detects the format (old vs new NASA format) and
    parses accordingly.

    Args:
        filepath: Path to the .dat, .thermo, or .inp file

    Returns:
        Dictionary mapping species names to SpeciesData objects

    Raises:
        NASAParserError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Thermodynamic data file not found: {path}")

    with open(path, encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Detect format and parse
    if _is_nasa9_format(content):
        return _parse_nasa9_format(content)
    else:
        return _parse_nasa7_format(content)


def _is_nasa9_format(content: str) -> bool:
    """Check if content is NASA-9 polynomial format (newer format)."""
    # NASA-9 format has different structure markers
    return "NASA-9" in content or "thermo nasa9" in content.lower()


def _parse_nasa7_format(content: str) -> SpeciesDatabase:
    """
    Parse NASA 7-term polynomial format (McBride format).

    Format specification:
    - Line 1: Species name (cols 1-18), date (19-24), formula (25-44),
              phase (45), T_low (46-55), T_high (56-65), T_mid (66-73),
              formula (80)
    - Line 2: Coefficients a1-a5 for high-T range (5 x 15 chars each)
    - Line 3: Coefficients a6-a7 high-T, a1-a3 low-T (5 x 15 chars)
    - Line 4: Coefficients a4-a7 low-T, H(298)/R (5 x 15 chars)

    Each coefficient field is 15 characters wide in exponential format.
    """
    species_db: SpeciesDatabase = {}
    lines = content.split('\n')

    # Find the THERMO section
    thermo_start = -1
    thermo_end = len(lines)

    for i, line in enumerate(lines):
        if line.strip().upper().startswith('THERMO'):
            thermo_start = i + 1
        elif line.strip().upper() == 'END' and thermo_start >= 0:
            thermo_end = i
            break

    if thermo_start < 0:
        # No THERMO marker, try parsing entire file
        thermo_start = 0

    # Skip temperature range line if present
    if thermo_start < len(lines):
        temp_line = lines[thermo_start].strip()
        if re.match(r'^[\d\s.]+$', temp_line):
            thermo_start += 1

    # Parse species entries (each is 4 lines)
    i = thermo_start
    while i + 3 < thermo_end:
        try:
            # Check if this looks like a species header
            header = lines[i]
            if len(header) < 45 or header.strip() == '' or header.startswith('!'):
                i += 1
                continue

            # Check for line number markers (1, 2, 3, 4 in column 80)
            if len(header) >= 80 and header[79] == '1':
                species = _parse_species_entry(
                    lines[i], lines[i+1], lines[i+2], lines[i+3]
                )
                if species:
                    species_db[species.name] = species
                i += 4
            else:
                i += 1

        except (IndexError, ValueError):
            # Skip malformed entries
            i += 1
            continue

    return species_db


def _parse_species_entry(
    line1: str,
    line2: str,
    line3: str,
    line4: str
) -> SpeciesData | None:
    """
    Parse a single species entry from 4 lines of NASA-7 format data.

    Returns None if parsing fails.
    """
    try:
        # Pad lines to 80 characters
        line1 = line1.ljust(80)
        line2 = line2.ljust(80)
        line3 = line3.ljust(80)
        line4 = line4.ljust(80)

        # Line 1: Header information
        name = line1[0:18].strip()
        if not name:
            return None

        # Clean up name - remove trailing numbers/markers
        name = re.sub(r'\s+\d+$', '', name).strip()

        # Date and formula info (optional)
        # date = line1[18:24].strip()

        # Phase (G=gas, L=liquid, S=solid)
        phase = line1[44] if len(line1) > 44 and line1[44] in 'GLS' else 'G'

        # Temperature ranges
        try:
            t_low = float(line1[45:55].strip())
            t_high = float(line1[55:65].strip())
            t_mid = float(line1[65:73].strip()) if line1[65:73].strip() else 1000.0
        except ValueError:
            t_low, t_mid, t_high = 200.0, 1000.0, 6000.0

        # Molecular weight (if present in columns 73-80)
        try:
            mw_str = line1[73:79].strip()
            molecular_weight = float(mw_str) if mw_str else _calculate_mw_from_name(name)
        except ValueError:
            molecular_weight = _calculate_mw_from_name(name)

        # Parse coefficients from lines 2-4
        # Line 2: a1-a5 (high T)
        coeffs_high = np.zeros(7, dtype=np.float64)
        coeffs_high[0] = _parse_coefficient(line2[0:15])
        coeffs_high[1] = _parse_coefficient(line2[15:30])
        coeffs_high[2] = _parse_coefficient(line2[30:45])
        coeffs_high[3] = _parse_coefficient(line2[45:60])
        coeffs_high[4] = _parse_coefficient(line2[60:75])

        # Line 3: a6, a7 (high T), a1, a2, a3 (low T)
        coeffs_high[5] = _parse_coefficient(line3[0:15])
        coeffs_high[6] = _parse_coefficient(line3[15:30])

        coeffs_low = np.zeros(7, dtype=np.float64)
        coeffs_low[0] = _parse_coefficient(line3[30:45])
        coeffs_low[1] = _parse_coefficient(line3[45:60])
        coeffs_low[2] = _parse_coefficient(line3[60:75])

        # Line 4: a4, a5, a6, a7 (low T)
        coeffs_low[3] = _parse_coefficient(line4[0:15])
        coeffs_low[4] = _parse_coefficient(line4[15:30])
        coeffs_low[5] = _parse_coefficient(line4[30:45])
        coeffs_low[6] = _parse_coefficient(line4[45:60])

        return SpeciesData(
            name=name,
            molecular_weight=molecular_weight,
            phase=phase,
            temp_ranges=[(t_low, t_mid, t_high)],
            coeffs_high=coeffs_high,
            coeffs_low=coeffs_low
        )

    except Exception:
        return None


def _parse_coefficient(field: str) -> float:
    """
    Parse a coefficient from NASA fixed-width format.

    Handles various exponential notation formats:
    - Standard: 1.234E+01
    - D notation: 1.234D+01 (Fortran)
    - No sign: 1.234E01
    """
    field = field.strip()
    if not field:
        return 0.0

    # Replace Fortran 'D' exponent with 'E'
    field = field.replace('D', 'E').replace('d', 'e')

    # Handle cases like "1.234+01" (missing E)
    if re.match(r'^-?\d+\.\d+[+-]\d+$', field):
        field = re.sub(r'([+-])(\d+)$', r'E\1\2', field)

    try:
        return float(field)
    except ValueError:
        return 0.0


def _parse_nasa9_format(content: str) -> SpeciesDatabase:
    """
    Parse NASA-9 polynomial format (newer 9-coefficient format).

    Similar structure but with 9 coefficients per temperature range
    and potentially more temperature intervals.
    """
    # For now, delegate to NASA-7 parser with warning
    # Full NASA-9 support would require different coefficient handling
    import warnings
    warnings.warn(
        "NASA-9 format detected but not fully supported. "
        "Falling back to NASA-7 parsing (may have reduced accuracy).",
        UserWarning, stacklevel=2
    )
    return _parse_nasa7_format(content)


# Atomic weights for molecular weight calculation (IUPAC 2021)
ATOMIC_WEIGHTS: dict[str, float] = {
    'H': 1.00794,
    'He': 4.002602,
    'C': 12.0107,
    'N': 14.0067,
    'O': 15.9994,
    'F': 18.9984032,
    'Ne': 20.1797,
    'S': 32.065,
    'Cl': 35.453,
    'Ar': 39.948,
    'Br': 79.904,
    'Kr': 83.798,
    'I': 126.90447,
    'Xe': 131.293,
    'Li': 6.941,
    'Be': 9.012182,
    'B': 10.811,
    'Na': 22.98977,
    'Mg': 24.305,
    'Al': 26.981538,
    'Si': 28.0855,
    'P': 30.973761,
    'K': 39.0983,
    'Ca': 40.078,
    'Ti': 47.867,
    'Fe': 55.845,
    'Ni': 58.6934,
    'Cu': 63.546,
    'Zn': 65.38,
}


def _calculate_mw_from_name(name: str) -> float:
    """
    Calculate molecular weight from species formula.

    Parses formulas like "H2O", "CO2", "CH4", "N2H4".
    """
    # Remove phase indicators and charges
    formula = re.sub(r'\([GLSC]\)$', '', name)
    formula = re.sub(r'[+-]+$', '', formula)

    mw = 0.0

    # Parse element-count pairs
    pattern = r'([A-Z][a-z]?)(\d*)'
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1

        if element in ATOMIC_WEIGHTS:
            mw += ATOMIC_WEIGHTS[element] * count

    return mw if mw > 0 else 28.0  # Default to N2 MW if parsing fails


def create_sample_database() -> SpeciesDatabase:
    """
    Create a small sample database with verified coefficients.

    These coefficients match NASA CEA reference data for common species.
    Useful for testing when full database file is not available.

    Reference: NASA/TP-2002-211556
    """
    db: SpeciesDatabase = {}

    # H2O (Water vapor) - from NASA Glenn database
    db['H2O'] = SpeciesData(
        name='H2O',
        molecular_weight=18.01528,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            2.67703787E+00,   # a1
            2.97318329E-03,   # a2
            -7.73769690E-07,  # a3
            9.44336689E-11,   # a4
            -4.26900959E-15,  # a5
            -2.98858938E+04,  # a6 (H/R integration constant)
            6.88255571E+00    # a7 (S/R integration constant)
        ], dtype=np.float64),
        coeffs_low=np.array([
            4.19864056E+00,   # a1
            -2.03643410E-03,  # a2
            6.52040211E-06,   # a3
            -5.48797062E-09,  # a4
            1.77197817E-12,   # a5
            -3.02937267E+04,  # a6
            -8.49032208E-01   # a7
        ], dtype=np.float64),
        h_formation_298=-241826.0  # J/mol
    )

    # O2 (Oxygen) - from NASA Glenn database
    db['O2'] = SpeciesData(
        name='O2',
        molecular_weight=31.9988,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            3.66096065E+00,
            6.56365811E-04,
            -1.41149627E-07,
            2.05797935E-11,
            -1.29913436E-15,
            -1.21597718E+03,
            3.41536279E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.78245636E+00,
            -2.99673416E-03,
            9.84730201E-06,
            -9.68129509E-09,
            3.24372837E-12,
            -1.06394356E+03,
            3.65767573E+00
        ], dtype=np.float64),
        h_formation_298=0.0
    )

    # H2 (Hydrogen) - from NASA Glenn database
    db['H2'] = SpeciesData(
        name='H2',
        molecular_weight=2.01588,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            2.93286575E+00,
            8.26608026E-04,
            -1.46402364E-07,
            1.54100414E-11,
            -6.88804800E-16,
            -8.13065581E+02,
            -1.02432865E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            2.34433112E+00,
            7.98052075E-03,
            -1.94781510E-05,
            2.01572094E-08,
            -7.37611761E-12,
            -9.17935173E+02,
            6.83010238E-01
        ], dtype=np.float64),
        h_formation_298=0.0
    )

    # N2 (Nitrogen) - from NASA Glenn database
    db['N2'] = SpeciesData(
        name='N2',
        molecular_weight=28.0134,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            2.95257637E+00,
            1.39690040E-03,
            -4.92631603E-07,
            7.86010195E-11,
            -4.60755204E-15,
            -9.23948688E+02,
            5.87188762E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.53100528E+00,
            -1.23660988E-04,
            -5.02999433E-07,
            2.43530612E-09,
            -1.40881235E-12,
            -1.04697628E+03,
            2.96747038E+00
        ], dtype=np.float64),
        h_formation_298=0.0
    )

    # CO2 (Carbon Dioxide) - from NASA Glenn database
    db['CO2'] = SpeciesData(
        name='CO2',
        molecular_weight=44.0095,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            4.63659493E+00,
            2.74131991E-03,
            -9.95828531E-07,
            1.60373011E-10,
            -9.16103468E-15,
            -4.90249341E+04,
            -1.93534855E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            2.35677352E+00,
            8.98459677E-03,
            -7.12356269E-06,
            2.45919022E-09,
            -1.43699548E-13,
            -4.83719697E+04,
            9.90105222E+00
        ], dtype=np.float64),
        h_formation_298=-393510.0  # J/mol
    )

    # CO (Carbon Monoxide) - from NASA Glenn database
    db['CO'] = SpeciesData(
        name='CO',
        molecular_weight=28.0101,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            3.04848583E+00,
            1.35172818E-03,
            -4.85794075E-07,
            7.88536486E-11,
            -4.69807489E-15,
            -1.42661171E+04,
            6.01709790E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.57953347E+00,
            -6.10353680E-04,
            1.01681433E-06,
            9.07005884E-10,
            -9.04424499E-13,
            -1.43440860E+04,
            3.50840928E+00
        ], dtype=np.float64),
        h_formation_298=-110530.0  # J/mol
    )

    # OH (Hydroxyl radical) - from NASA Glenn database
    db['OH'] = SpeciesData(
        name='OH',
        molecular_weight=17.00734,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            2.83864607E+00,
            1.10725586E-03,
            -2.93914978E-07,
            4.20524247E-11,
            -2.42169092E-15,
            3.69780808E+03,
            5.84452662E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.99198424E+00,
            -2.40106655E-03,
            4.61664033E-06,
            -3.87916306E-09,
            1.36319502E-12,
            3.36889836E+03,
            -1.03998477E-01
        ], dtype=np.float64),
        h_formation_298=38987.0  # J/mol
    )

    # CH4 (Methane) - from NASA Glenn database
    db['CH4'] = SpeciesData(
        name='CH4',
        molecular_weight=16.04246,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            1.65326226E+00,
            1.00263099E-02,
            -3.31661238E-06,
            5.36483138E-10,
            -3.14696758E-14,
            -1.00095936E+04,
            9.90506283E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            5.14987613E+00,
            -1.36709788E-02,
            4.91800599E-05,
            -4.84743026E-08,
            1.66693956E-11,
            -1.02466476E+04,
            -4.64130376E+00
        ], dtype=np.float64),
        h_formation_298=-74600.0  # J/mol
    )

    # H (Atomic Hydrogen) - from NASA Glenn database
    # Critical for high-temperature dissociation
    db['H'] = SpeciesData(
        name='H',
        molecular_weight=1.00794,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            2.50000286E+00,   # a1
            -5.65334214E-09,  # a2
            3.63251723E-12,   # a3
            -9.19949720E-16,  # a4
            7.95260746E-20,   # a5
            2.54736589E+04,   # a6
            -4.46698494E-01   # a7
        ], dtype=np.float64),
        coeffs_low=np.array([
            2.50000000E+00,   # a1
            0.00000000E+00,   # a2
            0.00000000E+00,   # a3
            0.00000000E+00,   # a4
            0.00000000E+00,   # a5
            2.54736599E+04,   # a6 (H/R = 25474 K means Hf = 218 kJ/mol)
            -4.46682853E-01   # a7
        ], dtype=np.float64),
        h_formation_298=217998.0  # J/mol (218 kJ/mol)
    )

    # O (Atomic Oxygen) - from NASA Glenn database
    # Critical for high-temperature dissociation
    db['O'] = SpeciesData(
        name='O',
        molecular_weight=15.9994,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            2.54363697E+00,   # a1
            -2.73162486E-05,  # a2
            -4.19029520E-09,  # a3
            4.95481845E-12,   # a4
            -4.79553694E-16,  # a5
            2.92260120E+04,   # a6
            4.92229457E+00    # a7
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.16826710E+00,   # a1
            -3.27931884E-03,  # a2
            6.64306396E-06,   # a3
            -6.12806624E-09,  # a4
            2.11265971E-12,   # a5
            2.91222592E+04,   # a6 (H/R = 29122 K means Hf = 249 kJ/mol)
            2.05193346E+00    # a7
        ], dtype=np.float64),
        h_formation_298=249175.0  # J/mol (249 kJ/mol)
    )

    # =========================================================================
    # Additional Propellants for Extended Simulation
    # =========================================================================

    # N2O4 (Nitrogen Tetroxide) - Storable oxidizer
    # From NASA CEA database
    db['N2O4'] = SpeciesData(
        name='N2O4',
        molecular_weight=92.011,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            1.15752932E+01,
            4.01615532E-03,
            -1.57178022E-06,
            2.68273657E-10,
            -1.66921538E-14,
            -1.08238125E+03,
            -3.12993556E+01
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.02002933E+00,
            2.95904306E-02,
            -3.01342572E-05,
            1.42360485E-08,
            -2.42361443E-12,
            -6.79238789E+02,
            1.18695821E+01
        ], dtype=np.float64),
        h_formation_298=9160.0  # J/mol
    )

    # NO2 (Nitrogen Dioxide) - Decomposition product
    db['NO2'] = SpeciesData(
        name='NO2',
        molecular_weight=46.0055,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            4.88475400E+00,
            2.17239555E-03,
            -8.28069088E-07,
            1.57475020E-10,
            -1.05108950E-14,
            2.31649722E+03,
            -1.17416951E-01
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.94403120E+00,
            -1.58542900E-03,
            1.66578120E-05,
            -2.04754260E-08,
            7.83505040E-12,
            2.89661800E+03,
            6.31199190E+00
        ], dtype=np.float64),
        h_formation_298=33100.0  # J/mol
    )

    # NO (Nitric Oxide) - Combustion product
    db['NO'] = SpeciesData(
        name='NO',
        molecular_weight=30.0061,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            3.26071234E+00,
            1.19101135E-03,
            -4.29122646E-07,
            6.94481463E-11,
            -4.03295681E-15,
            9.92143132E+03,
            6.36900518E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            4.21859896E+00,
            -4.63988124E-03,
            1.10443049E-05,
            -9.34055507E-09,
            2.80554874E-12,
            9.84509964E+03,
            2.28061001E+00
        ], dtype=np.float64),
        h_formation_298=91290.0  # J/mol
    )

    # N2H4 (Hydrazine) - Storable fuel
    db['N2H4'] = SpeciesData(
        name='N2H4',
        molecular_weight=32.04516,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            4.93957047E+00,
            5.44619867E-03,
            -1.81867996E-06,
            2.86416949E-10,
            -1.68234315E-14,
            9.28248610E+03,
            -2.93660548E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            3.83471652E+00,
            -6.49892297E-04,
            2.55942828E-05,
            -3.05458668E-08,
            1.22543904E-11,
            1.00893925E+04,
            5.75272614E+00
        ], dtype=np.float64),
        h_formation_298=95350.0  # J/mol
    )

    # C12H24 (RP-1 surrogate - n-Dodecane approximation)
    # RP-1 is approximated as C12H24 (average kerosene)
    db['RP1'] = SpeciesData(
        name='RP1',
        molecular_weight=168.319,  # C12H24 approx
        phase='L',
        temp_ranges=[(298.0, 1000.0, 5000.0)],
        coeffs_high=np.array([
            2.63340726E+01,
            5.15951294E-02,
            -1.76158656E-05,
            2.75303318E-09,
            -1.60259334E-13,
            -4.37463200E+04,
            -1.06582054E+02
        ], dtype=np.float64),
        coeffs_low=np.array([
            -2.21757100E+00,
            1.43567380E-01,
            -9.15679800E-05,
            2.99573500E-08,
            -3.95011800E-12,
            -3.53167100E+04,
            4.68196500E+01
        ], dtype=np.float64),
        h_formation_298=-290900.0  # J/mol (liquid)
    )

    # C (Solid carbon for soot calculations)
    db['C(s)'] = SpeciesData(
        name='C(s)',
        molecular_weight=12.0107,
        phase='S',
        temp_ranges=[(200.0, 1000.0, 5000.0)],
        coeffs_high=np.array([
            1.45571870E+00,
            1.71702470E-03,
            -6.97562390E-07,
            1.35277160E-10,
            -1.00328830E-14,
            -6.95137900E+02,
            -8.52583350E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            -3.10872240E-01,
            4.40353550E-03,
            1.90394100E-06,
            -6.38546880E-09,
            2.98964460E-12,
            -1.08650140E+02,
            1.11382480E+00
        ], dtype=np.float64),
        h_formation_298=0.0  # J/mol (reference element)
    )

    # =========================================================================
    # MMH, UDMH, H2O2, N2O - Storable Propellants (Market Requirement)
    # Reference: NASA CEA database, NIST, Thermodynamics of Organic Compounds
    # =========================================================================

    # MMH (Monomethyl Hydrazine) - CH6N2
    # Used in Soyuz, Dragon, many spacecraft RCS
    db['MMH'] = SpeciesData(
        name='MMH',
        molecular_weight=46.0717,  # CH3NHNH2
        phase='L',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            5.97286300E+00,
            1.08568840E-02,
            -3.90853780E-06,
            6.33853540E-10,
            -3.82653260E-14,
            1.10987100E+04,
            -7.03399280E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            2.83575980E+00,
            1.13608450E-02,
            1.59989500E-05,
            -2.65791360E-08,
            1.08673640E-11,
            1.23653780E+04,
            1.00936700E+01
        ], dtype=np.float64),
        h_formation_298=54840.0  # J/mol (liquid)
    )

    # UDMH (Unsymmetrical Dimethyl Hydrazine) - C2H8N2
    # Used in Proton, Long March, historical US rockets
    db['UDMH'] = SpeciesData(
        name='UDMH',
        molecular_weight=60.0986,  # (CH3)2NNH2
        phase='L',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            7.63116200E+00,
            1.47316050E-02,
            -5.27679720E-06,
            8.51247420E-10,
            -5.12137580E-14,
            5.17621100E+03,
            -1.60548670E+01
        ], dtype=np.float64),
        coeffs_low=np.array([
            1.55206330E+00,
            2.42217360E-02,
            1.11891880E-05,
            -3.22019830E-08,
            1.42818850E-11,
            7.08247380E+03,
            1.73696330E+01
        ], dtype=np.float64),
        h_formation_298=48300.0  # J/mol (liquid)
    )

    # H2O2 (Hydrogen Peroxide) - High concentration (90%+)
    # Used in RCS, hybrid rockets, monopropellant
    db['H2O2'] = SpeciesData(
        name='H2O2',
        molecular_weight=34.01468,
        phase='L',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            4.57316685E+00,
            4.33613639E-03,
            -1.47468882E-06,
            2.34890357E-10,
            -1.43165356E-14,
            -1.80069609E+04,
            6.64961940E-01
        ], dtype=np.float64),
        coeffs_low=np.array([
            4.31515149E+00,
            -8.47390622E-04,
            1.76404323E-05,
            -2.26762944E-08,
            9.08950158E-12,
            -1.77067437E+04,
            3.27373319E+00
        ], dtype=np.float64),
        h_formation_298=-187780.0  # J/mol (liquid)
    )

    # N2O (Nitrous Oxide) - "Laughing Gas"
    # Popular hybrid oxidizer for amateur rocketry
    db['N2O'] = SpeciesData(
        name='N2O',
        molecular_weight=44.0128,
        phase='G',
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array([
            4.82318400E+00,
            2.62693690E-03,
            -9.58508320E-07,
            1.60007330E-10,
            -9.77535890E-15,
            8.07340920E+03,
            -2.20172070E+00
        ], dtype=np.float64),
        coeffs_low=np.array([
            2.25715020E+00,
            1.13047280E-02,
            -1.36713190E-05,
            9.68198020E-09,
            -2.93071820E-12,
            8.74177280E+03,
            1.07579920E+01
        ], dtype=np.float64),
        h_formation_298=82050.0  # J/mol (gas)
    )

    return db
