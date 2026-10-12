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
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.types import SpeciesData, SpeciesDatabase


class NASAParserError(Exception):
    """Exception raised for errors during NASA data parsing."""


DEFAULT_THERMO_FILE = Path(__file__).resolve().parents[1] / "data" / "nasa_thermo.dat"


@lru_cache(maxsize=1)
def _default_database() -> SpeciesDatabase:
    return parse_nasa_file(DEFAULT_THERMO_FILE)


def load_default_database() -> SpeciesDatabase:
    """The thermodynamic database shipped with EnSim.

    The file is parsed once per process. The returned dictionary is a fresh
    copy, but the species objects are shared and must not be modified.
    """
    return dict(_default_database())


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

    with open(path, encoding="utf-8", errors="replace") as f:
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
    lines = content.split("\n")

    if any(
        i + 1 < len(lines)
        and _card_number(lines[i + 1]) == "1"
        and lines[i].strip()
        and not lines[i].lstrip().startswith("!")
        for i in range(len(lines) - 1)
    ):
        return _parse_ensim_nasa7_format(lines)

    species_db: SpeciesDatabase = {}

    # Find the THERMO section
    thermo_start = -1
    thermo_end = len(lines)

    for i, line in enumerate(lines):
        if line.strip().upper().startswith("THERMO"):
            thermo_start = i + 1
        elif line.strip().upper() == "END" and thermo_start >= 0:
            thermo_end = i
            break

    if thermo_start < 0:
        # No THERMO marker, try parsing entire file
        thermo_start = 0

    # Skip temperature range line if present
    if thermo_start < len(lines):
        temp_line = lines[thermo_start].strip()
        if re.match(r"^[\d\s.]+$", temp_line):
            thermo_start += 1

    # Parse species entries (each is 4 lines)
    i = thermo_start
    while i + 3 < thermo_end:
        try:
            # Check if this looks like a species header
            header = lines[i]
            if len(header) < 45 or header.strip() == "" or header.startswith("!"):
                i += 1
                continue

            # Check for line number markers (1, 2, 3, 4 in column 80)
            if len(header) >= 80 and header[79] == "1":
                species = _parse_species_entry(lines[i], lines[i + 1], lines[i + 2], lines[i + 3])
                if species is None:
                    raise NASAParserError(f"Malformed species entry near line {i + 1}")
                species_db[species.name] = species
                i += 4
            else:
                i += 1

        except IndexError as exc:
            raise NASAParserError(f"Truncated species entry near line {i + 1}") from exc

    return species_db


def _card_number(line: str) -> str:
    stripped = line.rstrip()
    return stripped[-1] if stripped and stripped[-1] in "1234" else ""


def _parse_ensim_nasa7_format(lines: list[str]) -> SpeciesDatabase:
    """Parse EnSim's five-card NASA-7 representation.

    The temperature intervals are global: the numeric line after ``THERMO``
    gives T_low, T_mid and T_high for every species in the file.
    """
    species_db: SpeciesDatabase = {}
    temp_range = (200.0, 1000.0, 6000.0)
    for line in lines[:5]:
        values = line.split()
        if len(values) == 3 and all(re.fullmatch(r"[\d.]+", value) for value in values):
            t_low, t_mid, t_high = (float(value) for value in values)
            if not t_low < t_mid < t_high:
                raise NASAParserError("THERMO temperature ranges must be increasing")
            temp_range = (t_low, t_mid, t_high)
            break
    i = 0

    while i < len(lines):
        header = lines[i]
        stripped = header.strip()
        if not stripped or stripped.startswith("!") or stripped.upper() in {"THERMO", "END"}:
            i += 1
            continue
        if re.fullmatch(r"[\d.\s+-]+", stripped):
            i += 1
            continue

        if i + 3 >= len(lines) or [_card_number(lines[i + offset]) for offset in (1, 2, 3)] != [
            "1",
            "2",
            "3",
        ]:
            raise NASAParserError(f"Malformed NASA-7 entry near line {i + 1}: {stripped}")

        parts = stripped.split()
        if len(parts) < 5:
            raise NASAParserError(f"Incomplete species header near line {i + 1}: {stripped}")

        name = header[:18].strip().split()[0]
        formula = parts[-4]
        try:
            molecular_weight = float(parts[-2])
            h_formation = float(parts[-1])
        except ValueError as exc:
            raise NASAParserError(
                f"Invalid species metadata near line {i + 1}: {stripped}"
            ) from exc

        line2 = lines[i + 1].ljust(80)
        line3 = lines[i + 2].ljust(80)
        line4 = lines[i + 3].ljust(80)
        coeffs_high = np.array(
            [_parse_coefficient(line2[j : j + 15]) for j in range(0, 75, 15)]
            + [_parse_coefficient(line3[j : j + 15]) for j in (0, 15)],
            dtype=np.float64,
        )
        coeffs_low = np.array(
            [_parse_coefficient(line3[j : j + 15]) for j in (30, 45, 60)]
            + [_parse_coefficient(line4[j : j + 15]) for j in (0, 15, 30, 45)],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(coeffs_high)) or not np.all(np.isfinite(coeffs_low)):
            raise NASAParserError(f"Non-finite coefficients for {name}")

        species_db[name] = SpeciesData(
            name=name,
            molecular_weight=molecular_weight,
            phase="G",
            temp_ranges=[temp_range],
            coeffs_high=coeffs_high,
            coeffs_low=coeffs_low,
            h_formation_298=h_formation,
            formula=formula,
        )
        i += 5

    if not species_db:
        raise NASAParserError("No NASA-7 species entries were found")
    return species_db


def _parse_species_entry(line1: str, line2: str, line3: str, line4: str) -> SpeciesData | None:
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
        name = re.sub(r"\s+\d+$", "", name).strip()

        # Date and formula info (optional)
        # date = line1[18:24].strip()

        # Phase (G=gas, L=liquid, S=solid)
        phase = line1[44] if len(line1) > 44 and line1[44] in "GLS" else "G"

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
            coeffs_low=coeffs_low,
            formula=name,
        )

    except (IndexError, TypeError, ValueError, NASAParserError):
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
    field = field.replace("D", "E").replace("d", "e")

    # Handle cases like "1.234+01" (missing E)
    if re.match(r"^-?\d+\.\d+[+-]\d+$", field):
        field = re.sub(r"([+-])(\d+)$", r"E\1\2", field)

    try:
        return float(field)
    except ValueError as exc:
        raise NASAParserError(f"Unreadable coefficient field '{field}'") from exc


def _parse_nasa9_format(content: str) -> SpeciesDatabase:
    """
    NASA-9 files (nine coefficients, several temperature intervals) are
    recognised so they fail clearly instead of being read with the NASA-7 layout.
    """
    raise NASAParserError(
        "NASA-9 polynomial files are not supported; convert the data to the NASA-7 format"
    )


# Standard atomic weights, g/mol (IUPAC 2007 values, as used in NASA/TP-2002-211556)
ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.00794,
    "He": 4.002602,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 20.1797,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "Br": 79.904,
    "Kr": 83.798,
    "I": 126.90447,
    "Xe": 131.293,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "Na": 22.98977,
    "Mg": 24.305,
    "Al": 26.981538,
    "Si": 28.0855,
    "P": 30.973761,
    "K": 39.0983,
    "Ca": 40.078,
    "Ti": 47.867,
    "Fe": 55.845,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
}


def _calculate_mw_from_name(name: str) -> float:
    """
    Calculate molecular weight from species formula.

    Parses formulas like "H2O", "CO2", "CH4", "N2H4".
    """
    # Remove phase indicators and charges
    formula = re.sub(r"\([GLSC]\)$", "", name)
    formula = re.sub(r"[+-]+$", "", formula)

    mw = 0.0

    # Parse element-count pairs
    pattern = r"([A-Z][a-z]?)(\d*)"
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1

        if element not in ATOMIC_WEIGHTS:
            raise NASAParserError(f"No atomic weight for element '{element}' in {name}")
        mw += ATOMIC_WEIGHTS[element] * count

    if mw <= 0.0:
        raise NASAParserError(f"Cannot derive a molecular weight from '{name}'")
    return mw
