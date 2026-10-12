"""Propellants as they are loaded into a vehicle.

A liquid propellant enters the chamber with the enthalpy of the liquid at its
storage temperature, not that of the gas at 298 K. For liquid hydrogen and
oxygen the difference is several percent of chamber temperature, so reactants
carry an assigned enthalpy, the same convention NASA CEA uses for its
reactant list.

Enthalpies are mass-specific (J/kg) and come from the reactant entries of the
NASA Glenn thermodynamic database distributed with CEA 3.3.2 (McBride, Zehe &
Gordon, NASA/TP-2002-211556); `scripts/generate_cea_references.py` reproduces
them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chemistry import parse_formula
from .types import ELEMENTS


@dataclass(frozen=True)
class Propellant:
    """A fuel or oxidizer in its storage state."""

    key: str  # identifier used by EngineInputs, e.g. "LOX"
    name: str  # display name
    role: str  # "fuel" or "oxidizer"
    formula: str  # elemental formula; may be fractional (RP-1 is CH1.9423)
    temperature: float  # K, storage temperature at which `enthalpy` applies
    enthalpy: float  # J/kg at `temperature`
    phase: str = "liquid"
    note: str = ""

    @property
    def molar_mass(self) -> float:
        """kg/mol, from the elemental formula."""
        grams = sum(
            ELEMENTS[symbol].atomic_weight * count
            for symbol, count in parse_formula(self.formula).items()
        )
        return grams / 1000.0

    @property
    def molar_enthalpy(self) -> float:
        """J/mol at the storage temperature."""
        return self.enthalpy * self.molar_mass


_TABLE = (
    Propellant(
        "LH2", "Liquid hydrogen", "fuel", "H2", 20.27, -4_470_504.2, note="normal boiling point"
    ),
    Propellant(
        "LCH4", "Liquid methane", "fuel", "CH4", 111.643, -5_562_301.5, note="normal boiling point"
    ),
    Propellant(
        "RP-1",
        "RP-1 kerosene",
        "fuel",
        "CH1.9423",
        298.15,
        -1_768_558.7,
        note="CEA surrogate formula CH1.9423",
    ),
    Propellant("N2H4", "Hydrazine", "fuel", "N2H4", 298.15, 1_572_156.3),
    Propellant("MMH", "Monomethylhydrazine", "fuel", "CH6N2", 298.15, 1_176_426.2),
    Propellant("UDMH", "Unsymmetrical dimethylhydrazine", "fuel", "C2H8N2", 298.15, 813_666.7),
    Propellant("ETHANOL", "Ethanol", "fuel", "C2H6O", 298.15, -6_023_863.7),
    Propellant(
        "LOX", "Liquid oxygen", "oxidizer", "O2", 90.17, -405_609.0, note="normal boiling point"
    ),
    Propellant("NTO", "Nitrogen tetroxide", "oxidizer", "N2O4", 298.15, -190_727.2),
    Propellant("HTP", "Hydrogen peroxide (100 %)", "oxidizer", "H2O2", 298.15, -5_520_557.6),
    Propellant(
        "N2O", "Nitrous oxide (vapour)", "oxidizer", "N2O", 298.15, 1_854_006.1, phase="gas"
    ),
)

PROPELLANTS: dict[str, Propellant] = {p.key: p for p in _TABLE}
FUELS = tuple(p.key for p in _TABLE if p.role == "fuel")
OXIDIZERS = tuple(p.key for p in _TABLE if p.role == "oxidizer")

# Names written by EnSim 3.x, which modelled every reactant as a gas at 298 K
LEGACY_NAMES = {
    "H2": "LH2",
    "CH4": "LCH4",
    "RP1": "RP-1",
    "O2": "LOX",
    "N2O4": "NTO",
    "H2O2": "HTP",
    "C2H5OH": "ETHANOL",
}


def get_propellant(key: str) -> Propellant:
    """Look up a propellant by key (case-insensitive)."""
    try:
        return PROPELLANTS[key.upper()]
    except KeyError:
        known = ", ".join(PROPELLANTS)
        raise KeyError(f"Unknown propellant '{key}'. Known: {known}") from None
