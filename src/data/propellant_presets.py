"""
Comprehensive propellant preset database.

Contains pre-configured propellant combinations with validated
properties from NASA CEA and literature sources.

Each preset includes:
- Fuel and oxidizer specifications
- Optimal mixture ratios
- Performance estimates
- Temperature and density data
- Application notes

References:
    - NASA CEA (Chemical Equilibrium with Applications)
    - Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed.
    - CPIA/M5 Liquid Propellant Manual
"""

from dataclasses import dataclass
from enum import Enum, auto


class PropellantCategory(Enum):
    """Propellant application categories."""
    CRYOGENIC = auto()           # LOX/LH2, LOX/LCH4
    HYDROCARBON = auto()         # LOX/RP-1, LOX/Ethanol
    STORABLE = auto()            # N2O4/UDMH, N2O4/MMH
    GREEN = auto()               # Non-toxic alternatives
    MONOPROPELLANT = auto()      # Hydrazine, HAN
    HYBRID = auto()              # LOX/HTPB, N2O/HTPB
    EXOTIC = auto()              # High-energy (F2, Be)


class ToxicityLevel(Enum):
    """Propellant toxicity classification."""
    BENIGN = auto()              # Water, N2
    LOW = auto()                 # Kerosene, ethanol
    MODERATE = auto()            # LOX, LH2
    HIGH = auto()                # MMH, N2H4
    EXTREME = auto()             # N2O4, F2, Be


@dataclass
class PropellantSpec:
    """Specification for a single propellant component."""
    name: str
    formula: str
    molecular_weight: float  # g/mol
    density: float           # kg/m³ at storage conditions
    boiling_point: float     # K at 1 atm
    melting_point: float     # K
    heat_of_formation: float # kJ/mol
    toxicity: ToxicityLevel
    storage_temp: float      # K (typical storage)
    vapor_pressure: float    # Pa at storage temp


@dataclass
class PropellantPreset:
    """
    Complete propellant combination preset.

    Includes both components and validated performance data.
    """
    name: str
    description: str
    category: PropellantCategory

    # Components
    fuel: PropellantSpec
    oxidizer: PropellantSpec

    # Mixture ratio
    of_ratio_optimal: float     # Optimal O/F for Isp
    of_ratio_range: tuple[float, float]  # Valid range

    # Performance at optimal O/F, Pc = 7 MPa, ε = 50
    isp_vacuum: float           # s
    isp_sea_level: float        # s
    c_star: float               # m/s
    chamber_temp: float         # K
    gamma: float                # Cp/Cv of products
    mean_mw: float              # g/mol of products

    # Bulk density
    density_bulk: float         # kg/m³ (mixture at optimal O/F)

    # Notes
    notes: str
    applications: list[str]


# =============================================================================
# Propellant Component Database
# =============================================================================

FUELS = {
    "LH2": PropellantSpec(
        name="Liquid Hydrogen",
        formula="H2",
        molecular_weight=2.016,
        density=70.8,
        boiling_point=20.3,
        melting_point=14.0,
        heat_of_formation=0.0,
        toxicity=ToxicityLevel.MODERATE,
        storage_temp=20.0,
        vapor_pressure=101325.0
    ),
    "CH4": PropellantSpec(
        name="Liquid Methane",
        formula="CH4",
        molecular_weight=16.04,
        density=422.0,
        boiling_point=111.7,
        melting_point=90.7,
        heat_of_formation=-74.87,
        toxicity=ToxicityLevel.LOW,
        storage_temp=112.0,
        vapor_pressure=101325.0
    ),
    "RP1": PropellantSpec(
        name="RP-1 Kerosene",
        formula="C12H24",
        molecular_weight=170.0,
        density=820.0,
        boiling_point=489.0,
        melting_point=225.0,
        heat_of_formation=-250.0,
        toxicity=ToxicityLevel.LOW,
        storage_temp=293.0,
        vapor_pressure=1000.0
    ),
    "C2H5OH": PropellantSpec(
        name="Ethanol",
        formula="C2H5OH",
        molecular_weight=46.07,
        density=789.0,
        boiling_point=351.5,
        melting_point=159.0,
        heat_of_formation=-277.0,
        toxicity=ToxicityLevel.LOW,
        storage_temp=293.0,
        vapor_pressure=5950.0
    ),
    "N2H4": PropellantSpec(
        name="Hydrazine",
        formula="N2H4",
        molecular_weight=32.05,
        density=1004.0,
        boiling_point=386.7,
        melting_point=274.7,
        heat_of_formation=50.63,
        toxicity=ToxicityLevel.HIGH,
        storage_temp=293.0,
        vapor_pressure=1900.0
    ),
    "MMH": PropellantSpec(
        name="Monomethylhydrazine",
        formula="CH3NHNH2",
        molecular_weight=46.07,
        density=880.0,
        boiling_point=360.8,
        melting_point=220.8,
        heat_of_formation=54.84,
        toxicity=ToxicityLevel.HIGH,
        storage_temp=293.0,
        vapor_pressure=4700.0
    ),
    "UDMH": PropellantSpec(
        name="Unsymmetrical Dimethylhydrazine",
        formula="(CH3)2NNH2",
        molecular_weight=60.10,
        density=791.0,
        boiling_point=336.0,
        melting_point=216.0,
        heat_of_formation=48.3,
        toxicity=ToxicityLevel.HIGH,
        storage_temp=293.0,
        vapor_pressure=16000.0
    ),
    "LNG": PropellantSpec(
        name="Liquefied Natural Gas",
        formula="CH4+C2H6",
        molecular_weight=17.5,
        density=430.0,
        boiling_point=113.0,
        melting_point=91.0,
        heat_of_formation=-78.0,
        toxicity=ToxicityLevel.LOW,
        storage_temp=113.0,
        vapor_pressure=101325.0
    ),
    "C3H8": PropellantSpec(
        name="Propane",
        formula="C3H8",
        molecular_weight=44.10,
        density=493.0,
        boiling_point=231.1,
        melting_point=85.5,
        heat_of_formation=-103.85,
        toxicity=ToxicityLevel.LOW,
        storage_temp=231.0,
        vapor_pressure=101325.0
    ),
    "HTPB": PropellantSpec(
        name="Hydroxyl-Terminated Polybutadiene",
        formula="(C4H6)n",
        molecular_weight=2800.0,  # Typical
        density=920.0,
        boiling_point=673.0,      # Decomposes
        melting_point=193.0,
        heat_of_formation=-46.0,
        toxicity=ToxicityLevel.BENIGN,
        storage_temp=293.0,
        vapor_pressure=0.0
    ),
}

OXIDIZERS = {
    "LOX": PropellantSpec(
        name="Liquid Oxygen",
        formula="O2",
        molecular_weight=32.0,
        density=1141.0,
        boiling_point=90.2,
        melting_point=54.4,
        heat_of_formation=0.0,
        toxicity=ToxicityLevel.MODERATE,
        storage_temp=90.0,
        vapor_pressure=101325.0
    ),
    "N2O4": PropellantSpec(
        name="Nitrogen Tetroxide",
        formula="N2O4",
        molecular_weight=92.01,
        density=1450.0,
        boiling_point=294.3,
        melting_point=261.9,
        heat_of_formation=9.08,
        toxicity=ToxicityLevel.EXTREME,
        storage_temp=293.0,
        vapor_pressure=96000.0
    ),
    "N2O": PropellantSpec(
        name="Nitrous Oxide",
        formula="N2O",
        molecular_weight=44.01,
        density=1220.0,
        boiling_point=184.7,
        melting_point=182.3,
        heat_of_formation=82.05,
        toxicity=ToxicityLevel.LOW,
        storage_temp=293.0,
        vapor_pressure=5150000.0
    ),
    "H2O2_90": PropellantSpec(
        name="High-Test Peroxide (90%)",
        formula="H2O2",
        molecular_weight=34.01,
        density=1390.0,
        boiling_point=423.0,
        melting_point=272.7,
        heat_of_formation=-187.8,
        toxicity=ToxicityLevel.MODERATE,
        storage_temp=293.0,
        vapor_pressure=200.0
    ),
    "LF2": PropellantSpec(
        name="Liquid Fluorine",
        formula="F2",
        molecular_weight=38.0,
        density=1505.0,
        boiling_point=85.0,
        melting_point=53.5,
        heat_of_formation=0.0,
        toxicity=ToxicityLevel.EXTREME,
        storage_temp=85.0,
        vapor_pressure=101325.0
    ),
    "IRFNA": PropellantSpec(
        name="Inhibited Red Fuming Nitric Acid",
        formula="HNO3+N2O4",
        molecular_weight=63.01,
        density=1560.0,
        boiling_point=359.0,
        melting_point=231.0,
        heat_of_formation=-174.1,
        toxicity=ToxicityLevel.EXTREME,
        storage_temp=293.0,
        vapor_pressure=6400.0
    ),
    "ClF5": PropellantSpec(
        name="Chlorine Pentafluoride",
        formula="ClF5",
        molecular_weight=130.45,
        density=1900.0,
        boiling_point=260.1,
        melting_point=170.0,
        heat_of_formation=-238.5,
        toxicity=ToxicityLevel.EXTREME,
        storage_temp=260.0,
        vapor_pressure=101325.0
    ),
}


# =============================================================================
# Propellant Preset Database (15+ Combinations)
# =============================================================================

PROPELLANT_PRESETS: dict[str, PropellantPreset] = {
    # =========================================================================
    # CRYOGENIC PROPELLANTS
    # =========================================================================
    "LOX_LH2": PropellantPreset(
        name="LOX/LH2",
        description="Liquid Oxygen / Liquid Hydrogen - Highest specific impulse bipropellant",
        category=PropellantCategory.CRYOGENIC,
        fuel=FUELS["LH2"],
        oxidizer=OXIDIZERS["LOX"],
        of_ratio_optimal=6.0,
        of_ratio_range=(4.0, 8.0),
        isp_vacuum=455.0,
        isp_sea_level=363.0,
        c_star=2386.0,
        chamber_temp=3250.0,
        gamma=1.14,
        mean_mw=12.0,
        density_bulk=320.0,  # Low due to LH2
        notes="Best Isp but very low density. Requires large tanks and complex handling.",
        applications=["Upper stages", "Space Shuttle Main Engine", "RS-25", "RL-10", "Vulcain"]
    ),

    "LOX_LCH4": PropellantPreset(
        name="LOX/LCH4",
        description="Liquid Oxygen / Liquid Methane - Modern high-performance choice",
        category=PropellantCategory.CRYOGENIC,
        fuel=FUELS["CH4"],
        oxidizer=OXIDIZERS["LOX"],
        of_ratio_optimal=3.6,
        of_ratio_range=(2.8, 4.2),
        isp_vacuum=363.0,
        isp_sea_level=311.0,
        c_star=1850.0,
        chamber_temp=3550.0,
        gamma=1.15,
        mean_mw=20.0,
        density_bulk=828.0,
        notes="Good Isp and density balance. Cleaner burning than RP-1. ISRU potential on Mars.",
        applications=["Raptor", "BE-4", "Vulcan", "Starship", "Mars missions"]
    ),

    "LOX_LNG": PropellantPreset(
        name="LOX/LNG",
        description="Liquid Oxygen / Liquefied Natural Gas - Cost-effective cryogenic",
        category=PropellantCategory.CRYOGENIC,
        fuel=FUELS["LNG"],
        oxidizer=OXIDIZERS["LOX"],
        of_ratio_optimal=3.4,
        of_ratio_range=(2.6, 4.0),
        isp_vacuum=358.0,
        isp_sea_level=306.0,
        c_star=1820.0,
        chamber_temp=3500.0,
        gamma=1.15,
        mean_mw=20.5,
        density_bulk=810.0,
        notes="Similar to LOX/LCH4 but uses cheaper industrial LNG. Slight Isp penalty.",
        applications=["Cost-sensitive applications", "Reusable boosters"]
    ),

    # =========================================================================
    # HYDROCARBON PROPELLANTS
    # =========================================================================
    "LOX_RP1": PropellantPreset(
        name="LOX/RP-1",
        description="Liquid Oxygen / RP-1 Kerosene - Workhorse combination",
        category=PropellantCategory.HYDROCARBON,
        fuel=FUELS["RP1"],
        oxidizer=OXIDIZERS["LOX"],
        of_ratio_optimal=2.72,
        of_ratio_range=(2.2, 3.2),
        isp_vacuum=338.0,
        isp_sea_level=282.0,
        c_star=1780.0,
        chamber_temp=3670.0,
        gamma=1.15,
        mean_mw=23.0,
        density_bulk=1030.0,
        notes="High density, moderate Isp. Coking at high temps can limit regenerative cooling.",
        applications=["Merlin", "F-1", "RD-180", "Atlas", "Falcon 9", "Electron"]
    ),

    "LOX_Ethanol": PropellantPreset(
        name="LOX/Ethanol",
        description="Liquid Oxygen / Ethanol - Historic and educational",
        category=PropellantCategory.HYDROCARBON,
        fuel=FUELS["C2H5OH"],
        oxidizer=OXIDIZERS["LOX"],
        of_ratio_optimal=1.8,
        of_ratio_range=(1.4, 2.4),
        isp_vacuum=308.0,
        isp_sea_level=255.0,
        c_star=1680.0,
        chamber_temp=3400.0,
        gamma=1.16,
        mean_mw=24.0,
        density_bulk=990.0,
        notes="Lower performance than RP-1 but cleaner and easier to handle. Good for amateurs.",
        applications=["V-2", "Amateur rockets", "Educational", "Small launchers"]
    ),

    "LOX_Propane": PropellantPreset(
        name="LOX/Propane",
        description="Liquid Oxygen / Propane - Alternative hydrocarbon",
        category=PropellantCategory.HYDROCARBON,
        fuel=FUELS["C3H8"],
        oxidizer=OXIDIZERS["LOX"],
        of_ratio_optimal=3.2,
        of_ratio_range=(2.5, 3.8),
        isp_vacuum=350.0,
        isp_sea_level=298.0,
        c_star=1800.0,
        chamber_temp=3580.0,
        gamma=1.15,
        mean_mw=21.5,
        density_bulk=860.0,
        notes="Between methane and RP-1 in performance. Self-pressurizing at ambient temp.",
        applications=["Small launchers", "Pressure-fed engines"]
    ),

    # =========================================================================
    # STORABLE PROPELLANTS
    # =========================================================================
    "N2O4_UDMH": PropellantPreset(
        name="N2O4/UDMH",
        description="Nitrogen Tetroxide / UDMH - Classic storable hypergolic",
        category=PropellantCategory.STORABLE,
        fuel=FUELS["UDMH"],
        oxidizer=OXIDIZERS["N2O4"],
        of_ratio_optimal=2.6,
        of_ratio_range=(1.8, 3.2),
        isp_vacuum=318.0,
        isp_sea_level=270.0,
        c_star=1720.0,
        chamber_temp=3200.0,
        gamma=1.17,
        mean_mw=24.0,
        density_bulk=1150.0,
        notes="Hypergolic ignition. Very toxic but excellent storability. Used in ICBMs.",
        applications=["Proton", "Titan", "Long March", "Military missiles"]
    ),

    "N2O4_MMH": PropellantPreset(
        name="N2O4/MMH",
        description="Nitrogen Tetroxide / MMH - Standard spacecraft propellant",
        category=PropellantCategory.STORABLE,
        fuel=FUELS["MMH"],
        oxidizer=OXIDIZERS["N2O4"],
        of_ratio_optimal=2.2,
        of_ratio_range=(1.6, 2.8),
        isp_vacuum=326.0,
        isp_sea_level=276.0,
        c_star=1740.0,
        chamber_temp=3250.0,
        gamma=1.17,
        mean_mw=23.0,
        density_bulk=1190.0,
        notes="Higher Isp than UDMH. Standard for spacecraft maneuvering systems.",
        applications=["Space Shuttle OMS", "Dragon", "Satellites", "Apollo LM"]
    ),

    "N2O4_N2H4": PropellantPreset(
        name="N2O4/N2H4",
        description="Nitrogen Tetroxide / Hydrazine - High-density storable",
        category=PropellantCategory.STORABLE,
        fuel=FUELS["N2H4"],
        oxidizer=OXIDIZERS["N2O4"],
        of_ratio_optimal=1.3,
        of_ratio_range=(1.0, 1.6),
        isp_vacuum=315.0,
        isp_sea_level=267.0,
        c_star=1700.0,
        chamber_temp=3280.0,
        gamma=1.17,
        mean_mw=22.5,
        density_bulk=1210.0,
        notes="Highest density storable. Requires more oxidizer mass.",
        applications=["Small thrusters", "Reaction control systems"]
    ),

    "IRFNA_UDMH": PropellantPreset(
        name="IRFNA/UDMH",
        description="Red Fuming Nitric Acid / UDMH - Military storable",
        category=PropellantCategory.STORABLE,
        fuel=FUELS["UDMH"],
        oxidizer=OXIDIZERS["IRFNA"],
        of_ratio_optimal=3.0,
        of_ratio_range=(2.2, 3.6),
        isp_vacuum=310.0,
        isp_sea_level=262.0,
        c_star=1680.0,
        chamber_temp=3100.0,
        gamma=1.18,
        mean_mw=25.0,
        density_bulk=1200.0,
        notes="Lower cost than N2O4 but more corrosive. Used in older missiles.",
        applications=["Nike Ajax", "Scud", "Historical military systems"]
    ),

    # =========================================================================
    # GREEN / LOW-TOXICITY PROPELLANTS
    # =========================================================================
    "LOX_Ethanol_75": PropellantPreset(
        name="LOX/75% Ethanol",
        description="Liquid Oxygen / 75% Ethanol-Water - Green bipropellant",
        category=PropellantCategory.GREEN,
        fuel=FUELS["C2H5OH"],
        oxidizer=OXIDIZERS["LOX"],
        of_ratio_optimal=1.4,
        of_ratio_range=(1.1, 1.8),
        isp_vacuum=285.0,
        isp_sea_level=235.0,
        c_star=1580.0,
        chamber_temp=3100.0,
        gamma=1.18,
        mean_mw=25.0,
        density_bulk=950.0,
        notes="Water content reduces performance but improves cooling and safety.",
        applications=["Educational", "Amateur", "Low-cost small launchers"]
    ),

    "N2O_HTPB": PropellantPreset(
        name="N2O/HTPB",
        description="Nitrous Oxide / HTPB - Hybrid rocket propellant",
        category=PropellantCategory.HYBRID,
        fuel=FUELS["HTPB"],
        oxidizer=OXIDIZERS["N2O"],
        of_ratio_optimal=7.0,
        of_ratio_range=(5.0, 9.0),
        isp_vacuum=250.0,
        isp_sea_level=215.0,
        c_star=1550.0,
        chamber_temp=2800.0,
        gamma=1.20,
        mean_mw=26.0,
        density_bulk=1080.0,
        notes="Simple, safe, and low cost. Self-pressurizing oxidizer.",
        applications=["SpaceShipOne/Two", "Amateur rockets", "Educational"]
    ),

    "H2O2_RP1": PropellantPreset(
        name="H2O2/RP-1",
        description="High-Test Peroxide / RP-1 - Non-toxic storable",
        category=PropellantCategory.GREEN,
        fuel=FUELS["RP1"],
        oxidizer=OXIDIZERS["H2O2_90"],
        of_ratio_optimal=7.5,
        of_ratio_range=(6.0, 9.0),
        isp_vacuum=310.0,
        isp_sea_level=265.0,
        c_star=1650.0,
        chamber_temp=2950.0,
        gamma=1.18,
        mean_mw=24.0,
        density_bulk=1130.0,
        notes="Non-hypergolic but can be catalytically started. Lower toxicity than N2O4.",
        applications=["Black Arrow", "Small launchers", "RCS systems"]
    ),

    # =========================================================================
    # HIGH-PERFORMANCE / EXOTIC
    # =========================================================================
    "LF2_LH2": PropellantPreset(
        name="LF2/LH2",
        description="Liquid Fluorine / Liquid Hydrogen - Maximum theoretical Isp",
        category=PropellantCategory.EXOTIC,
        fuel=FUELS["LH2"],
        oxidizer=OXIDIZERS["LF2"],
        of_ratio_optimal=12.0,
        of_ratio_range=(8.0, 16.0),
        isp_vacuum=479.0,
        isp_sea_level=390.0,
        c_star=2520.0,
        chamber_temp=4700.0,
        gamma=1.12,
        mean_mw=10.0,
        density_bulk=580.0,
        notes="Highest Isp possible but extremely toxic HF exhaust. Never used operationally.",
        applications=["Theoretical maximum", "Nuclear thermal comparison baseline"]
    ),

    "ClF5_N2H4": PropellantPreset(
        name="ClF5/N2H4",
        description="Chlorine Pentafluoride / Hydrazine - Extreme hypergolic",
        category=PropellantCategory.EXOTIC,
        fuel=FUELS["N2H4"],
        oxidizer=OXIDIZERS["ClF5"],
        of_ratio_optimal=2.8,
        of_ratio_range=(2.0, 3.5),
        isp_vacuum=350.0,
        isp_sea_level=300.0,
        c_star=1850.0,
        chamber_temp=3800.0,
        gamma=1.14,
        mean_mw=20.0,
        density_bulk=1300.0,
        notes="Extremely reactive and toxic. Theoretical high-density storable.",
        applications=["Research only", "Theoretical studies"]
    ),

    # =========================================================================
    # MONOPROPELLANTS
    # =========================================================================
    "N2H4_MONO": PropellantPreset(
        name="Hydrazine Monopropellant",
        description="Catalytic decomposition of hydrazine",
        category=PropellantCategory.MONOPROPELLANT,
        fuel=FUELS["N2H4"],
        oxidizer=FUELS["N2H4"],  # Self-oxidizing
        of_ratio_optimal=0.0,    # N/A for monoprop
        of_ratio_range=(0.0, 0.0),
        isp_vacuum=235.0,
        isp_sea_level=195.0,
        c_star=1180.0,
        chamber_temp=1100.0,
        gamma=1.27,
        mean_mw=15.0,
        density_bulk=1004.0,
        notes="Simple catalyst bed decomposition. Standard for RCS and small satellites.",
        applications=["Attitude control", "Small thrusters", "Satellites", "Curiosity EDL"]
    ),

    "H2O2_MONO": PropellantPreset(
        name="HTP Monopropellant",
        description="Catalytic decomposition of 90% hydrogen peroxide",
        category=PropellantCategory.MONOPROPELLANT,
        fuel=OXIDIZERS["H2O2_90"],
        oxidizer=OXIDIZERS["H2O2_90"],
        of_ratio_optimal=0.0,
        of_ratio_range=(0.0, 0.0),
        isp_vacuum=185.0,
        isp_sea_level=155.0,
        c_star=980.0,
        chamber_temp=1020.0,
        gamma=1.28,
        mean_mw=18.0,
        density_bulk=1390.0,
        notes="Non-toxic monopropellant. Lower Isp than hydrazine but safer.",
        applications=["Green propulsion", "Small spacecraft", "Rocket belts"]
    ),
}


def get_preset(name: str) -> PropellantPreset | None:
    """Get a propellant preset by name."""
    return PROPELLANT_PRESETS.get(name)


def get_presets_by_category(category: PropellantCategory) -> list[PropellantPreset]:
    """Get all presets in a category."""
    return [p for p in PROPELLANT_PRESETS.values() if p.category == category]


def get_all_preset_names() -> list[str]:
    """Get list of all preset names."""
    return list(PROPELLANT_PRESETS.keys())


def find_preset_by_isp(min_isp: float, max_isp: float = 500.0) -> list[PropellantPreset]:
    """Find presets within an Isp range."""
    return [p for p in PROPELLANT_PRESETS.values()
            if min_isp <= p.isp_vacuum <= max_isp]


def find_preset_by_density(min_density: float) -> list[PropellantPreset]:
    """Find presets with bulk density above threshold."""
    return [p for p in PROPELLANT_PRESETS.values()
            if p.density_bulk >= min_density]


def get_non_toxic_presets() -> list[PropellantPreset]:
    """Get presets with low toxicity (no extreme/high toxicity components)."""
    non_toxic = []
    for preset in PROPELLANT_PRESETS.values():
        if (preset.fuel.toxicity in [ToxicityLevel.BENIGN, ToxicityLevel.LOW, ToxicityLevel.MODERATE] and
            preset.oxidizer.toxicity in [ToxicityLevel.BENIGN, ToxicityLevel.LOW, ToxicityLevel.MODERATE]):
            non_toxic.append(preset)
    return non_toxic

