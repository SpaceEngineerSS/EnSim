"""Preset library of famous rocket engines."""

from dataclasses import dataclass


@dataclass
class EnginePreset:
    """Configuration for a rocket engine preset."""
    name: str
    manufacturer: str
    fuel: str
    oxidizer: str
    chamber_pressure_bar: float
    of_ratio: float
    expansion_ratio: float
    throat_area_cm2: float
    description: str
    reference_isp_vacuum: float | None = None
    reference_thrust_kn: float | None = None


# Famous rocket engine presets
ENGINE_PRESETS: dict[str, EnginePreset] = {

    "SpaceX Merlin 1D": EnginePreset(
        name="SpaceX Merlin 1D",
        manufacturer="SpaceX",
        fuel="RP1",
        oxidizer="O2",
        chamber_pressure_bar=97.2,
        of_ratio=2.36,
        expansion_ratio=16.0,
        throat_area_cm2=490.0,
        description="First stage engine for Falcon 9. Gas-generator cycle.",
        reference_isp_vacuum=311.0,
        reference_thrust_kn=914.0
    ),

    "SpaceX Merlin 1D Vacuum": EnginePreset(
        name="SpaceX Merlin 1D Vacuum",
        manufacturer="SpaceX",
        fuel="RP1",
        oxidizer="O2",
        chamber_pressure_bar=97.2,
        of_ratio=2.36,
        expansion_ratio=165.0,
        throat_area_cm2=490.0,
        description="Second stage vacuum-optimized Merlin.",
        reference_isp_vacuum=348.0,
        reference_thrust_kn=981.0
    ),

    "SpaceX Raptor 2": EnginePreset(
        name="SpaceX Raptor 2",
        manufacturer="SpaceX",
        fuel="CH4",
        oxidizer="O2",
        chamber_pressure_bar=300.0,
        of_ratio=3.6,
        expansion_ratio=40.0,
        throat_area_cm2=380.0,
        description="Full-flow staged combustion. Starship main engine.",
        reference_isp_vacuum=350.0,
        reference_thrust_kn=2300.0
    ),

    "SpaceX Raptor Vacuum": EnginePreset(
        name="SpaceX Raptor Vacuum",
        manufacturer="SpaceX",
        fuel="CH4",
        oxidizer="O2",
        chamber_pressure_bar=300.0,
        of_ratio=3.6,
        expansion_ratio=200.0,
        throat_area_cm2=380.0,
        description="Vacuum-optimized Raptor for Starship upper stage.",
        reference_isp_vacuum=380.0,
        reference_thrust_kn=2500.0
    ),

    "NASA RS-25 (SSME)": EnginePreset(
        name="NASA RS-25 (SSME)",
        manufacturer="Aerojet Rocketdyne",
        fuel="H2",
        oxidizer="O2",
        chamber_pressure_bar=206.4,
        of_ratio=6.0,
        expansion_ratio=77.5,
        throat_area_cm2=607.0,
        description="Space Shuttle Main Engine. Staged combustion cycle.",
        reference_isp_vacuum=452.3,
        reference_thrust_kn=2279.0
    ),

    "Rocket Lab Rutherford": EnginePreset(
        name="Rocket Lab Rutherford",
        manufacturer="Rocket Lab",
        fuel="RP1",
        oxidizer="O2",
        chamber_pressure_bar=120.0,
        of_ratio=2.5,
        expansion_ratio=12.0,
        throat_area_cm2=25.0,
        description="Electric pump-fed engine for Electron rocket.",
        reference_isp_vacuum=311.0,
        reference_thrust_kn=25.8
    ),

    "Blue Origin BE-4": EnginePreset(
        name="Blue Origin BE-4",
        manufacturer="Blue Origin",
        fuel="CH4",
        oxidizer="O2",
        chamber_pressure_bar=134.0,
        of_ratio=3.6,
        expansion_ratio=35.0,
        throat_area_cm2=1450.0,
        description="Oxygen-rich staged combustion for New Glenn/Vulcan.",
        reference_isp_vacuum=341.0,
        reference_thrust_kn=2400.0
    ),

    "RD-180": EnginePreset(
        name="RD-180",
        manufacturer="NPO Energomash",
        fuel="RP1",
        oxidizer="O2",
        chamber_pressure_bar=266.8,
        of_ratio=2.72,
        expansion_ratio=36.4,
        throat_area_cm2=1200.0,
        description="Russian dual-chamber staged combustion for Atlas V.",
        reference_isp_vacuum=338.4,
        reference_thrust_kn=4152.0
    ),

    "RL-10B-2": EnginePreset(
        name="RL-10B-2",
        manufacturer="Aerojet Rocketdyne",
        fuel="H2",
        oxidizer="O2",
        chamber_pressure_bar=43.5,
        of_ratio=5.88,
        expansion_ratio=285.0,
        throat_area_cm2=55.0,
        description="Expander cycle upper stage engine. Delta IV, SLS.",
        reference_isp_vacuum=465.5,
        reference_thrust_kn=110.0
    ),

    "Aerojet AJ-26": EnginePreset(
        name="Aerojet AJ-26",
        manufacturer="Aerojet Rocketdyne",
        fuel="RP1",
        oxidizer="O2",
        chamber_pressure_bar=147.0,
        of_ratio=2.63,
        expansion_ratio=32.0,
        throat_area_cm2=680.0,
        description="Refurbished NK-33. Used on Antares first stage.",
        reference_isp_vacuum=331.9,
        reference_thrust_kn=1815.0
    ),

    # Hypothetical/Educational
    "H2/O2 Ideal (Educational)": EnginePreset(
        name="H2/O2 Ideal (Educational)",
        manufacturer="Educational",
        fuel="H2",
        oxidizer="O2",
        chamber_pressure_bar=68.0,
        of_ratio=8.0,
        expansion_ratio=50.0,
        throat_area_cm2=100.0,
        description="Textbook stoichiometric H2/O2 for educational purposes.",
        reference_isp_vacuum=420.0,
        reference_thrust_kn=None
    ),
}


def get_preset(name: str) -> EnginePreset | None:
    """Get an engine preset by name."""
    return ENGINE_PRESETS.get(name)


def get_preset_names() -> list:
    """Get list of all preset names."""
    return list(ENGINE_PRESETS.keys())


def get_presets_by_fuel(fuel: str) -> dict[str, EnginePreset]:
    """Get all presets using a specific fuel."""
    return {name: preset for name, preset in ENGINE_PRESETS.items()
            if preset.fuel == fuel}
