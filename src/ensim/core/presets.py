"""Reproducible input cases for the engine workspace."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnginePreset:
    name: str
    group: str
    fuel: str
    oxidizer: str
    chamber_pressure_bar: float
    of_ratio: float
    expansion_ratio: float
    throat_area_cm2: float
    description: str


ENGINE_PRESETS: dict[str, EnginePreset] = {
    "H2/O2 Vacuum Study": EnginePreset(
        name="H2/O2 Vacuum Study",
        group="Reference cases",
        fuel="H2",
        oxidizer="O2",
        chamber_pressure_bar=70.0,
        of_ratio=6.0,
        expansion_ratio=60.0,
        throat_area_cm2=100.0,
        description="Fuel-rich hydrogen/oxygen input case for a vacuum expansion study.",
    ),
    "CH4/O2 Engine Study": EnginePreset(
        name="CH4/O2 Engine Study",
        group="Reference cases",
        fuel="CH4",
        oxidizer="O2",
        chamber_pressure_bar=100.0,
        of_ratio=3.4,
        expansion_ratio=35.0,
        throat_area_cm2=100.0,
        description="Methane/oxygen input case for comparative thermochemistry.",
    ),
    "RP-1/O2 Engine Study": EnginePreset(
        name="RP-1/O2 Engine Study",
        group="Reference cases",
        fuel="RP1",
        oxidizer="O2",
        chamber_pressure_bar=100.0,
        of_ratio=2.6,
        expansion_ratio=25.0,
        throat_area_cm2=100.0,
        description="RP-1 surrogate/oxygen case using the packaged C12H26 reactant model.",
    ),
}


def get_preset(name: str) -> EnginePreset | None:
    return ENGINE_PRESETS.get(name)


def get_preset_names() -> list[str]:
    return list(ENGINE_PRESETS)


def get_presets_by_fuel(fuel: str) -> dict[str, EnginePreset]:
    return {name: preset for name, preset in ENGINE_PRESETS.items() if preset.fuel == fuel}
