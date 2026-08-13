"""Reduced frozen-gas nozzle parameter sweeps."""

from dataclasses import dataclass

import numpy as np

from .propulsion import NozzleConditions, calculate_performance


@dataclass(frozen=True)
class SweepConfig:
    parameter: str
    start: float
    end: float
    steps: int
    base_chamber_pressure_bar: float
    base_expansion_ratio: float
    base_throat_area_cm2: float
    eta_cstar: float = 1.0
    eta_cf: float = 1.0
    alpha_deg: float = 15.0


@dataclass(frozen=True)
class SweepResult:
    parameter_values: np.ndarray
    isp_vacuum: np.ndarray
    isp_sea_level: np.ndarray
    thrust_vacuum: np.ndarray
    thrust_sea_level: np.ndarray


def run_sweep(
    config: SweepConfig,
    *,
    gamma: float,
    mean_mw: float,
    temperature: float,
) -> SweepResult:
    """Evaluate a nozzle sweep while holding chamber gas properties fixed."""
    if config.parameter not in {"chamber_pressure", "expansion_ratio"}:
        raise ValueError("parameter must be 'chamber_pressure' or 'expansion_ratio'")
    values = np.linspace(config.start, config.end, config.steps)
    if config.steps < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Sweep bounds must be positive and finite, with at least two steps")
    if config.parameter == "expansion_ratio" and np.any(values < 1.0):
        raise ValueError("Every expansion ratio must be at least one")
    if not 1.0 < gamma < 2.0 or mean_mw <= 0.0 or temperature <= 0.0:
        raise ValueError("Frozen chamber properties are outside their physical domain")
    if config.base_chamber_pressure_bar <= 0.0 or config.base_expansion_ratio < 1.0:
        raise ValueError("Baseline chamber pressure and expansion ratio are invalid")
    if config.base_throat_area_cm2 <= 0.0:
        raise ValueError("Baseline throat area must be positive")

    isp_vacuum = np.empty(config.steps)
    isp_sea_level = np.empty(config.steps)
    thrust_vacuum = np.empty(config.steps)
    thrust_sea_level = np.empty(config.steps)
    throat_area = config.base_throat_area_cm2 * 1e-4

    for index, value in enumerate(values):
        pressure = (
            value * 1e5
            if config.parameter == "chamber_pressure"
            else config.base_chamber_pressure_bar * 1e5
        )
        expansion_ratio = (
            value if config.parameter == "expansion_ratio" else config.base_expansion_ratio
        )
        common = {
            "T_chamber": temperature,
            "P_chamber": pressure,
            "gamma": gamma,
            "mean_molecular_weight": mean_mw,
            "eta_cstar": config.eta_cstar,
            "eta_cf": config.eta_cf,
            "alpha_deg": config.alpha_deg,
        }
        vacuum = calculate_performance(
            **common,
            nozzle=NozzleConditions(expansion_ratio, pressure, 0.0, throat_area),
        )
        sea_level = calculate_performance(
            **common,
            nozzle=NozzleConditions(expansion_ratio, pressure, 101_325.0, throat_area),
        )
        isp_vacuum[index] = vacuum.isp
        isp_sea_level[index] = sea_level.isp
        thrust_vacuum[index] = vacuum.thrust
        thrust_sea_level[index] = sea_level.thrust

    return SweepResult(
        parameter_values=values,
        isp_vacuum=isp_vacuum,
        isp_sea_level=isp_sea_level,
        thrust_vacuum=thrust_vacuum,
        thrust_sea_level=thrust_sea_level,
    )
