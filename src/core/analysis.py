"""Parametric sweep analysis tool."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .propulsion import NozzleConditions, calculate_performance


@dataclass
class SweepConfig:
    """Configuration for a parametric sweep."""

    parameter: str  # 'chamber_pressure', 'expansion_ratio', 'of_ratio'
    start: float
    end: float
    steps: int

    # Fixed parameters
    fuel: str = "H2"
    oxidizer: str = "O2"
    base_of_ratio: float = 8.0
    base_chamber_pressure_bar: float = 68.0
    base_expansion_ratio: float = 50.0
    base_throat_area_cm2: float = 100.0


@dataclass
class SweepResult:
    """Results from a parametric sweep."""

    parameter_name: str
    parameter_values: np.ndarray
    isp_vacuum: np.ndarray
    isp_sea_level: np.ndarray
    thrust: np.ndarray
    c_star: np.ndarray
    temperature: np.ndarray
    exit_mach: np.ndarray


def run_sweep(
    config: SweepConfig,
    gamma: float,
    mean_mw: float,
    temperature: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SweepResult:
    """
    Run a parametric sweep analysis.

    Args:
        config: Sweep configuration
        gamma: Ratio of specific heats (from equilibrium)
        mean_mw: Mean molecular weight g/mol (from equilibrium)
        temperature: Chamber temperature K (from equilibrium)
        progress_callback: Optional callback(current, total)

    Returns:
        SweepResult with arrays of results
    """
    # Create parameter array
    param_values = np.linspace(config.start, config.end, config.steps)

    # Initialize result arrays
    n = config.steps
    isp_vac = np.zeros(n)
    isp_sl = np.zeros(n)
    thrust_arr = np.zeros(n)
    c_star_arr = np.zeros(n)
    temps = np.zeros(n)
    exit_mach = np.zeros(n)

    for i, val in enumerate(param_values):
        if progress_callback:
            progress_callback(i + 1, n)

        # Set parameters based on sweep type
        if config.parameter == "chamber_pressure":
            Pc = val * 1e5  # bar to Pa
            eps = config.base_expansion_ratio
            At = config.base_throat_area_cm2 * 1e-4  # cm² to m²
        elif config.parameter == "expansion_ratio":
            Pc = config.base_chamber_pressure_bar * 1e5
            eps = val
            At = config.base_throat_area_cm2 * 1e-4
        else:
            raise ValueError(f"Unknown parameter: {config.parameter}")

        # Create nozzle conditions for vacuum
        nozzle_vac = NozzleConditions(
            area_ratio=eps, chamber_pressure=Pc, ambient_pressure=0.0, throat_area=At
        )

        # Calculate vacuum performance
        perf_vac = calculate_performance(
            T_chamber=temperature,
            P_chamber=Pc,
            gamma=gamma,
            mean_molecular_weight=mean_mw,
            nozzle=nozzle_vac,
        )

        # Create nozzle conditions for sea level
        nozzle_sl = NozzleConditions(
            area_ratio=eps, chamber_pressure=Pc, ambient_pressure=101325.0, throat_area=At
        )

        perf_sl = calculate_performance(
            T_chamber=temperature,
            P_chamber=Pc,
            gamma=gamma,
            mean_molecular_weight=mean_mw,
            nozzle=nozzle_sl,
        )

        isp_vac[i] = perf_vac.isp
        isp_sl[i] = perf_sl.isp
        thrust_arr[i] = perf_vac.thrust
        c_star_arr[i] = perf_vac.c_star
        temps[i] = temperature
        exit_mach[i] = perf_vac.exit_mach

    return SweepResult(
        parameter_name=config.parameter,
        parameter_values=param_values,
        isp_vacuum=isp_vac,
        isp_sea_level=isp_sl,
        thrust=thrust_arr,
        c_star=c_star_arr,
        temperature=temps,
        exit_mach=exit_mach,
    )


def get_sweep_parameters() -> list[tuple[str, str, str]]:
    """
    Get available sweep parameters.

    Returns:
        List of (key, label, unit) tuples
    """
    return [
        ("chamber_pressure", "Chamber Pressure", "bar"),
        ("expansion_ratio", "Expansion Ratio", "ε"),
    ]
