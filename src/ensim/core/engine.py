"""
Engine analysis in one call: chamber equilibrium, then nozzle performance.

    >>> from ensim.core.engine import EngineInputs, analyze_engine
    >>> result = analyze_engine(EngineInputs("RP1", "O2", of_ratio=2.6,
    ...                                      chamber_pressure=7e6, expansion_ratio=25))
    >>> round(result.vacuum.isp)  # doctest: +SKIP
    336

The chamber composition comes from Gibbs energy minimisation; the nozzle
expands it frozen (no recombination), which slightly understates Isp
compared with a shifting-equilibrium calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .chemistry import CombustionProblem
from .constants import P_ATM
from .propulsion import NozzleConditions, PerformanceResult, calculate_performance
from .types import ConvergenceError, EquilibriumResult, SpeciesDatabase


@dataclass(frozen=True)
class EngineInputs:
    """Operating point of a bipropellant engine. SI units."""

    fuel: str
    oxidizer: str
    of_ratio: float  # oxidizer-to-fuel mass ratio
    chamber_pressure: float  # Pa
    expansion_ratio: float  # Ae/At
    throat_area: float | None = None  # m², needed for thrust and mass flow
    ambient_pressure: float = 0.0  # Pa, the operating point reported as `operating`
    eta_cstar: float = 1.0
    eta_cf: float = 1.0
    nozzle_half_angle_deg: float = 15.0
    fuel_temperature: float = 298.15  # K
    oxidizer_temperature: float = 298.15  # K

    def __post_init__(self) -> None:
        if self.of_ratio <= 0.0:
            raise ValueError("O/F ratio must be positive")
        if self.chamber_pressure <= 0.0:
            raise ValueError("Chamber pressure must be positive")
        if self.expansion_ratio <= 1.0:
            raise ValueError("Expansion ratio must be greater than one")
        if self.throat_area is not None and self.throat_area <= 0.0:
            raise ValueError("Throat area must be positive")
        if self.ambient_pressure < 0.0:
            raise ValueError("Ambient pressure cannot be negative")


@dataclass(frozen=True)
class EngineResult:
    """Chamber state and nozzle performance at three back pressures."""

    inputs: EngineInputs
    equilibrium: EquilibriumResult
    vacuum: PerformanceResult
    operating: PerformanceResult
    sea_level: PerformanceResult | None  # None if the nozzle cannot run choked at 1 atm
    composition: dict[str, float] = field(default_factory=dict)  # mole fractions >= 1e-6

    @property
    def chamber_temperature(self) -> float:
        return self.equilibrium.temperature

    @property
    def gamma(self) -> float:
        return self.equilibrium.gamma

    @property
    def molecular_weight(self) -> float:
        return self.equilibrium.mean_molecular_weight

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation (SI units)."""

        def performance(value: PerformanceResult | None) -> dict[str, Any] | None:
            return None if value is None else {k: _plain(v) for k, v in asdict(value).items()}

        return {
            "inputs": asdict(self.inputs),
            "chamber": {
                "temperature": self.chamber_temperature,
                "pressure": self.inputs.chamber_pressure,
                "gamma": self.gamma,
                "molecular_weight": self.molecular_weight,
                "composition": dict(self.composition),
            },
            "vacuum": performance(self.vacuum),
            "sea_level": performance(self.sea_level),
            "operating": performance(self.operating),
        }


def _plain(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _performance(
    inputs: EngineInputs, equilibrium: EquilibriumResult, ambient: float
) -> PerformanceResult:
    return calculate_performance(
        T_chamber=equilibrium.temperature,
        P_chamber=inputs.chamber_pressure,
        gamma=equilibrium.gamma,
        mean_molecular_weight=equilibrium.mean_molecular_weight,
        nozzle=NozzleConditions(
            area_ratio=inputs.expansion_ratio,
            chamber_pressure=inputs.chamber_pressure,
            ambient_pressure=ambient,
            throat_area=inputs.throat_area,
        ),
        eta_cstar=inputs.eta_cstar,
        eta_cf=inputs.eta_cf,
        alpha_deg=inputs.nozzle_half_angle_deg,
    )


def analyze_engine(inputs: EngineInputs, database: SpeciesDatabase | None = None) -> EngineResult:
    """Solve the chamber equilibrium and the nozzle for ``inputs``.

    Args:
        inputs: Operating point.
        database: Species data; the packaged NASA Glenn set by default.

    Raises:
        KeyError: A propellant is not in the database.
        ConvergenceError: The equilibrium solver did not converge.
        ValueError: The operating ambient pressure is too high for choked flow.
    """
    if database is None:
        from ensim.utils.nasa_parser import load_default_database

        database = load_default_database()
    for role, name in (("Fuel", inputs.fuel), ("Oxidizer", inputs.oxidizer)):
        if name not in database:
            raise KeyError(f"{role} '{name}' is not in the thermodynamic database")

    fuel_mw = database[inputs.fuel].molecular_weight
    oxidizer_mw = database[inputs.oxidizer].molecular_weight
    problem = CombustionProblem(database)
    problem.add_fuel(inputs.fuel, moles=1.0, temperature=inputs.fuel_temperature)
    problem.add_oxidizer(
        inputs.oxidizer,
        moles=inputs.of_ratio * fuel_mw / oxidizer_mw,
        temperature=inputs.oxidizer_temperature,
    )
    equilibrium = problem.solve(pressure=inputs.chamber_pressure)
    if not equilibrium.converged:
        raise ConvergenceError(
            f"Chamber equilibrium did not converge for {inputs.fuel}/{inputs.oxidizer} "
            f"at O/F {inputs.of_ratio:g} and {inputs.chamber_pressure / 1e5:g} bar"
        )

    vacuum = _performance(inputs, equilibrium, 0.0)
    operating = _performance(inputs, equilibrium, inputs.ambient_pressure)
    try:
        sea_level = _performance(inputs, equilibrium, P_ATM)
    except ValueError:
        sea_level = None

    composition = {
        name: float(fraction)
        for name, fraction in sorted(
            zip(equilibrium.species_names, equilibrium.mole_fractions, strict=True),
            key=lambda item: -item[1],
        )
        if fraction >= 1e-6
    }
    return EngineResult(
        inputs=inputs,
        equilibrium=equilibrium,
        vacuum=vacuum,
        operating=operating,
        sea_level=sea_level,
        composition=composition,
    )
