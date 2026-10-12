"""
Engine analysis in one call: chamber equilibrium, then nozzle performance.

    >>> from ensim.core.engine import EngineInputs, analyze_engine
    >>> result = analyze_engine(EngineInputs("RP-1", "LOX", of_ratio=2.6,
    ...                                      chamber_pressure=7e6, expansion_ratio=25))
    >>> round(result.vacuum.isp)  # doctest: +SKIP
    326

Propellants are named by the keys of :data:`ensim.core.propellants.PROPELLANTS`
(LH2, LOX, RP-1, NTO, ...) and enter the chamber in their storage state. Any
other name is looked up in the thermo database and treated as a gas.

The chamber composition comes from Gibbs energy minimisation. The nozzle
expands that composition frozen, with temperature-dependent heat capacity
(:mod:`ensim.core.frozen_flow`); this matches NASA CEA's frozen rocket problem
and slightly understates Isp compared with shifting equilibrium.

Efficiencies scale the ideal result: c* by ``eta_cstar``, the thrust
coefficient by ``eta_cf``, and the momentum term by the conical divergence
factor (1 + cos alpha) / 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .chemistry import CombustionProblem
from .constants import G0, P_ATM
from .frozen_flow import FrozenNozzle, frozen_nozzle
from .propellants import PROPELLANTS
from .propulsion import PerformanceResult
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
    fuel_temperature: float | None = None  # K, gas reactants only (default 298.15)
    oxidizer_temperature: float | None = None  # K, gas reactants only (default 298.15)

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
    nozzle: FrozenNozzle | None = None  # ideal frozen expansion behind the three results
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

    def performance_at(self, ambient_pressure: float) -> PerformanceResult | None:
        """Performance at another back pressure; None if the flow cannot stay choked."""
        if self.nozzle is None:
            raise ValueError("This result has no nozzle solution")
        try:
            return _performance(self.inputs, self.nozzle, ambient_pressure)
        except ValueError:
            return None

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


def _performance(inputs: EngineInputs, nozzle: FrozenNozzle, ambient: float) -> PerformanceResult:
    if ambient > nozzle.throat.pressure:
        raise ValueError("Ambient pressure is too high for choked nozzle flow")
    divergence = 0.5 * (1.0 + np.cos(np.radians(inputs.nozzle_half_angle_deg)))
    momentum = divergence * nozzle.exit.velocity / nozzle.c_star
    pressure = (nozzle.exit.pressure - ambient) * nozzle.area_ratio / inputs.chamber_pressure
    c_star = nozzle.c_star * inputs.eta_cstar
    c_f = inputs.eta_cf * (momentum + pressure)
    thrust = mass_flow = None
    if inputs.throat_area is not None:
        thrust = c_f * inputs.chamber_pressure * inputs.throat_area
        mass_flow = inputs.chamber_pressure * inputs.throat_area / c_star
    return PerformanceResult(
        c_star=c_star,
        c_f=c_f,
        exit_velocity=nozzle.exit.velocity,
        exit_mach=nozzle.exit.mach,
        exit_pressure=nozzle.exit.pressure,
        exit_temperature=nozzle.exit.temperature,
        thrust=thrust if thrust is not None else c_f * inputs.chamber_pressure,
        isp=c_f * c_star / G0,
        mass_flow_rate=mass_flow,
    )


@dataclass(frozen=True)
class _Reactant:
    label: str
    molar_mass: float  # kg/mol
    kwargs: dict


def _reactant(
    role: str, name: str, temperature: float | None, database: SpeciesDatabase
) -> _Reactant:
    propellant = PROPELLANTS.get(name.upper())
    if propellant is not None:
        if temperature is not None and abs(temperature - propellant.temperature) > 1e-6:
            raise ValueError(
                f"{propellant.name} is tabulated at {propellant.temperature:g} K; "
                "use a database species name for a gas at another temperature"
            )
        return _Reactant(
            propellant.key,
            propellant.molar_mass,
            {
                "temperature": propellant.temperature,
                "formula": propellant.formula,
                "molar_enthalpy": propellant.molar_enthalpy,
            },
        )
    if name not in database:
        raise KeyError(f"{role} '{name}' is neither a known propellant nor a database species")
    return _Reactant(
        name,
        database[name].molecular_weight / 1000.0,
        {"temperature": 298.15 if temperature is None else temperature},
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
    fuel = _reactant("Fuel", inputs.fuel, inputs.fuel_temperature, database)
    oxidizer = _reactant("Oxidizer", inputs.oxidizer, inputs.oxidizer_temperature, database)

    # One kilogram of fuel and of_ratio kilograms of oxidizer
    problem = CombustionProblem(database)
    problem.add_fuel(fuel.label, moles=1.0 / fuel.molar_mass, **fuel.kwargs)
    problem.add_oxidizer(
        oxidizer.label, moles=inputs.of_ratio / oxidizer.molar_mass, **oxidizer.kwargs
    )
    equilibrium = problem.solve(pressure=inputs.chamber_pressure)
    if not equilibrium.converged:
        raise ConvergenceError(
            f"Chamber equilibrium did not converge for {inputs.fuel}/{inputs.oxidizer} "
            f"at O/F {inputs.of_ratio:g} and {inputs.chamber_pressure / 1e5:g} bar"
        )

    nozzle = frozen_nozzle(equilibrium, database, inputs.expansion_ratio)
    vacuum = _performance(inputs, nozzle, 0.0)
    operating = _performance(inputs, nozzle, inputs.ambient_pressure)
    try:
        sea_level = _performance(inputs, nozzle, P_ATM)
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
        nozzle=nozzle,
        composition=composition,
    )
