"""
Frozen-composition nozzle expansion with temperature-dependent heat capacity.

The chamber mixture keeps its composition and expands isentropically:
s(T, p) = s_chamber, with s and h from the NASA polynomials of every
product species. The throat is where the mass flux rho*v peaks, and the exit
pressure follows from A_e/A_t = (rho v)_t / (rho v)_e. This is NASA CEA's
"frozen" rocket problem; unlike the constant-gamma relations in
:mod:`ensim.core.propulsion`, it keeps the drop of cp as the gas cools.

References:
    Gordon & McBride, NASA RP-1311 (1994), part I, section 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .constants import G0, GAS_CONSTANT, P_REF
from .types import EquilibriumResult, SpeciesDatabase


@dataclass(frozen=True)
class FrozenStation:
    """Flow state at one nozzle station, per unit mass of propellant."""

    pressure: float  # Pa
    temperature: float  # K
    velocity: float  # m/s
    density: float  # kg/m³
    mach: float

    @property
    def mass_flux(self) -> float:
        return self.density * self.velocity


@dataclass(frozen=True)
class FrozenNozzle:
    """Ideal (loss-free) frozen expansion to one area ratio."""

    throat: FrozenStation
    exit: FrozenStation
    area_ratio: float
    c_star: float  # m/s
    chamber_pressure: float  # Pa

    def thrust_coefficient(self, ambient_pressure: float = 0.0) -> float:
        momentum = self.exit.velocity / self.c_star
        pressure = (self.exit.pressure - ambient_pressure) * self.area_ratio / self.chamber_pressure
        return momentum + pressure

    def isp(self, ambient_pressure: float = 0.0) -> float:
        return self.thrust_coefficient(ambient_pressure) * self.c_star / G0


class _Mixture:
    """Frozen ideal-gas mixture built from an equilibrium result."""

    def __init__(self, equilibrium: EquilibriumResult, database: SpeciesDatabase):
        keep = equilibrium.mole_fractions > 1e-12
        names = [n for n, k in zip(equilibrium.species_names, keep, strict=True) if k]
        self.x = equilibrium.mole_fractions[keep] / equilibrium.mole_fractions[keep].sum()
        species = [database[name] for name in names]
        self.low = np.array([s.coeffs_low for s in species])
        self.high = np.array([s.coeffs_high for s in species])
        self.t_mid = np.array([s.t_mid for s in species])
        self.molar_mass = float(np.dot(self.x, [s.molecular_weight for s in species])) / 1000.0
        positive = self.x > 0.0
        self._mixing = float(np.sum(self.x[positive] * np.log(self.x[positive])))

    def _coeffs(self, temperature: float) -> np.ndarray:
        return np.where((temperature >= self.t_mid)[:, None], self.high, self.low)

    def cp(self, temperature: float) -> float:
        """J/(mol K)."""
        c = self._coeffs(temperature)
        t = temperature
        cp_r = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * (c[:, 3] + t * c[:, 4])))
        return float(np.dot(self.x, cp_r)) * GAS_CONSTANT

    def enthalpy(self, temperature: float) -> float:
        """J/mol."""
        c = self._coeffs(temperature)
        t = temperature
        h_rt = (
            c[:, 0]
            + t * (c[:, 1] / 2 + t * (c[:, 2] / 3 + t * (c[:, 3] / 4 + t * c[:, 4] / 5)))
            + c[:, 5] / t
        )
        return float(np.dot(self.x, h_rt)) * GAS_CONSTANT * t

    def s0_over_r(self, temperature: float) -> float:
        """Mixture standard-state entropy over R, without mixing or pressure terms."""
        c = self._coeffs(temperature)
        t = temperature
        s_r = (
            c[:, 0] * np.log(t)
            + t * (c[:, 1] + t * (c[:, 2] / 2 + t * (c[:, 3] / 3 + t * c[:, 4] / 4)))
            + c[:, 6]
        )
        return float(np.dot(self.x, s_r))

    def entropy(self, temperature: float, pressure: float) -> float:
        """J/(mol K), ideal mixture."""
        return (
            self.s0_over_r(temperature) - self._mixing - np.log(pressure / P_REF)
        ) * GAS_CONSTANT


class _Isentrope:
    """The frozen isentrope through the chamber state, parametrised by temperature.

    Along s = const the pressure follows in closed form,
    ln(p / p0) = s0(T)/R - sum(x ln x) - s_c/R, so every quantity is an
    explicit function of T and no inner iteration is needed.
    """

    def __init__(self, equilibrium: EquilibriumResult, database: SpeciesDatabase):
        self.mix = _Mixture(equilibrium, database)
        self.tc = equilibrium.temperature
        self.pc = equilibrium.pressure
        self._s_c_r = self.mix.entropy(self.tc, self.pc) / GAS_CONSTANT
        self._h_c = self.mix.enthalpy(self.tc)

    def station(self, temperature: float) -> FrozenStation:
        mix = self.mix
        pressure = P_REF * np.exp(mix.s0_over_r(temperature) - mix._mixing - self._s_c_r)
        drop = self._h_c - mix.enthalpy(temperature)
        velocity = np.sqrt(max(2.0 * drop / mix.molar_mass, 0.0))
        cp = mix.cp(temperature)
        sound = np.sqrt(cp / (cp - GAS_CONSTANT) * GAS_CONSTANT * temperature / mix.molar_mass)
        density = pressure * mix.molar_mass / (GAS_CONSTANT * temperature)
        return FrozenStation(
            float(pressure), temperature, float(velocity), density, float(velocity / sound)
        )

    def throat(self) -> FrozenStation:
        best = minimize_scalar(
            lambda t: -self.station(t).mass_flux,
            bounds=(0.6 * self.tc, 0.999 * self.tc),
            method="bounded",
            options={"xatol": 1e-9 * self.tc},
        )
        return self.station(float(best.x))

    def supersonic_temperature(self, throat: FrozenStation, area_ratio: float) -> float:
        target = np.log(area_ratio)

        def residual(t: float) -> float:
            return np.log(throat.mass_flux / self.station(t).mass_flux) - target

        lower = 0.5 * throat.temperature
        while residual(lower) < 0.0:
            lower *= 0.5
            if lower < 5.0:
                raise ValueError("Area ratio is beyond the reach of the frozen expansion")
        return brentq(residual, lower, throat.temperature, xtol=1e-10 * throat.temperature)


def frozen_nozzle(
    equilibrium: EquilibriumResult, database: SpeciesDatabase, area_ratio: float
) -> FrozenNozzle:
    """Expand the chamber state of ``equilibrium`` to ``area_ratio`` with frozen composition."""
    if area_ratio <= 1.0:
        raise ValueError("Area ratio must be greater than one")
    isentrope = _Isentrope(equilibrium, database)
    throat = isentrope.throat()
    exit_station = isentrope.station(isentrope.supersonic_temperature(throat, area_ratio))
    return FrozenNozzle(
        throat=throat,
        exit=exit_station,
        area_ratio=area_ratio,
        c_star=isentrope.pc / throat.mass_flux,
        chamber_pressure=isentrope.pc,
    )


def frozen_profile(
    equilibrium: EquilibriumResult,
    database: SpeciesDatabase,
    area_ratios: np.ndarray,
    supersonic: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flow state at stations of known A/A*, subsonic or supersonic per station.

    The isentrope is tabulated once on a fine temperature grid and each
    station interpolated in ln(A/A*). Returns arrays ``pressure``,
    ``temperature``, ``mach`` and ``velocity``.
    """
    area_ratios = np.asarray(area_ratios, dtype=float)
    supersonic = np.asarray(supersonic, dtype=bool)
    if area_ratios.shape != supersonic.shape or np.any(area_ratios < 1.0 - 1e-9):
        raise ValueError("Area ratios must be at least one and match the branch flags")
    isentrope = _Isentrope(equilibrium, database)
    throat = isentrope.throat()
    t_throat = throat.temperature
    largest = float(area_ratios[supersonic].max()) if supersonic.any() else 1.0
    t_exit = isentrope.supersonic_temperature(throat, largest) if largest > 1.0 + 1e-9 else t_throat

    def table(temperatures: np.ndarray):
        stations = [isentrope.station(float(t)) for t in temperatures]
        log_area = np.log(throat.mass_flux / np.array([s.mass_flux for s in stations]))
        columns = {
            "pressure": np.array([s.pressure for s in stations]),
            "temperature": temperatures,
            "mach": np.array([s.mach for s in stations]),
            "velocity": np.array([s.velocity for s in stations]),
        }
        return np.maximum(log_area, 0.0), columns

    # Supersonic branch: area grows as T falls from the throat to the exit
    super_t = np.geomspace(t_throat, t_exit, 400) if t_exit < t_throat else np.array([t_throat])
    super_area, super_cols = table(super_t)
    # Subsonic branch: area grows as T rises from the throat towards Tc
    sub_t = t_throat + (isentrope.tc - t_throat) * (1.0 - np.geomspace(1.0, 1e-7, 400))
    sub_area, sub_cols = table(sub_t)

    out = {key: np.empty(len(area_ratios)) for key in super_cols}
    log_ratio = np.log(np.maximum(area_ratios, 1.0))
    for key in out:
        out[key][supersonic] = np.interp(log_ratio[supersonic], super_area, super_cols[key])
        out[key][~supersonic] = np.interp(log_ratio[~supersonic], sub_area, sub_cols[key])
    return out
