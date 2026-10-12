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

    def entropy(self, temperature: float, pressure: float) -> float:
        """J/(mol K), ideal mixture."""
        c = self._coeffs(temperature)
        t = temperature
        s_r = (
            c[:, 0] * np.log(t)
            + t * (c[:, 1] + t * (c[:, 2] / 2 + t * (c[:, 3] / 3 + t * c[:, 4] / 4)))
            + c[:, 6]
        )
        return (float(np.dot(self.x, s_r)) - self._mixing - np.log(pressure / P_REF)) * GAS_CONSTANT


def _station(
    mixture: _Mixture, s_chamber: float, h_chamber: float, t_guess: float, pressure: float
) -> FrozenStation:
    temperature = t_guess
    for _ in range(50):
        residual = mixture.entropy(temperature, pressure) - s_chamber
        step = residual * temperature / mixture.cp(temperature)
        temperature = max(temperature - step, 0.5 * temperature)
        if abs(step) < 1e-9 * temperature:
            break
    drop = h_chamber - mixture.enthalpy(temperature)
    velocity = np.sqrt(max(2.0 * drop / mixture.molar_mass, 0.0))
    cp = mixture.cp(temperature)
    gamma = cp / (cp - GAS_CONSTANT)
    sound = np.sqrt(gamma * GAS_CONSTANT * temperature / mixture.molar_mass)
    density = pressure * mixture.molar_mass / (GAS_CONSTANT * temperature)
    return FrozenStation(pressure, temperature, float(velocity), density, float(velocity / sound))


def frozen_nozzle(
    equilibrium: EquilibriumResult, database: SpeciesDatabase, area_ratio: float
) -> FrozenNozzle:
    """Expand the chamber state of ``equilibrium`` to ``area_ratio`` with frozen composition."""
    if area_ratio <= 1.0:
        raise ValueError("Area ratio must be greater than one")
    mixture = _Mixture(equilibrium, database)
    pc = equilibrium.pressure
    tc = equilibrium.temperature
    s_c = mixture.entropy(tc, pc)
    h_c = mixture.enthalpy(tc)

    def at(pressure: float, guess: float) -> FrozenStation:
        return _station(mixture, s_c, h_c, guess, pressure)

    best = minimize_scalar(
        lambda ratio: -at(ratio * pc, 0.9 * tc).mass_flux,
        bounds=(0.3, 0.8),
        method="bounded",
        options={"xatol": 1e-10},
    )
    throat = at(best.x * pc, 0.9 * tc)

    def area_residual(log_ratio: float) -> float:
        station = at(np.exp(log_ratio) * throat.pressure, throat.temperature)
        return np.log(throat.mass_flux / station.mass_flux) - np.log(area_ratio)

    upper = 0.0
    lower = -1.0
    while area_residual(lower) < 0.0:
        lower *= 2.0
        if lower < -60.0:
            raise ValueError("Area ratio is beyond the reach of the frozen expansion")
    exit_log = brentq(area_residual, lower, upper - 1e-12, xtol=1e-13)
    exit_station = at(np.exp(exit_log) * throat.pressure, throat.temperature)
    return FrozenNozzle(
        throat=throat,
        exit=exit_station,
        area_ratio=area_ratio,
        c_star=pc / throat.mass_flux,
        chamber_pressure=pc,
    )
