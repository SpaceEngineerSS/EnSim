"""Regenerate tests/reference/cea_liquid_rockets.csv with NASA CEA.

Requires NASA's official Python package (``pip install cea==3.3.2``), which
is not an EnSim dependency. Each case is a rocket problem with the
reactants as stored: cryogens at their normal boiling points, storables
as liquids at 298.15 K. Composition is frozen at the chamber (n_frz=1), the
same assumption EnSim's nozzle model makes.

    python scripts/generate_cea_references.py
"""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

import cea
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "reference" / "cea_liquid_rockets.csv"

# (case, fuel, fuel T, oxidizer, oxidizer T, O/F, Pc bar, Ae/At)
CASES = [
    ("hydrolox", "H2(L)", 20.27, "O2(L)", 90.17, 6.0, 68.9476, 40.0),
    ("methalox", "CH4(L)", 111.643, "O2(L)", 90.17, 3.4, 100.0, 35.0),
    ("kerolox", "RP-1", 298.15, "O2(L)", 90.17, 2.6, 100.0, 25.0),
    ("hydrazine_nto", "N2H4(L)", 298.15, "N2O4(L)", 298.15, 1.3, 20.0, 60.0),
    ("mmh_nto", "CH6N2(L)", 298.15, "N2O4(L)", 298.15, 2.0, 10.0, 100.0),
]

FIELDS = [
    "case",
    "fuel",
    "oxidizer",
    "of_ratio",
    "chamber_pressure_bar",
    "area_ratio",
    "chamber_temperature",
    "molecular_weight",
    "gamma_frozen",
    "c_star",
    "isp_vacuum_frozen",
]


def enthalpy(name: str, temperature: float) -> float:
    """Mass-specific enthalpy of a single reactant, J/kg."""
    return cea.Mixture([name]).calc_property(cea.ENTHALPY, np.array([1.0]), temperature)


def run_case(fuel, fuel_t, oxidizer, oxidizer_t, of_ratio, pc_bar, area_ratio):
    reactants = cea.Mixture([fuel, oxidizer])
    products = cea.Mixture([fuel, oxidizer], products_from_reactants=True)
    weights = reactants.of_ratio_to_weights(np.array([0.0, 1.0]), np.array([1.0, 0.0]), of_ratio)
    h_mixture = (enthalpy(fuel, fuel_t) + of_ratio * enthalpy(oxidizer, oxidizer_t)) / (
        1.0 + of_ratio
    )

    solver = cea.RocketSolver(products, reactants=reactants)
    solution = cea.RocketSolution(solver)
    solver.solve(
        solution, weights, pc_bar, supar=[area_ratio], iac=True, n_frz=1, hc=h_mixture / cea.R
    )
    if not solution.converged:
        raise RuntimeError(f"CEA did not converge for {fuel}/{oxidizer}")

    eq = cea.EqSolver(products, reactants=reactants)
    chamber = cea.EqSolution(eq)
    eq.solve(chamber, cea.HP, h_mixture / cea.R, pc_bar, weights)
    return {
        "chamber_temperature": chamber.T,
        "molecular_weight": chamber.MW,
        "gamma_frozen": chamber.cp_fr / chamber.cv_fr,
        "c_star": solution.c_star[-1],
        "isp_vacuum_frozen": solution.Isp_vacuum[-1] / 9.80665,
    }


def main() -> None:
    rows = []
    for case, fuel, fuel_t, oxidizer, oxidizer_t, of_ratio, pc_bar, area_ratio in CASES:
        values = run_case(fuel, fuel_t, oxidizer, oxidizer_t, of_ratio, pc_bar, area_ratio)
        rows.append(
            {
                "case": case,
                "fuel": fuel,
                "oxidizer": oxidizer,
                "of_ratio": of_ratio,
                "chamber_pressure_bar": pc_bar,
                "area_ratio": area_ratio,
                **{key: f"{value:.6f}" for key, value in values.items()},
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# NASA CEA {version('cea')} rocket problems, frozen at the chamber\n")
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
