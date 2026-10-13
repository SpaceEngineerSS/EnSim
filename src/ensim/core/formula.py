"""Element counts of chemical formulas. No heavy imports: the GUI uses it at start-up."""

from __future__ import annotations

import re


def parse_formula(formula: str) -> dict[str, float]:
    """Element counts of a formula such as "N2H4" or "CH1.9423"."""
    formula = re.sub(r"\([GLSCgl]\)$", "", formula)
    elements: dict[str, float] = {}
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        elements[symbol] = elements.get(symbol, 0.0) + (float(count) if count else 1.0)
    return elements
