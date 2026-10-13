"""Draw the data-driven documentation figures.

    python scripts/make_doc_figures.py

Writes docs/assets/figures/cea_deviation.svg from the NASA CEA reference set in
tests/reference. The SVG uses currentColor and var(--signal) so the docs theme
colours it; it is meant to be inlined with a snippet, not loaded as an image.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ensim.core.engine import EngineInputs, analyze_engine

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "cea_liquid_rockets.csv"
OUTPUT = ROOT / "docs" / "assets" / "figures" / "cea_deviation.svg"

CEA_TO_ENSIM = {
    "H2(L)": "LH2",
    "O2(L)": "LOX",
    "CH4(L)": "LCH4",
    "RP-1": "RP-1",
    "N2H4(L)": "N2H4",
    "N2O4(L)": "NTO",
    "CH6N2(L)": "MMH",
}
QUANTITIES = (
    ("Tc", "chamber_temperature", lambda e: e.chamber_temperature),
    ("M", "molecular_weight", lambda e: e.molecular_weight),
    ("c*", "c_star", lambda e: e.vacuum.c_star),
    ("Isp", "isp_vacuum_frozen", lambda e: e.vacuum.isp),
)
LIMIT = 0.15  # percent, half-width of the axis


def deviations() -> list[tuple[str, list[float]]]:
    with REFERENCE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    result = []
    for row in rows:
        engine = analyze_engine(
            EngineInputs(
                CEA_TO_ENSIM[row["fuel"]],
                CEA_TO_ENSIM[row["oxidizer"]],
                float(row["of_ratio"]),
                float(row["chamber_pressure_bar"]) * 1e5,
                float(row["area_ratio"]),
                nozzle_half_angle_deg=0.0,
            )
        )
        label = f"{CEA_TO_ENSIM[row['fuel']]}/{CEA_TO_ENSIM[row['oxidizer']]}"
        values = [100.0 * (get(engine) / float(row[key]) - 1.0) for _, key, get in QUANTITIES]
        result.append((label, values))
    return result


def draw(data: list[tuple[str, list[float]]]) -> str:
    left, top, column, row_height = 110, 46, 150, 34
    width = left + column * len(QUANTITIES) + 10
    height = top + row_height * len(data) + 34
    half = (column - 30) / 2

    def x_of(index: int, value: float) -> float:
        centre = left + column * index + column / 2
        return centre + max(-LIMIT, min(LIMIT, value)) / LIMIT * half

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Deviation of EnSim from NASA CEA" font-family="IBM Plex Sans, sans-serif" '
        'font-size="12" fill="none" stroke="currentColor" stroke-width="1">'
    ]
    for index, (name, _, _) in enumerate(QUANTITIES):
        centre = left + column * index + column / 2
        bottom = top + row_height * len(data) - 8
        parts.append(
            f'<text x="{centre}" y="22" text-anchor="middle" fill="currentColor" stroke="none" '
            f'font-weight="600">{name}</text>'
        )
        parts.append(
            f'<line x1="{centre - half}" y1="{bottom}" x2="{centre + half}" y2="{bottom}" '
            'stroke="var(--line)"/>'
        )
        parts.append(
            f'<line x1="{centre}" y1="{top - 12}" x2="{centre}" y2="{bottom}" '
            'stroke-dasharray="2 3"/>'
        )
        for tick in (-LIMIT, LIMIT):
            x = x_of(index, tick)
            parts.append(f'<line x1="{x}" y1="{bottom}" x2="{x}" y2="{bottom + 4}"/>')
            parts.append(
                f'<text x="{x}" y="{bottom + 17}" text-anchor="middle" font-size="10" '
                f'fill="var(--muted)" stroke="none">{tick:+.2f} %</text>'
            )
    for row, (label, values) in enumerate(data):
        y = top + row_height * row + 8
        parts.append(
            f'<text x="{left - 14}" y="{y + 4}" text-anchor="end" fill="currentColor" '
            f'stroke="none" font-family="IBM Plex Mono, monospace" font-size="11">{label}</text>'
        )
        for index, value in enumerate(values):
            centre = left + column * index + column / 2
            x = x_of(index, value)
            parts.append(
                f'<line x1="{centre}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="var(--signal)" '
                'stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y}" r="3.5" fill="var(--signal)" stroke="none">'
                f"<title>{label} {QUANTITIES[index][0]}: {value:+.4f} %</title></circle>"
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUTPUT.write_text(draw(deviations()), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
