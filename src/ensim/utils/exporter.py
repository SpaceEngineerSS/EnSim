"""Export utilities for simulation data."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def export_nozzle_profile_csv(
    profile: dict[str, np.ndarray], filepath: Path, metadata: dict[str, Any] | None = None
) -> bool:
    """
    Export nozzle profile data to CSV.

    Args:
        profile: Dict with 'area_ratio', 'mach', 'temperature', 'pressure', 'velocity'
        filepath: Output file path
        metadata: Optional header metadata (engine name, date, etc.)

    Returns:
        True if export successful
    """
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write metadata header
            if metadata:
                writer.writerow(["# EnSim Nozzle Profile Export"])
                writer.writerow([f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                for key, value in metadata.items():
                    writer.writerow([f"# {key}: {value}"])
                writer.writerow([])

            # Write column headers
            writer.writerow(
                [
                    "Area_Ratio_A/At",
                    "Mach_Number",
                    "Temperature_K",
                    "Pressure_Pa",
                    "Velocity_m/s",
                    "P/Pc",
                    "T/Tc",
                ]
            )

            # Write data rows
            n = len(profile["area_ratio"])
            for i in range(n):
                writer.writerow(
                    [
                        f"{profile['area_ratio'][i]:.4f}",
                        f"{profile['mach'][i]:.4f}",
                        f"{profile['temperature'][i]:.2f}",
                        f"{profile['pressure'][i]:.2f}",
                        f"{profile['velocity'][i]:.2f}",
                        f"{profile['P_ratio'][i]:.6f}",
                        f"{profile['T_ratio'][i]:.6f}",
                    ]
                )

        return True
    except (OSError, KeyError, TypeError, ValueError):
        logger.exception("Could not export nozzle profile to %s", filepath)
        return False


def generate_report_markdown(
    fuel: str,
    oxidizer: str,
    of_ratio: float,
    chamber_pressure_bar: float,
    expansion_ratio: float,
    temperature: float,
    isp_vacuum: float,
    isp_sea_level: float,
    thrust_kN: float,
    c_star: float,
    gamma: float,
    mean_mw: float,
    species_fractions: dict[str, float],
) -> str:
    """
    Generate a formatted simulation report.

    Returns:
        Markdown-formatted report string
    """
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# EnSim Simulation Report
Generated: {date_str}

---

## Engine Configuration

| Parameter | Value |
|-----------|-------|
| Fuel | {fuel} |
| Oxidizer | {oxidizer} |
| O/F Ratio | {of_ratio:.2f} |
| Chamber Pressure | {chamber_pressure_bar:.1f} bar |
| Expansion Ratio (ε) | {expansion_ratio:.1f} |

---

## Performance Results

| Metric | Value |
|--------|-------|
| **Chamber Temperature** | {temperature:.0f} K |
| **Vacuum Isp** | {isp_vacuum:.1f} s |
| **Sea Level Isp** | {isp_sea_level:.1f} s |
| **Thrust (Vacuum)** | {thrust_kN:.2f} kN |
| **C*** | {c_star:.1f} m/s |
| **Gamma (γ)** | {gamma:.4f} |
| **Mean MW** | {mean_mw:.2f} g/mol |

---

## Combustion Products

| Species | Mole Fraction |
|---------|---------------|
"""

    ranked = sorted(species_fractions.items(), key=lambda item: item[1], reverse=True)
    for species, frac in ((name, value) for name, value in ranked if value >= 1e-6):
        report += f"| {species} | {frac * 100:.2f}% |\n"

    report += """
---

## Notes

- Chamber composition from Gibbs energy minimisation (Gordon & McBride, NASA RP-1311)
- Nozzle expansion with the chamber composition frozen
- Product species are the ones listed above; trace species below 1e-6 are omitted

---

This report contains idealized engineering estimates. Review the documented
model limitations before using the results in a design decision.
"""

    return report


def export_report(filepath: Path, **kwargs) -> bool:
    """
    Export simulation report to file.

    Args:
        filepath: Output file path (.md or .txt)
        **kwargs: Report parameters (fuel, oxidizer, etc.)

    Returns:
        True if export successful
    """
    try:
        report = generate_report_markdown(**kwargs)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        return True
    except (OSError, TypeError, ValueError):
        logger.exception("Could not export report to %s", filepath)
        return False
