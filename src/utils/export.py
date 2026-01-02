"""
Data Export Utilities.

Export simulation data to various formats for external analysis.
"""

import csv
import os
from datetime import datetime


def export_flight_csv(flight_result, filepath: str) -> bool:
    """
    Export flight simulation data to CSV.

    Columns:
    - Time (s)
    - Altitude (m)
    - Velocity (m/s)
    - Acceleration (m/s²)
    - Mass (kg)
    - Mach
    - Stability Margin (cal)
    - AoA (deg)
    - Thrust (N)
    - Drag (N)

    Args:
        flight_result: FlightResult object
        filepath: Output file path

    Returns:
        True if export successful
    """
    try:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Time (s)',
                'Altitude (m)',
                'Velocity (m/s)',
                'Acceleration (m/s²)',
                'Mass (kg)',
                'Mach',
                'Stability Margin (cal)',
                'AoA (deg)',
                'Thrust (N)',
                'Drag (N)'
            ])

            # Data rows
            n = len(flight_result.time)
            for i in range(n):
                writer.writerow([
                    f"{flight_result.time[i]:.3f}",
                    f"{flight_result.altitude[i]:.2f}",
                    f"{flight_result.velocity[i]:.2f}",
                    f"{flight_result.acceleration[i]:.2f}",
                    f"{flight_result.mass[i]:.3f}",
                    f"{flight_result.mach[i]:.4f}",
                    f"{flight_result.stability_margin[i]:.2f}",
                    f"{flight_result.angle_of_attack[i]:.2f}",
                    f"{flight_result.thrust[i]:.2f}",
                    f"{flight_result.drag[i]:.2f}"
                ])

        return True

    except Exception as e:
        print(f"Export error: {e}")
        return False


def export_summary_txt(flight_result, rocket, filepath: str) -> bool:
    """
    Export flight summary as text report.

    Args:
        flight_result: FlightResult object
        rocket: Rocket configuration
        filepath: Output file path

    Returns:
        True if export successful
    """
    try:
        lines = []
        lines.append("=" * 50)
        lines.append("EnSim Flight Summary Report")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Rocket info
        lines.append("ROCKET CONFIGURATION")
        lines.append("-" * 30)
        lines.append(f"Name: {rocket.name}")
        lines.append(f"Total Length: {rocket.total_length:.3f} m")
        lines.append(f"Diameter: {rocket.reference_diameter:.3f} m")
        lines.append(f"Dry Mass: {rocket.dry_mass:.2f} kg")
        lines.append(f"Wet Mass: {rocket.wet_mass:.2f} kg")
        lines.append("")

        # Flight events
        lines.append("FLIGHT EVENTS")
        lines.append("-" * 30)
        lines.append(f"Liftoff: {flight_result.liftoff_time:.2f} s")
        lines.append(f"Burnout: {flight_result.burnout_time:.2f} s @ {flight_result.burnout_altitude:.0f} m")
        lines.append(f"Apogee: {flight_result.apogee_time:.2f} s @ {flight_result.apogee_altitude:.0f} m")
        lines.append("")

        # Performance
        lines.append("PERFORMANCE")
        lines.append("-" * 30)
        lines.append(f"Max Velocity: {flight_result.max_velocity:.1f} m/s")
        lines.append(f"Max Mach: {flight_result.max_mach:.2f}")
        lines.append(f"Max Acceleration: {flight_result.max_acceleration:.1f} G")
        lines.append(f"Max Dynamic Pressure: {flight_result.max_q:.0f} Pa")
        lines.append(f"Max Angle of Attack: {flight_result.max_aoa:.1f}°")
        lines.append("")

        lines.append("=" * 50)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return True

    except Exception as e:
        print(f"Export error: {e}")
        return False
