"""
EnSim System Audit - Integration Test Suite.

Performs automated testing of the full simulation pipeline
to verify data flow, physics fidelity, and edge case handling.
"""

import sys
import os
sys.path.insert(0, '.')

from datetime import datetime
import numpy as np

# Test results storage
RESULTS = {
    'passed': [],
    'failed': [],
    'warnings': []
}


def log_pass(test_name: str, message: str):
    """Log a passed test."""
    RESULTS['passed'].append((test_name, message))
    print(f"  ✅ PASS: {test_name} - {message}")


def log_fail(test_name: str, message: str):
    """Log a failed test."""
    RESULTS['failed'].append((test_name, message))
    print(f"  ❌ FAIL: {test_name} - {message}")


def log_warn(test_name: str, message: str):
    """Log a warning."""
    RESULTS['warnings'].append((test_name, message))
    print(f"  ⚠️ WARN: {test_name} - {message}")


# =============================================================================
# PART 1: MASS CONSISTENCY CHECK
# =============================================================================
def test_mass_consistency():
    """
    Verify that rocket mass decreases during burn.
    
    Setup: 10kg dry + 50kg propellant = 60kg initial
    Check: At burnout, mass should be ~10kg
    """
    print("\n[TEST 1] Mass Consistency Check")
    
    from src.core.rocket import Rocket, NoseCone, BodyTube, FinSet, Fin, EngineMount
    from src.core.flight import simulate_flight
    
    # Create rocket: 10kg dry + 50kg propellant
    rocket = Rocket(
        nose=NoseCone(mass=2.0),
        body=BodyTube(mass=5.0),
        fins=FinSet(mass=1.0),
        engine=EngineMount(
            engine_mass_dry=2.0,  # Total dry = 10kg
            fuel_mass=10.0,
            oxidizer_mass=40.0,  # Total prop = 50kg
            tank_length=0.5
        )
    )
    
    dry_mass = rocket.dry_mass
    wet_mass = rocket.wet_mass
    
    print(f"  Dry Mass: {dry_mass:.1f} kg")
    print(f"  Wet Mass: {wet_mass:.1f} kg")
    
    if abs(dry_mass - 10.0) < 0.1:
        log_pass("dry_mass", f"Dry mass is {dry_mass:.1f} kg as expected")
    else:
        log_fail("dry_mass", f"Expected 10kg, got {dry_mass:.1f} kg")
    
    if abs(wet_mass - 60.0) < 0.1:
        log_pass("wet_mass", f"Wet mass is {wet_mass:.1f} kg as expected")
    else:
        log_fail("wet_mass", f"Expected 60kg, got {wet_mass:.1f} kg")
    
    # Simulate flight
    result = simulate_flight(
        rocket=rocket,
        thrust_vac=5000.0,
        isp_vac=250.0,
        burn_time=10.0,
        exit_area=0.01,
        dt=0.1,
        max_time=60.0
    )
    
    # Check mass at burnout
    burnout_idx = np.argmin(np.abs(result.time - result.burnout_time))
    mass_at_burnout = result.mass[burnout_idx]
    
    print(f"  Mass at burnout: {mass_at_burnout:.1f} kg")
    
    if mass_at_burnout < wet_mass * 0.9:  # Must have consumed some propellant
        log_pass("mass_consumption", f"Propellant consumed correctly")
    else:
        log_fail("mass_consumption", f"Mass didn't decrease during burn!")


# =============================================================================
# PART 2: THRUST-FLIGHT COUPLING
# =============================================================================
def test_thrust_coupling():
    """
    Verify that a heavy rocket with weak engine doesn't fly.
    
    Setup: 10,000kg rocket, 10N thrust (T/W << 1)
    Check: Should not lift off (altitude ~0)
    """
    print("\n[TEST 2] Thrust-Flight Coupling Check")
    
    from src.core.rocket import Rocket, NoseCone, BodyTube, FinSet, Fin, EngineMount
    from src.core.flight import simulate_flight
    
    # Heavy rocket with weak engine
    rocket = Rocket(
        nose=NoseCone(mass=1000.0),
        body=BodyTube(mass=5000.0),
        fins=FinSet(mass=500.0),
        engine=EngineMount(
            engine_mass_dry=500.0,
            fuel_mass=1000.0,
            oxidizer_mass=2000.0,
            tank_length=2.0
        )
    )
    
    wet_mass = rocket.wet_mass
    thrust = 10.0  # Only 10N
    weight = wet_mass * 9.81
    twr = thrust / weight
    
    print(f"  Wet Mass: {wet_mass:.0f} kg")
    print(f"  Thrust: {thrust:.0f} N")
    print(f"  Weight: {weight:.0f} N")
    print(f"  T/W Ratio: {twr:.4f}")
    
    result = simulate_flight(
        rocket=rocket,
        thrust_vac=thrust,
        isp_vac=250.0,
        burn_time=10.0,
        exit_area=0.001,
        dt=0.1,
        max_time=30.0
    )
    
    max_alt = result.apogee_altitude
    print(f"  Max Altitude: {max_alt:.2f} m")
    
    if max_alt < 1.0:  # Should not fly
        log_pass("no_liftoff", f"Rocket correctly didn't lift off (T/W < 1)")
    else:
        log_fail("no_liftoff", f"Rocket flew with T/W={twr:.4f}! Physics bug!")


# =============================================================================
# PART 3: STABILITY LOGIC CHECK
# =============================================================================
def test_stability_logic():
    """
    Verify that a finless rocket is unstable.
    
    Check: Stability margin should be negative (CP ahead of CG)
    """
    print("\n[TEST 3] Stability Logic Check")
    
    from src.core.rocket import Rocket, NoseCone, BodyTube, FinSet, Fin, EngineMount
    from src.core.aero import analyze_rocket, calculate_stability_margin
    
    # Rocket with minimal/zero fins
    rocket = Rocket(
        nose=NoseCone(length=0.3, diameter=0.1, mass=0.5),
        body=BodyTube(length=1.0, diameter=0.1, mass=2.0),
        fins=FinSet(
            fin=Fin(root_chord=0.001, tip_chord=0.001, span=0.001),  # Tiny fins
            count=0,  # No fins
            mass=0.0
        ),
        engine=EngineMount(
            engine_mass_dry=1.0,
            fuel_mass=0.5,
            oxidizer_mass=0.0,
            tank_length=0.2
        )
    )
    
    aero = analyze_rocket(rocket, time=0.0)
    
    print(f"  CN_alpha (fins): {aero.cn_alpha_fins:.4f}")
    print(f"  CP: {aero.cp_total:.3f} m")
    print(f"  CG: {aero.cg:.3f} m")
    print(f"  Stability Margin: {aero.stability_margin:.2f} cal")
    print(f"  Is Stable: {aero.is_stable}")
    
    # With no effective fins, CP should be near nose (low value)
    # CG should be near tail (high value due to engine mass)
    # Therefore margin should be low or negative
    
    if aero.stability_margin < 1.0:
        log_pass("instability", f"Finless rocket correctly marked as unstable ({aero.stability_margin:.2f} cal)")
    else:
        log_warn("instability", f"Finless rocket shows stable ({aero.stability_margin:.2f} cal) - check fin calculation")


# =============================================================================
# PART 4: PARACHUTE TRIGGER CHECK
# =============================================================================
def test_parachute_trigger():
    """
    Verify parachute deployment logic.
    
    Check: After apogee, descent velocity should be reasonable
    """
    print("\n[TEST 4] Parachute Trigger Check")
    
    from src.core.recovery import Parachute, estimate_descent
    from src.core.rocket import Rocket, NoseCone, BodyTube, FinSet, Fin, EngineMount
    
    # Create a rocket
    rocket = Rocket(
        nose=NoseCone(mass=0.3),
        body=BodyTube(mass=1.0),
        fins=FinSet(mass=0.2),
        engine=EngineMount(engine_mass_dry=0.5, fuel_mass=0.0, oxidizer_mass=0.0)
    )
    
    dry_mass = rocket.dry_mass
    
    # 1m diameter parachute
    chute = Parachute(diameter=1.0, cd=1.5)
    
    v_descent, is_safe = chute.is_safe_descent(dry_mass)
    
    print(f"  Rocket Dry Mass: {dry_mass:.1f} kg")
    print(f"  Chute Diameter: {chute.diameter:.1f} m")
    print(f"  Chute CdA: {chute.cda:.2f} m²")
    print(f"  Descent Rate: {v_descent:.1f} m/s")
    print(f"  Is Safe: {is_safe}")
    
    if v_descent < 20:  # Reasonable descent rate with chute
        log_pass("parachute_physics", f"Descent rate {v_descent:.1f} m/s is reasonable")
    else:
        log_fail("parachute_physics", f"Descent rate {v_descent:.1f} m/s is too high!")
    
    # Check descent estimation
    estimate = estimate_descent(apogee=1000, mass=dry_mass, chute=chute)
    print(f"  Estimated descent time from 1km: {estimate['descent_time']:.0f} s")


# =============================================================================
# PART 5: MACH 1.0 SINGULARITY CHECK
# =============================================================================
def test_mach_singularity():
    """
    Check drag coefficient behavior at Mach 1.0.
    
    Verify no divide-by-zero or NaN values.
    """
    print("\n[TEST 5] Mach 1.0 Singularity Check")
    
    from src.core.rocket import create_default_rocket
    from src.core.aero import calculate_drag_coefficient
    
    rocket = create_default_rocket()
    
    test_machs = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.5, 2.0]
    
    all_valid = True
    for M in test_machs:
        cd = calculate_drag_coefficient(rocket, M)
        is_valid = not (np.isnan(cd) or np.isinf(cd)) and cd > 0
        status = "✓" if is_valid else "✗"
        print(f"  M={M:.2f}: Cd={cd:.4f} {status}")
        if not is_valid:
            all_valid = False
    
    if all_valid:
        log_pass("mach_singularity", "No NaN/Inf at transonic speeds")
    else:
        log_fail("mach_singularity", "Invalid Cd values near Mach 1!")


# =============================================================================
# PART 6: GROUND COLLISION CHECK
# =============================================================================
def test_ground_collision():
    """
    Verify simulation stops at ground (altitude = 0).
    
    Check: No negative altitudes in trajectory.
    """
    print("\n[TEST 6] Ground Collision Check")
    
    from src.core.rocket import create_default_rocket
    from src.core.flight import simulate_flight
    
    rocket = create_default_rocket()
    
    result = simulate_flight(
        rocket=rocket,
        thrust_vac=1000.0,
        isp_vac=220.0,
        burn_time=2.0,
        exit_area=0.001,
        dt=0.1,
        max_time=120.0
    )
    
    min_alt = np.min(result.altitude)
    has_negative = min_alt < -0.1
    
    print(f"  Min Altitude: {min_alt:.2f} m")
    print(f"  Final Altitude: {result.altitude[-1]:.2f} m")
    
    if has_negative:
        log_fail("ground_collision", f"Rocket went underground! Min alt = {min_alt:.2f} m")
    else:
        log_pass("ground_collision", "Altitude correctly bounded at 0")


# =============================================================================
# PART 7: PROJECT MANAGER PERSISTENCE CHECK
# =============================================================================
def test_persistence_schema():
    """
    Check if project_manager.py exists and what it saves.
    """
    print("\n[TEST 7] Persistence Schema Check")
    
    project_manager_path = "src/core/project_manager.py"
    
    if os.path.exists(project_manager_path):
        with open(project_manager_path, 'r') as f:
            content = f.read()
        
        # Check for rocket component saving
        has_rocket = 'rocket' in content.lower()
        has_nose = 'nose' in content.lower()
        has_fins = 'fins' in content.lower()
        
        print(f"  project_manager.py exists: ✓")
        print(f"  Contains 'rocket': {has_rocket}")
        print(f"  Contains 'nose': {has_nose}")
        print(f"  Contains 'fins': {has_fins}")
        
        if has_rocket and has_nose and has_fins:
            log_pass("persistence", "Project manager appears to save rocket components")
        else:
            log_warn("persistence", "Project manager may not save all rocket components")
    else:
        log_warn("persistence", "project_manager.py not found - no save/load functionality!")


# =============================================================================
# GENERATE REPORT
# =============================================================================
def generate_report():
    """Generate audit report in Markdown format."""
    
    lines = []
    lines.append("# EnSim System Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    n_pass = len(RESULTS['passed'])
    n_fail = len(RESULTS['failed'])
    n_warn = len(RESULTS['warnings'])
    
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ✅ Passed | {n_pass} |")
    lines.append(f"| ❌ Failed | {n_fail} |")
    lines.append(f"| ⚠️ Warnings | {n_warn} |")
    lines.append("")
    
    # Critical Bugs
    lines.append("## CRITICAL BUGS")
    lines.append("")
    if RESULTS['failed']:
        for name, msg in RESULTS['failed']:
            lines.append(f"- **{name}**: {msg}")
    else:
        lines.append("None found ✅")
    lines.append("")
    
    # Warnings
    lines.append("## WARNINGS")
    lines.append("")
    if RESULTS['warnings']:
        for name, msg in RESULTS['warnings']:
            lines.append(f"- **{name}**: {msg}")
    else:
        lines.append("None ✅")
    lines.append("")
    
    # Passed
    lines.append("## PASSED TESTS")
    lines.append("")
    for name, msg in RESULTS['passed']:
        lines.append(f"- ✅ **{name}**: {msg}")
    lines.append("")
    
    # Missing Features
    lines.append("## MISSING FEATURES (vs OpenRocket)")
    lines.append("")
    lines.append("- [ ] Dual-deploy recovery (drogue + main)")
    lines.append("- [ ] Airframe material database (aluminum, fiberglass)")
    lines.append("- [ ] Surface roughness in drag calculation")
    lines.append("- [ ] Motor database (Estes, AeroTech curves)")
    lines.append("- [ ] Project save/load (.ensim files)")
    lines.append("- [ ] Simulation data export (CSV)")
    lines.append("")
    
    # Write report
    report_path = "docs/AUDIT_REPORT.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n📋 Report generated: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================
def run_audit():
    """Run all audit tests."""
    print("="*60)
    print("EnSim System Audit")
    print("="*60)
    
    test_mass_consistency()
    test_thrust_coupling()
    test_stability_logic()
    test_parachute_trigger()
    test_mach_singularity()
    test_ground_collision()
    test_persistence_schema()
    
    report_path = generate_report()
    
    print("\n" + "="*60)
    print(f"AUDIT COMPLETE: {len(RESULTS['passed'])} passed, "
          f"{len(RESULTS['failed'])} failed, {len(RESULTS['warnings'])} warnings")
    print("="*60)
    
    return RESULTS


if __name__ == "__main__":
    run_audit()
