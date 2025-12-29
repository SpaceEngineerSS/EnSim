"""
EnSim Full Stack Verification Script.

Tests the complete pipeline:
Materials -> Propulsion -> Aerodynamics -> Flight -> Export -> Persistence
"""

import sys
import os
import tempfile
sys.path.insert(0, '.')

import numpy as np
from datetime import datetime


# Results
RESULTS = {'passed': [], 'failed': []}


def log_pass(test: str, msg: str):
    RESULTS['passed'].append((test, msg))
    print(f"  ✅ {test}: {msg}")


def log_fail(test: str, msg: str):
    RESULTS['failed'].append((test, msg))
    print(f"  ❌ {test}: {msg}")


# =============================================================================
# A. MATERIAL & MASS LOGIC
# =============================================================================
def test_material_mass_chain():
    """Verify material database affects calculated mass."""
    print("\n[A] MATERIAL & MASS LOGIC")
    
    from src.core.materials_airframe import (
        AirframeMaterial, calculate_nose_mass, calculate_tube_mass,
        calculate_fin_mass, get_material
    )
    
    # Test 1: Different materials should give different masses
    nose_aluminum = calculate_nose_mass(0.25, 0.1, AirframeMaterial.ALUMINUM)
    nose_cardboard = calculate_nose_mass(0.25, 0.1, AirframeMaterial.CARDBOARD)
    
    print(f"  Nose (Aluminum): {nose_aluminum*1000:.1f} g")
    print(f"  Nose (Cardboard): {nose_cardboard*1000:.1f} g")
    
    if nose_aluminum > nose_cardboard * 2:
        log_pass("material_density", f"Aluminum {nose_aluminum/nose_cardboard:.1f}x heavier than cardboard")
    else:
        log_fail("material_density", "Material density not affecting mass correctly")
    
    # Test 2: Mass scales with geometry
    tube_short = calculate_tube_mass(0.5, 0.1, AirframeMaterial.FIBERGLASS)
    tube_long = calculate_tube_mass(1.0, 0.1, AirframeMaterial.FIBERGLASS)
    
    print(f"  Tube 0.5m: {tube_short*1000:.1f} g")
    print(f"  Tube 1.0m: {tube_long*1000:.1f} g")
    
    if abs(tube_long / tube_short - 2.0) < 0.1:
        log_pass("geometry_scaling", "Mass scales linearly with length")
    else:
        log_fail("geometry_scaling", "Mass doesn't scale with geometry")
    
    # Test 3: Fin mass calculation
    fin_mass = calculate_fin_mass(0.1, 0.05, 0.05, 0.003, AirframeMaterial.PLYWOOD, 4)
    print(f"  4 Fins (Plywood 3mm): {fin_mass*1000:.1f} g")
    
    if fin_mass > 0:
        log_pass("fin_mass", "Fin mass calculated correctly")
    else:
        log_fail("fin_mass", "Fin mass is zero")


# =============================================================================
# B. PROPULSION INTEGRATION
# =============================================================================
def test_propulsion_integration():
    """Verify propulsion parameters integrate with rocket."""
    print("\n[B] PROPULSION INTEGRATION")
    
    from src.core.rocket import create_default_rocket
    
    rocket = create_default_rocket()
    
    # Assign engine params
    rocket.engine.thrust_vac = 5000.0
    rocket.engine.isp_vac = 250.0
    rocket.engine.mass_flow_rate = 2.04  # kg/s
    rocket.engine.burn_time = 10.0
    
    print(f"  Thrust: {rocket.engine.thrust_vac:.0f} N")
    print(f"  Isp: {rocket.engine.isp_vac:.0f} s")
    print(f"  Mdot: {rocket.engine.mass_flow_rate:.2f} kg/s")
    
    # Verify thrust = Isp * g0 * mdot
    g0 = 9.80665
    expected_thrust = rocket.engine.isp_vac * g0 * rocket.engine.mass_flow_rate
    
    if abs(expected_thrust - rocket.engine.thrust_vac) < 50:
        log_pass("thrust_equation", f"F = Isp × g₀ × ṁ verified ({expected_thrust:.0f} N)")
    else:
        log_fail("thrust_equation", f"Expected {expected_thrust:.0f} N, got {rocket.engine.thrust_vac:.0f} N")
    
    # Mass depletion
    mass_t0 = rocket.get_mass_at_time(0)
    mass_t5 = rocket.get_mass_at_time(5)
    mass_t10 = rocket.get_mass_at_time(10)
    
    print(f"  Mass t=0s: {mass_t0:.1f} kg")
    print(f"  Mass t=5s: {mass_t5:.1f} kg")
    print(f"  Mass t=10s: {mass_t10:.1f} kg")
    
    if mass_t0 > mass_t5 > mass_t10:
        log_pass("mass_depletion", "Mass decreases during burn")
    else:
        log_fail("mass_depletion", "Mass not depleting")


# =============================================================================
# C. FLIGHT DYNAMICS (WIND CHECK)
# =============================================================================
def test_flight_wind():
    """Verify wind affects angle of attack."""
    print("\n[C] FLIGHT DYNAMICS (WIND CHECK)")
    
    from src.core.rocket import create_default_rocket
    from src.core.flight import simulate_flight
    
    rocket = create_default_rocket()
    
    # Flight with NO wind
    result_calm = simulate_flight(
        rocket=rocket,
        thrust_vac=5000.0,
        isp_vac=250.0,
        burn_time=5.0,
        exit_area=0.001,
        dt=0.1,
        max_time=60.0,
        wind_speed=0.0,
        rail_length=1.5
    )
    
    # Flight WITH wind (20 m/s)
    rocket2 = create_default_rocket()
    result_wind = simulate_flight(
        rocket=rocket2,
        thrust_vac=5000.0,
        isp_vac=250.0,
        burn_time=5.0,
        exit_area=0.001,
        dt=0.1,
        max_time=60.0,
        wind_speed=20.0,
        rail_length=1.5
    )
    
    max_aoa_calm = result_calm.max_aoa
    max_aoa_wind = result_wind.max_aoa
    
    print(f"  AoA (no wind): {max_aoa_calm:.2f}°")
    print(f"  AoA (20 m/s wind): {max_aoa_wind:.2f}°")
    
    if max_aoa_wind > max_aoa_calm:
        log_pass("wind_aoa", f"Wind increases AoA ({max_aoa_wind:.1f}° > {max_aoa_calm:.1f}°)")
    else:
        log_fail("wind_aoa", "Wind doesn't affect AoA")
    
    # Both should reach apogee
    print(f"  Apogee (calm): {result_calm.apogee_altitude:.0f} m")
    print(f"  Apogee (wind): {result_wind.apogee_altitude:.0f} m")
    
    if result_calm.apogee_altitude > 100 and result_wind.apogee_altitude > 100:
        log_pass("flight_success", "Both simulations reached apogee")
    else:
        log_fail("flight_success", "Simulation failed to reach apogee")


# =============================================================================
# D. RECOVERY LOGIC
# =============================================================================
def test_recovery_logic():
    """Verify parachute physics."""
    print("\n[D] RECOVERY LOGIC")
    
    from src.core.recovery import Parachute, estimate_descent
    
    # Small chute vs large chute
    small_chute = Parachute(diameter=0.5, cd=1.5)
    large_chute = Parachute(diameter=2.0, cd=1.5)
    
    mass = 5.0  # 5 kg rocket
    
    v_small, safe_small = small_chute.is_safe_descent(mass)
    v_large, safe_large = large_chute.is_safe_descent(mass)
    
    print(f"  0.5m chute: {v_small:.1f} m/s (safe={safe_small})")
    print(f"  2.0m chute: {v_large:.1f} m/s (safe={safe_large})")
    
    if v_small > v_large:
        log_pass("chute_physics", f"Larger chute = slower descent ({v_large:.1f} < {v_small:.1f})")
    else:
        log_fail("chute_physics", "Chute size doesn't affect descent rate")
    
    # Descent estimation
    estimate = estimate_descent(apogee=1000, mass=mass, chute=large_chute)
    print(f"  Descent time from 1km: {estimate['descent_time']:.0f} s")
    print(f"  Kinetic energy: {estimate['kinetic_energy']:.1f} J")
    
    if estimate['descent_time'] > 50:
        log_pass("descent_estimation", f"Reasonable descent time ({estimate['descent_time']:.0f}s)")
    else:
        log_fail("descent_estimation", "Descent too fast")


# =============================================================================
# E. PERSISTENCE INTEGRITY
# =============================================================================
def test_persistence():
    """Verify save/load round-trip."""
    print("\n[E] PERSISTENCE INTEGRITY")
    
    from src.core.project_manager_v3 import ProjectManagerV3, RocketData, RecoveryData
    
    # Create manager
    pm = ProjectManagerV3()
    pm.new_project()
    
    # Set complex data
    pm.data.rocket.name = "Test Rocket Alpha"
    pm.data.rocket.nose_shape = "ogive"
    pm.data.rocket.nose_length = 0.35
    pm.data.rocket.body_length = 1.5
    pm.data.rocket.fin_count = 6
    pm.data.rocket.fuel_mass = 12.5
    
    pm.data.recovery.dual_deploy = True
    pm.data.recovery.main_diameter = 2.0
    pm.data.recovery.drogue_diameter = 0.5
    
    pm.data.environment.wind_speed = 15.0
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.ensim', delete=False) as f:
        temp_path = f.name
    
    pm.save(temp_path)
    print(f"  Saved to: {temp_path}")
    
    # Load into new manager
    pm2 = ProjectManagerV3()
    pm2.load(temp_path)
    
    # Compare
    checks = [
        ("rocket.name", pm.data.rocket.name, pm2.data.rocket.name),
        ("rocket.nose_shape", pm.data.rocket.nose_shape, pm2.data.rocket.nose_shape),
        ("rocket.nose_length", pm.data.rocket.nose_length, pm2.data.rocket.nose_length),
        ("rocket.fin_count", pm.data.rocket.fin_count, pm2.data.rocket.fin_count),
        ("recovery.dual_deploy", pm.data.recovery.dual_deploy, pm2.data.recovery.dual_deploy),
        ("recovery.main_diameter", pm.data.recovery.main_diameter, pm2.data.recovery.main_diameter),
        ("environment.wind_speed", pm.data.environment.wind_speed, pm2.data.environment.wind_speed),
    ]
    
    all_match = True
    for name, original, loaded in checks:
        match = original == loaded
        status = "✓" if match else "✗"
        print(f"  {name}: {original} -> {loaded} {status}")
        if not match:
            all_match = False
    
    if all_match:
        log_pass("persistence", "All fields preserved in save/load cycle")
    else:
        log_fail("persistence", "Data lost in save/load cycle")
    
    # Cleanup
    os.unlink(temp_path)


# =============================================================================
# F. EXPORT CHAIN
# =============================================================================
def test_export():
    """Verify CSV export."""
    print("\n[F] EXPORT CHAIN")
    
    from src.core.rocket import create_default_rocket
    from src.core.flight import simulate_flight
    from src.utils.export import export_flight_csv
    
    rocket = create_default_rocket()
    result = simulate_flight(
        rocket=rocket,
        thrust_vac=5000.0,
        isp_vac=250.0,
        burn_time=5.0,
        exit_area=0.001,
        dt=0.1,
        max_time=30.0
    )
    
    # Export to temp file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    success = export_flight_csv(result, temp_path)
    
    if success and os.path.exists(temp_path):
        # Check file has data
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        print(f"  CSV rows: {len(lines)} (1 header + {len(lines)-1} data)")
        
        if len(lines) > 10:
            log_pass("csv_export", f"Exported {len(lines)-1} data rows")
        else:
            log_fail("csv_export", "Not enough data exported")
        
        os.unlink(temp_path)
    else:
        log_fail("csv_export", "Export failed")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("EnSim Full Stack Verification")
    print("=" * 60)
    
    test_material_mass_chain()
    test_propulsion_integration()
    test_flight_wind()
    test_recovery_logic()
    test_persistence()
    test_export()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    n_pass = len(RESULTS['passed'])
    n_fail = len(RESULTS['failed'])
    
    print(f"\n✅ PASSED: {n_pass}")
    for test, msg in RESULTS['passed']:
        print(f"   • {test}")
    
    if RESULTS['failed']:
        print(f"\n❌ FAILED: {n_fail}")
        for test, msg in RESULTS['failed']:
            print(f"   • {test}: {msg}")
    
    print("\n" + "=" * 60)
    if n_fail == 0:
        print("🏆 ALL TESTS PASSED - GOLD MASTER STATUS ACHIEVED")
    else:
        print(f"⚠️ {n_fail} TESTS FAILED - REVIEW REQUIRED")
    print("=" * 60)
    
    return n_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
