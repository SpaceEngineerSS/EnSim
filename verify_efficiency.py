"""Verification script for efficiency factor integration."""

import sys
sys.path.insert(0, '.')

from src.core.propulsion import NozzleConditions, calculate_performance

def test_efficiency_integration():
    """
    Verify that efficiency factors are correctly applied.
    
    Test cases:
    1. η_c* = 1.0, η_Cf = 1.0 (ideal)
    2. η_c* = 0.5, η_Cf = 1.0 (50% combustion efficiency)
    
    Expected: C* in case 2 should be 50% of case 1
    """
    print("=" * 60)
    print("EFFICIENCY FACTOR VERIFICATION")
    print("=" * 60)
    
    # Common parameters
    T_chamber = 3500.0  # K
    P_chamber = 68e5    # Pa (68 bar)
    gamma = 1.2
    mean_mw = 18.0  # g/mol (approx for H2O)
    
    nozzle = NozzleConditions(
        area_ratio=50.0,
        chamber_pressure=P_chamber,
        ambient_pressure=0.0,  # Vacuum
        throat_area=0.01  # m²
    )
    
    # Case 1: Ideal (η = 1.0)
    print("\nCase 1: IDEAL (η_c* = 1.0, η_Cf = 1.0)")
    perf_ideal = calculate_performance(
        T_chamber=T_chamber,
        P_chamber=P_chamber,
        gamma=gamma,
        mean_molecular_weight=mean_mw,
        nozzle=nozzle,
        eta_cstar=1.0,
        eta_cf=1.0
    )
    print(f"  C* = {perf_ideal.c_star:.2f} m/s")
    print(f"  Cf = {perf_ideal.c_f:.4f}")
    print(f"  Isp = {perf_ideal.isp:.2f} s")
    print(f"  Thrust = {perf_ideal.thrust:.2f} N")
    
    # Case 2: 50% combustion efficiency
    print("\nCase 2: η_c* = 0.5, η_Cf = 1.0")
    perf_half_cstar = calculate_performance(
        T_chamber=T_chamber,
        P_chamber=P_chamber,
        gamma=gamma,
        mean_molecular_weight=mean_mw,
        nozzle=nozzle,
        eta_cstar=0.5,
        eta_cf=1.0
    )
    print(f"  C* = {perf_half_cstar.c_star:.2f} m/s")
    print(f"  Cf = {perf_half_cstar.c_f:.4f}")
    print(f"  Isp = {perf_half_cstar.isp:.2f} s")
    print(f"  Thrust = {perf_half_cstar.thrust:.2f} N")
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    cstar_ratio = perf_half_cstar.c_star / perf_ideal.c_star
    print(f"  C* ratio: {cstar_ratio:.4f} (expected: 0.5)")
    
    # C* should be exactly 50%
    assert abs(cstar_ratio - 0.5) < 0.001, f"C* ratio mismatch: {cstar_ratio}"
    print("  ✓ C* correctly scaled by η_c*")
    
    # Cf should be unchanged (η_Cf = 1.0)
    cf_ratio = perf_half_cstar.c_f / perf_ideal.c_f
    print(f"  Cf ratio: {cf_ratio:.4f} (expected: 1.0)")
    assert abs(cf_ratio - 1.0) < 0.001, f"Cf ratio mismatch: {cf_ratio}"
    print("  ✓ Cf correctly unchanged")
    
    # Isp should be 50% (Isp = C* × Cf / g0)
    isp_ratio = perf_half_cstar.isp / perf_ideal.isp
    print(f"  Isp ratio: {isp_ratio:.4f} (expected: 0.5)")
    assert abs(isp_ratio - 0.5) < 0.001, f"Isp ratio mismatch: {isp_ratio}"
    print("  ✓ Isp correctly scaled")
    
    print("\n" + "=" * 60)
    print("✅ ALL EFFICIENCY TESTS PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_efficiency_integration()
    sys.exit(0 if success else 1)
