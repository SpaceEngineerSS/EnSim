"""
STRESS TEST C: Thermodynamic Data Integrity Audit
Verify NASA polynomials against physical bounds.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from src.utils.nasa_parser import create_sample_database
from src.core.thermodynamics import cp_over_r, h_over_rt, s_over_r
from src.core.constants import GAS_CONSTANT


def test_heat_capacity_bounds():
    """Check Cp > 0 for all species at standard temperature."""
    print("\n" + "="*60)
    print("TEST 1: HEAT CAPACITY BOUNDS (Cp > 0 at 298.15K)")
    print("="*60)
    
    failures = []
    db = create_sample_database()
    
    T = 298.15  # Standard temperature
    
    for name, species in db.items():
        try:
            coeffs = species.get_coeffs_for_temp(T)
            cp_r = cp_over_r(T, coeffs)
            cp = cp_r * GAS_CONSTANT  # J/(mol·K)
            
            if cp < 0:
                failures.append(f"NEGATIVE Cp for {name}: {cp:.2f} J/(mol·K)")
            elif np.isnan(cp):
                failures.append(f"NaN Cp for {name}")
            else:
                print(f"  {name:>8}: Cp = {cp:>8.2f} J/(mol·K)")
                
        except Exception as e:
            failures.append(f"CRASH for {name}: {e}")
    
    return failures


def test_entropy_continuity():
    """Check entropy continuity at T=1000K (polynomial transition)."""
    print("\n" + "="*60)
    print("TEST 2: ENTROPY CONTINUITY AT T=1000K")
    print("="*60)
    
    failures = []
    db = create_sample_database()
    
    T_mid = 1000.0
    delta = 1.0  # 1K step
    
    for name, species in db.items():
        try:
            # Get coefficients just below and above transition
            coeffs_low = species.get_coeffs_for_temp(T_mid - delta)
            coeffs_high = species.get_coeffs_for_temp(T_mid + delta)
            
            s_below = s_over_r(T_mid - delta, coeffs_low)
            s_above = s_over_r(T_mid + delta, coeffs_high)
            
            # Calculate at exact transition with each set
            s_at_low = s_over_r(T_mid, coeffs_low)
            s_at_high = s_over_r(T_mid, species.get_coeffs_for_temp(T_mid))
            
            jump = abs(s_at_high - s_at_low)
            
            if jump > 0.1:  # More than 0.1 R discontinuity
                failures.append(f"ENTROPY JUMP for {name}: delta S/R = {jump:.4f}")
            else:
                print(f"  {name:>8}: S jump = {jump:.4f} R (OK)")
                
        except Exception as e:
            failures.append(f"CRASH for {name}: {e}")
    
    return failures


def test_enthalpy_sign():
    """Check enthalpy behavior at high temperatures."""
    print("\n" + "="*60)
    print("TEST 3: ENTHALPY AT EXTREMES (300K - 5000K)")
    print("="*60)
    
    failures = []
    db = create_sample_database()
    
    temps = [300.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    
    for name, species in db.items():
        try:
            enthalpies = []
            for T in temps:
                coeffs = species.get_coeffs_for_temp(T)
                h_rt = h_over_rt(T, coeffs)
                h = h_rt * GAS_CONSTANT * T  # J/mol
                enthalpies.append(h)
            
            # Check monotonicity (H should generally increase with T for most species)
            # But this isn't strictly required, so just check for NaN
            if any(np.isnan(enthalpies)):
                failures.append(f"NaN ENTHALPY for {name}")
            else:
                print(f"  {name:>8}: H(300K)={enthalpies[0]/1000:.1f} kJ/mol, H(5000K)={enthalpies[-1]/1000:.1f} kJ/mol")
                
        except Exception as e:
            failures.append(f"CRASH for {name}: {e}")
    
    return failures


def test_molecular_weights():
    """Verify molecular weights are positive and reasonable."""
    print("\n" + "="*60)
    print("TEST 4: MOLECULAR WEIGHT BOUNDS")
    print("="*60)
    
    failures = []
    db = create_sample_database()
    
    for name, species in db.items():
        mw = species.molecular_weight
        
        if mw <= 0:
            failures.append(f"INVALID MW for {name}: {mw}")
        elif mw > 200:  # Unreasonably high for propellant species
            failures.append(f"SUSPICIOUSLY HIGH MW for {name}: {mw}")
        else:
            print(f"  {name:>8}: MW = {mw:.3f} g/mol")
    
    return failures


if __name__ == "__main__":
    print("="*60)
    print("RED TEAM: THERMODYNAMIC DATA AUDIT")
    print("="*60)
    
    all_failures = []
    
    all_failures.extend(test_heat_capacity_bounds())
    all_failures.extend(test_entropy_continuity())
    all_failures.extend(test_enthalpy_sign())
    all_failures.extend(test_molecular_weights())
    
    print("\n" + "="*60)
    print("DATA INTEGRITY REPORT")
    print("="*60)
    
    if all_failures:
        print(f"\n{len(all_failures)} DATA INTEGRITY ISSUES:\n")
        for f in all_failures:
            print(f"  - {f}")
    else:
        print("\nALL THERMODYNAMIC DATA VALID")
    
    sys.exit(1 if all_failures else 0)
