"""
STRESS TEST A: Physics Singularity Test
Push the solver to absolute limits to find crashes/hangs/NaN.
"""

import sys
import time
import numpy as np
sys.path.insert(0, '.')

from src.core.chemistry import CombustionProblem
from src.core.propulsion import NozzleConditions, calculate_performance
from src.utils.nasa_parser import create_sample_database

TIMEOUT_SECONDS = 5.0

def test_pressure_extremes():
    """Test pressure from 0.01 bar to 1000 bar."""
    print("\n" + "="*60)
    print("TEST 1: PRESSURE EXTREMES (0.01 - 1000 bar)")
    print("="*60)
    
    failures = []
    db = create_sample_database()
    
    pressures = [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 500.0, 1000.0]
    
    for P_bar in pressures:
        P_Pa = P_bar * 1e5
        try:
            start = time.time()
            
            problem = CombustionProblem(db)
            problem.add_fuel("H2", moles=2.0)
            problem.add_oxidizer("O2", moles=1.0)
            
            result = problem.solve(pressure=P_Pa, initial_temp_guess=3000.0, max_iterations=50)
            
            elapsed = time.time() - start
            
            if elapsed > TIMEOUT_SECONDS:
                failures.append(f"TIMEOUT at Pc={P_bar} bar ({elapsed:.1f}s)")
            elif not result.converged:
                failures.append(f"NO CONVERGENCE at Pc={P_bar} bar")
            elif np.isnan(result.temperature):
                failures.append(f"NaN TEMPERATURE at Pc={P_bar} bar")
            elif result.temperature < 0:
                failures.append(f"NEGATIVE TEMP at Pc={P_bar} bar: T={result.temperature}")
            else:
                print(f"  Pc={P_bar:>7.2f} bar: T={result.temperature:.0f}K, converged={result.converged} ({elapsed:.2f}s)")
                
        except Exception as e:
            failures.append(f"CRASH at Pc={P_bar} bar: {type(e).__name__}: {e}")
    
    return failures


def test_mixture_extremes():
    """Test O/F ratios from 0.1 (fuel rich) to 50.0 (oxidizer rich)."""
    print("\n" + "="*60)
    print("TEST 2: MIXTURE EXTREMES (O/F 0.1 - 50.0)")
    print("="*60)
    
    failures = []
    db = create_sample_database()
    
    # O/F mass ratios
    of_ratios = [0.1, 0.5, 1.0, 3.0, 8.0, 15.0, 30.0, 50.0]
    
    for of_ratio in of_ratios:
        try:
            start = time.time()
            
            # Calculate moles for given O/F
            mw_h2, mw_o2 = 2.016, 32.0
            fuel_moles = 2.0
            ox_moles = of_ratio * fuel_moles * mw_h2 / mw_o2
            
            problem = CombustionProblem(db)
            problem.add_fuel("H2", moles=fuel_moles)
            problem.add_oxidizer("O2", moles=ox_moles)
            
            result = problem.solve(pressure=68e5, initial_temp_guess=3000.0, max_iterations=50)
            
            elapsed = time.time() - start
            
            # Check for negative mole fractions
            if any(x < 0 for x in result.mole_fractions):
                failures.append(f"NEGATIVE MOLE FRACTION at O/F={of_ratio}")
            elif np.any(np.isnan(result.mole_fractions)):
                failures.append(f"NaN MOLE FRACTIONS at O/F={of_ratio}")
            elif not result.converged:
                failures.append(f"NO CONVERGENCE at O/F={of_ratio}")
            else:
                print(f"  O/F={of_ratio:>5.1f}: T={result.temperature:.0f}K, gamma={result.gamma:.3f} ({elapsed:.2f}s)")
                
        except Exception as e:
            failures.append(f"CRASH at O/F={of_ratio}: {type(e).__name__}: {e}")
    
    return failures


def test_temperature_extremes():
    """Test with extreme initial temperature guesses."""
    print("\n" + "="*60)
    print("TEST 3: INITIAL TEMP EXTREMES (10K - 10000K)")
    print("="*60)
    
    failures = []
    db = create_sample_database()
    
    temps = [10.0, 100.0, 500.0, 1000.0, 3000.0, 5000.0, 8000.0, 10000.0]
    
    for T_init in temps:
        try:
            start = time.time()
            
            problem = CombustionProblem(db)
            problem.add_fuel("H2", moles=2.0)
            problem.add_oxidizer("O2", moles=1.0)
            
            result = problem.solve(pressure=68e5, initial_temp_guess=T_init, max_iterations=100)
            
            elapsed = time.time() - start
            
            if elapsed > TIMEOUT_SECONDS:
                failures.append(f"TIMEOUT at T_init={T_init}K ({elapsed:.1f}s)")
            elif not result.converged:
                failures.append(f"NO CONVERGENCE at T_init={T_init}K")
            else:
                print(f"  T_init={T_init:>6.0f}K: T_final={result.temperature:.0f}K ({elapsed:.2f}s)")
                
        except Exception as e:
            failures.append(f"CRASH at T_init={T_init}K: {type(e).__name__}: {e}")
    
    return failures


def test_nozzle_extremes():
    """Test nozzle with extreme expansion ratios."""
    print("\n" + "="*60)
    print("TEST 4: NOZZLE EXTREMES (eps 1.01 - 500)")
    print("="*60)
    
    failures = []
    
    expansion_ratios = [1.01, 2.0, 10.0, 50.0, 100.0, 200.0, 500.0]
    
    for eps in expansion_ratios:
        try:
            nozzle = NozzleConditions(
                area_ratio=eps,
                chamber_pressure=68e5,
                ambient_pressure=0.0,
                throat_area=0.01
            )
            
            perf = calculate_performance(
                T_chamber=3500.0,
                P_chamber=68e5,
                gamma=1.2,
                mean_molecular_weight=18.0,
                nozzle=nozzle
            )
            
            if np.isnan(perf.isp) or np.isinf(perf.isp):
                failures.append(f"NaN/Inf ISP at eps={eps}")
            elif perf.isp < 0:
                failures.append(f"NEGATIVE ISP at eps={eps}: {perf.isp}")
            elif perf.exit_mach < 1.0:
                failures.append(f"SUBSONIC EXIT at eps={eps}: M={perf.exit_mach}")
            else:
                print(f"  eps={eps:>6.1f}: Isp={perf.isp:.1f}s, M_exit={perf.exit_mach:.2f}")
                
        except Exception as e:
            failures.append(f"CRASH at eps={eps}: {type(e).__name__}: {e}")
    
    return failures


if __name__ == "__main__":
    print("="*60)
    print("RED TEAM STRESS TEST: PHYSICS SINGULARITIES")
    print("="*60)
    
    all_failures = []
    
    all_failures.extend(test_pressure_extremes())
    all_failures.extend(test_mixture_extremes())
    all_failures.extend(test_temperature_extremes())
    all_failures.extend(test_nozzle_extremes())
    
    print("\n" + "="*60)
    print("FAILURE REPORT")
    print("="*60)
    
    if all_failures:
        print(f"\n{len(all_failures)} FAILURES FOUND:\n")
        for i, f in enumerate(all_failures, 1):
            print(f"  {i}. {f}")
    else:
        print("\nNO FAILURES - All edge cases handled!")
    
    sys.exit(1 if all_failures else 0)
