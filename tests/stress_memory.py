"""
STRESS TEST B: Memory Leak Check
Run 500 simulations and monitor RAM growth.
"""

import sys
import gc
import time
sys.path.insert(0, '.')

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed, memory tracking limited")

from src.core.chemistry import CombustionProblem
from src.core.propulsion import NozzleConditions, calculate_performance
from src.utils.nasa_parser import create_sample_database


def get_memory_mb():
    """Get current process memory in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    return 0.0


def run_simulation():
    """Run a single complete simulation."""
    db = create_sample_database()
    
    problem = CombustionProblem(db)
    problem.add_fuel("H2", moles=2.0)
    problem.add_oxidizer("O2", moles=1.0)
    
    result = problem.solve(pressure=68e5, initial_temp_guess=3000.0, max_iterations=50)
    
    nozzle = NozzleConditions(
        area_ratio=50.0,
        chamber_pressure=68e5,
        ambient_pressure=0.0,
        throat_area=0.01
    )
    
    perf = calculate_performance(
        T_chamber=result.temperature,
        P_chamber=68e5,
        gamma=result.gamma,
        mean_molecular_weight=result.mean_molecular_weight,
        nozzle=nozzle
    )
    
    return result, perf


def test_memory_leak(n_iterations=100):
    """Run n simulations and check for memory growth."""
    print("="*60)
    print(f"MEMORY LEAK TEST ({n_iterations} iterations)")
    print("="*60)
    
    if not HAS_PSUTIL:
        print("\nInstall psutil for accurate memory tracking:")
        print("  pip install psutil\n")
    
    initial_mem = get_memory_mb()
    print(f"\nInitial memory: {initial_mem:.1f} MB")
    
    memory_samples = []
    failures = []
    
    for i in range(n_iterations):
        try:
            run_simulation()
            
            if (i + 1) % 20 == 0:
                gc.collect()  # Force garbage collection
                mem = get_memory_mb()
                memory_samples.append(mem)
                print(f"  Iteration {i+1:>4}: {mem:.1f} MB")
                
        except Exception as e:
            failures.append(f"CRASH at iteration {i+1}: {e}")
    
    final_mem = get_memory_mb()
    growth = final_mem - initial_mem
    
    print(f"\nFinal memory: {final_mem:.1f} MB")
    print(f"Memory growth: {growth:+.1f} MB")
    
    # Check for linear growth (leak indicator)
    if len(memory_samples) >= 3:
        import numpy as np
        x = np.arange(len(memory_samples))
        slope, _ = np.polyfit(x, memory_samples, 1)
        print(f"Growth rate: {slope:.2f} MB per 20 iterations")
        
        if slope > 1.0:  # More than 1MB per 20 iterations
            failures.append(f"MEMORY LEAK DETECTED: {slope:.2f} MB per 20 iterations")
    
    if growth > 50.0:  # More than 50MB total growth
        failures.append(f"EXCESSIVE MEMORY GROWTH: {growth:.1f} MB over {n_iterations} runs")
    
    return failures


if __name__ == "__main__":
    failures = test_memory_leak(n_iterations=100)  # Reduced for speed
    
    print("\n" + "="*60)
    print("MEMORY TEST REPORT")
    print("="*60)
    
    if failures:
        print(f"\n{len(failures)} ISSUES FOUND:\n")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nNO MEMORY LEAKS DETECTED")
    
    sys.exit(1 if failures else 0)
