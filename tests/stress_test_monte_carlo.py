"""
Monte Carlo Stress Test

Tests parallel execution of Monte Carlo dispersion analysis.
Reports execution time and validates multiprocessing works correctly.
"""

import sys
import time
import multiprocessing

# Add project root to path
sys.path.insert(0, '.')

from src.core.rocket import Rocket
from src.core.monte_carlo import (
    run_monte_carlo,
    DispersionConfig,
    print_dispersion_summary
)


def main():
    """Run Monte Carlo stress test."""
    print("=" * 60)
    print("MONTE CARLO STRESS TEST")
    print("=" * 60)
    print(f"CPU cores available: {multiprocessing.cpu_count()}")
    print()
    
    # Create test rocket
    rocket = Rocket()
    rocket.engine.fuel_mass = 5.0
    rocket.engine.oxidizer_mass = 15.0
    
    # Test parameters
    thrust_vac = 10000.0   # N
    isp_vac = 250.0        # s
    burn_time = 10.0       # s
    
    # Configure Monte Carlo
    config = DispersionConfig(
        num_simulations=100,
        thrust_sigma=0.02,         # 2% thrust variation
        isp_sigma=0.01,            # 1% ISP variation
        cd_sigma=0.05,             # 5% drag variation
        wind_speed_mean=3.0,       # 3 m/s mean wind
        wind_speed_sigma=2.0,      # 2 m/s wind variation
        launch_angle_sigma=0.5,    # 0.5 deg launch angle error
        seed=42                    # Reproducible results
    )
    
    print(f"Running {config.num_simulations} simulations...")
    print()
    
    # Run Monte Carlo
    start = time.time()
    
    result = run_monte_carlo(
        rocket=rocket,
        thrust_vac=thrust_vac,
        isp_vac=isp_vac,
        burn_time=burn_time,
        config=config,
        launch_angle_deg=85.0,
        dt=0.02,
        max_time=60.0,
        use_adaptive=False,
        verbose=True
    )
    
    elapsed = time.time() - start
    
    # Print summary
    print_dispersion_summary(result)
    
    print()
    print("STRESS TEST RESULTS:")
    print(f"  Total execution time: {elapsed:.2f} seconds")
    print(f"  Simulations per second: {config.num_simulations / elapsed:.1f}")
    print(f"  Time per simulation: {elapsed / config.num_simulations * 1000:.1f} ms")
    print()
    
    if result.n_successful >= 90:  # Allow 10% failure rate
        print("✓ STRESS TEST PASSED")
        return 0
    else:
        print("✗ STRESS TEST FAILED: Too many simulation failures")
        return 1


if __name__ == '__main__':
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
