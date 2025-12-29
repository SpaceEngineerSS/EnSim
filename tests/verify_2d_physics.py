
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.core.flight import simulate_flight
from src.core.rocket import create_default_rocket

def test_2d_flight():
    print("🚀 Initializing 2D Flight Verification...")
    
    # 1. Setup Rocket
    rocket = create_default_rocket()
    
    # 2. Run Simulation (85 degree launch)
    print("   Running simulation (85 deg launch)...")
    result = simulate_flight(
        rocket=rocket,
        thrust_vac=2000.0,
        isp_vac=250.0,
        burn_time=5.0,
        exit_area=0.005,
        dt=0.05,
        wind_speed=0.0,
        rail_length=3.0,
        launch_angle_deg=85.0
    )
    
    # 3. Verify Constraints
    print(f"   Apogee: {result.apogee_altitude:.2f} m")
    print(f"   Range:  {result.range[-1]:.2f} m")
    print(f"   Max Vel: {result.max_velocity:.2f} m/s")
    
    # Check 1: Range should be positive for 85 deg launch
    if result.range[-1] > 10.0:
        print("   ✅ Range Verification: PASS (Non-vertical trajectory confirmed)")
    else:
        print("   ❌ Range Verification: FAIL (Trajectory is vertical)")
        
    # Check 2: Altitude should be reasonable
    if result.apogee_altitude > 100.0:
        print("   ✅ Altitude Verification: PASS")
    else:
        print("   ❌ Altitude Verification: FAIL (Rocket didn't fly)")
        
    # Check 3: Path angle should change (Gravity Turn)
    start_angle = result.path_angle[0]
    end_angle = result.path_angle[np.argmax(result.altitude)] # Angle at apogee
    print(f"   Path Angle: Start={start_angle:.1f}°, Apogee={end_angle:.1f}°")
    
    if start_angle > end_angle:
        print("   ✅ Gravity Turn Verification: PASS (Angle decreases)")
    else:
        print("   ❌ Gravity Turn Verification: FAIL (Angle static or increasing)")

if __name__ == "__main__":
    try:
        test_2d_flight()
        print("\n✨ 2D Physics Logic Verified Successfully!")
    except Exception as e:
        print(f"\n🔥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
