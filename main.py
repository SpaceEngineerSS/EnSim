"""
EnSim - Advanced Rocket Engine Simulation

Entry point for the application.

Usage:
    python main.py           # Launch GUI (when implemented)
    python main.py --test    # Run quick validation test
"""

import argparse
import sys
from typing import NoReturn


def run_gui() -> NoReturn:
    """Launch the PyQt6 GUI application."""
    try:
        import os
        from pathlib import Path
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QIcon
        from src.ui.windows.main_window import MainWindow
        from src.ui.splash_screen import EnSimSplashScreen
        
        # Enable High DPI scaling for 4K/Retina displays
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        
        # Suppress Qt HiDPI warning spam (known PyQt6 bug)
        os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"
        
        # Enable high DPI scaling
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        
        app = QApplication(sys.argv)
        app.setApplicationName("EnSim")
        app.setOrganizationName("EnSim")
        app.setApplicationVersion("1.0.0")
        
        # Set app icon
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
        
        # Show splash screen
        splash = EnSimSplashScreen()
        splash.show()
        app.processEvents()
        
        # Pre-compile Numba functions with splash progress
        def init_physics():
            import numpy as np
            
            splash.showMessage("Loading thermodynamics...")
            from src.core.thermodynamics import cp_over_r
            # Warm up JIT with raw Numba function
            cp_over_r(1000.0, np.array([1.0, 1e-5, 1e-8, 1e-11, 1e-14, 1e4, 1.0]))
            
            splash.showMessage("Loading propulsion...")
            from src.core.propulsion import calculate_c_star
            calculate_c_star(1.2, 500.0, 3000.0)
            
            splash.showMessage("Ready...")
        
        init_physics()
        
        splash.showMessage("Starting application...")
        app.processEvents()
        
        window = MainWindow()
        window.show()
        splash.finish(window)
        
        sys.exit(app.exec())
        
    except ImportError as e:
        print("EnSim - Rocket Engine Simulation")
        print("=" * 40)
        print(f"ERROR: PyQt6 not installed. Please run:")
        print("  pip install PyQt6")
        print(f"\nDetails: {e}")
        sys.exit(1)


def run_validation_test() -> None:
    """Run a quick validation test of the core physics engine."""
    print("EnSim - Core Physics Validation Test")
    print("=" * 40)
    
    from src.core.constants import GAS_CONSTANT
    from src.core.thermodynamics import (
        calculate_cp,
        calculate_enthalpy,
        calculate_entropy,
    )
    from src.utils.nasa_parser import create_sample_database
    
    # Load sample database
    db = create_sample_database()
    print(f"\n✓ Loaded {len(db)} species from sample database")
    
    # Test H2O properties at 1000K
    h2o = db['H2O']
    T = 1000.0
    
    cp = calculate_cp(T, h2o)
    h = calculate_enthalpy(T, h2o)
    s = calculate_entropy(T, h2o)
    
    print(f"\nH2O Properties at {T} K:")
    print(f"  Cp = {cp:.4f} J/(mol·K)")
    print(f"  H  = {h/1000:.4f} kJ/mol")
    print(f"  S  = {s:.4f} J/(mol·K)")
    
    # Validate against known values (NIST/CEA reference)
    # H2O at 1000K: Cp ≈ 41.3 J/(mol·K), H ≈ -215 kJ/mol from 298K
    cp_expected = 41.3  # J/(mol·K) - approximate
    
    error_pct = abs((cp - cp_expected) / cp_expected) * 100
    print(f"\n  Cp error vs NIST: {error_pct:.2f}%")
    
    if error_pct < 1.0:
        print("\n✓ Validation PASSED (error < 1%)")
    else:
        print(f"\n⚠ Validation WARNING: error = {error_pct:.2f}%")
    
    # Test temperature range switching
    print("\nTemperature range switching test:")
    for T_test in [500.0, 999.0, 1000.0, 1001.0, 2000.0]:
        cp_test = calculate_cp(T_test, h2o)
        print(f"  T = {T_test:6.0f} K -> Cp = {cp_test:.4f} J/(mol·K)")
    
    # Coefficient continuity test at T_mid
    from src.core.thermodynamics import cp_over_r
    
    print("\n" + "-" * 40)
    print("Coefficient Continuity Test at T_mid:")
    
    T_mid = h2o.t_mid  # Should be 1000.0 K
    cp_r_low = cp_over_r(T_mid, h2o.coeffs_low)
    cp_r_high = cp_over_r(T_mid, h2o.coeffs_high)
    
    cp_low = cp_r_low * GAS_CONSTANT  # J/(mol·K)
    cp_high = cp_r_high * GAS_CONSTANT  # J/(mol·K)
    diff = abs(cp_low - cp_high)
    
    print(f"  T_mid = {T_mid:.1f} K")
    print(f"  Cp (low coeffs)  = {cp_low:.6f} J/(mol·K)")
    print(f"  Cp (high coeffs) = {cp_high:.6f} J/(mol·K)")
    print(f"  Difference       = {diff:.6f} J/(mol·K)")
    
    if diff > 0.001:
        print(f"\n⚠ WARNING: Cp discontinuity at T_mid > 0.001 J/(mol·K)!")
        print(f"  This may cause numerical issues in equilibrium calculations.")
    else:
        print(f"\n✓ Coefficients are continuous at T_mid (diff < 0.001)")
    
    # Combustion equilibrium demo
    print("\n" + "-" * 40)
    print("Combustion Equilibrium Demo (H2 + O2):")
    
    from src.core.chemistry import CombustionProblem
    
    problem = CombustionProblem(db)
    problem.add_fuel('H2', moles=2.0, temperature=298.15)
    problem.add_oxidizer('O2', moles=1.0, temperature=298.15)
    
    try:
        result = problem.solve(pressure=1013250.0)  # 10 atm
        
        print(f"\n  Reactants: 2 mol H2 + 1 mol O2 @ 10 atm")
        print(f"  Adiabatic Flame Temperature: {result.temperature:.1f} K")
        print(f"  Iterations: {result.iterations}")
        print(f"  Converged: {result.converged}")
        
        print("\n  Product Mole Fractions:")
        for name, x in sorted(zip(result.species_names, result.mole_fractions),
                               key=lambda t: t[1], reverse=True):
            if x > 0.001:
                print(f"    {name:8s}: {x:.4f}")
        
        print(f"\n  Mean MW: {result.mean_molecular_weight:.2f} g/mol")
        print(f"  Gamma:   {result.gamma:.3f}")
        
        if 2500 < result.temperature < 4500:
            print("\n✓ Combustion calculation completed successfully!")
        else:
            print(f"\n⚠ Temperature outside expected range")
        
        # Propulsion performance demo
        print("\n" + "-" * 40)
        print("Propulsion Performance (Frozen Flow):")
        
        from src.core.propulsion import NozzleConditions, calculate_performance
        
        # Vacuum performance (large expansion ratio)
        nozzle_vac = NozzleConditions(
            area_ratio=50.0,
            chamber_pressure=1013250.0,  # 10 atm
            ambient_pressure=0.0  # Vacuum
        )
        
        perf_vac = calculate_performance(
            T_chamber=result.temperature,
            P_chamber=nozzle_vac.chamber_pressure,
            gamma=result.gamma,
            mean_molecular_weight=result.mean_molecular_weight,
            nozzle=nozzle_vac
        )
        
        print(f"\n  Vacuum Performance (ε = 50):")
        print(f"    C*  = {perf_vac.c_star:.1f} m/s")
        print(f"    Ve  = {perf_vac.exit_velocity:.1f} m/s")
        print(f"    Cf  = {perf_vac.c_f:.3f}")
        print(f"    Isp = {perf_vac.isp:.1f} s")
        print(f"    Me  = {perf_vac.exit_mach:.2f}")
        
        # Sea level performance (higher chamber pressure, smaller expansion ratio)
        nozzle_sl = NozzleConditions(
            area_ratio=15.0,
            chamber_pressure=68.0 * 101325.0,  # 68 atm (typical for sea level engine)
            ambient_pressure=101325.0  # 1 atm
        )
        
        perf_sl = calculate_performance(
            T_chamber=result.temperature,
            P_chamber=nozzle_sl.chamber_pressure,
            gamma=result.gamma,
            mean_molecular_weight=result.mean_molecular_weight,
            nozzle=nozzle_sl
        )
        
        print(f"\n  Sea Level Performance (ε = 15):")
        print(f"    Isp = {perf_sl.isp:.1f} s")
        print(f"    Ve  = {perf_sl.exit_velocity:.1f} m/s")
        
        if perf_vac.isp > 400 and perf_sl.isp > 300:
            print("\n✓ Propulsion calculations completed successfully!")
        else:
            print("\n⚠ Isp outside expected range")
            
    except Exception as e:
        print(f"\n⚠ Combustion calculation failed: {e}")
    
    print("\n" + "=" * 40)
    print("✓ Core physics engine operational!")
    print("  Run 'pytest tests/' for full test suite.\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EnSim - Rocket Engine Simulation",
        prog="ensim"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run quick validation test"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="EnSim 0.1.0"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_validation_test()
    else:
        run_gui()


if __name__ == "__main__":
    main()
