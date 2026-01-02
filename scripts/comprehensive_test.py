#!/usr/bin/env python3
"""
EnSim Comprehensive System Test
================================
Tests all modules, functions, and integrations.
Generates detailed report of what's working and what needs attention.
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test results storage
RESULTS = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "skipped": []
}


def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            try:
                result = func()
                if result is True or result is None:
                    RESULTS["passed"].append(f"[PASS] {name}")
                    return True
                elif result == "skip":
                    RESULTS["skipped"].append(f"[SKIP] {name}")
                    return None
                else:
                    RESULTS["warnings"].append(f"[WARN] {name}: {result}")
                    return False
            except Exception as e:
                RESULTS["failed"].append(f"[FAIL] {name}: {str(e)[:100]}")
                return False
        return wrapper
    return decorator


# =============================================================================
# PHASE 1: Core Module Tests
# =============================================================================

@test("Import core modules")
def test_core_imports():
    from src.core import (
        GAS_CONSTANT, G0, NASA_R,
        SpeciesData, SpeciesDatabase,
        CombustionProblem,
        NozzleConditions, PerformanceResult,
        calculate_performance, calculate_c_star
    )
    return True


@test("Import thermodynamics functions")
def test_thermo_imports():
    from src.core.thermodynamics import (
        cp_over_r, h_over_rt, s_over_r, g_over_rt,
        calculate_compressibility_factor_virial,
        calculate_compressibility_factor_rk,
        get_mixture_compressibility,
        CRITICAL_PROPERTIES
    )
    return True


@test("Import propulsion functions")
def test_propulsion_imports():
    from src.core.propulsion import (
        calculate_boundary_layer_displacement_thickness,
        calculate_boundary_layer_loss_factor,
        calculate_throat_reynolds,
        calculate_thrust_coefficient,
        calculate_exit_velocity
    )
    return True


@test("Constants are correct")
def test_constants():
    from src.core.constants import G0, GAS_CONSTANT, NASA_R
    assert abs(G0 - 9.80665) < 1e-6, f"G0 wrong: {G0}"
    # GAS_CONSTANT is in J/(mol·K), NASA_R is in J/(kmol·K)
    assert abs(GAS_CONSTANT - 8.314462) < 0.001, f"R wrong: {GAS_CONSTANT}"
    assert abs(NASA_R - 8314.46) < 0.1, f"NASA_R wrong: {NASA_R}"
    return True


# =============================================================================
# PHASE 2: New Modules Tests
# =============================================================================

@test("Import staging module")
def test_staging_imports():
    from src.core.staging import (
        Stage, StageEngine, MultiStageVehicle,
        StagingTrigger, StageStatus,
        create_falcon_9_like, create_saturn_v_like
    )
    return True


@test("Create Falcon 9-like vehicle")
def test_falcon9_creation():
    from src.core.staging import create_falcon_9_like
    vehicle = create_falcon_9_like()
    assert vehicle.name == "Falcon 9 (Full Thrust)"
    assert len(vehicle.stages) == 2
    assert vehicle.stages[0].name == "S1"
    assert vehicle.stages[1].name == "S2"
    return True


@test("Falcon 9 delta-v calculation")
def test_falcon9_deltav():
    from src.core.staging import create_falcon_9_like
    vehicle = create_falcon_9_like()
    vehicle.payload_mass = 15000  # 15 ton payload
    dv = vehicle.get_total_delta_v()
    # Falcon 9 should have ~9-10 km/s delta-v
    assert 8000 < dv < 12000, f"Delta-v out of range: {dv}"
    return True


@test("Create Saturn V-like vehicle")
def test_saturnv_creation():
    from src.core.staging import create_saturn_v_like
    vehicle = create_saturn_v_like()
    assert len(vehicle.stages) == 3
    return True


@test("Import optimization module")
def test_optimization_imports():
    from src.core.optimization import (
        OptimizationResult, TrajectoryConstraints,
        optimize_gravity_turn,
        optimize_nozzle_expansion_ratio,
        optimize_stage_mass_allocation,
        optimize_engine_parameters
    )
    return True


@test("Nozzle expansion ratio optimization")
def test_nozzle_optimization():
    from src.core.optimization import optimize_nozzle_expansion_ratio
    result = optimize_nozzle_expansion_ratio(
        chamber_pressure=7e6,
        ambient_pressure=101325,
        gamma=1.2
    )
    assert result.success or result.optimal_value > 0
    assert 5 < result.optimal_params['area_ratio'] < 200
    return True


@test("Stage mass allocation optimization")
def test_mass_optimization():
    from src.core.optimization import optimize_stage_mass_allocation
    result = optimize_stage_mass_allocation(
        total_propellant=100000,
        num_stages=2,
        payload_mass=5000,
        stage_isps=[310, 340]
    )
    assert result.success
    assert result.optimal_params['total_delta_v'] > 5000
    return True


@test("Import cooling module")
def test_cooling_imports():
    from src.core.cooling import (
        CoolantType, CoolingType,
        CoolingChannel, CoolingSystemDesign,
        bartz_heat_transfer_coefficient,
        dittus_boelter_coefficient,
        design_cooling_channels
    )
    return True


@test("Bartz heat transfer coefficient")
def test_bartz():
    from src.core.cooling import bartz_heat_transfer_coefficient
    h_g = bartz_heat_transfer_coefficient(
        D_throat=0.1,
        P_chamber=7e6,
        c_star=1800,
        T_chamber=3500,
        gamma=1.2,
        Pr=0.7,
        mu_ref=5e-5,
        area_ratio=10,
        local_diameter=0.3
    )
    assert h_g > 0, "Heat transfer coefficient should be positive"
    assert h_g < 1e6, "Heat transfer coefficient seems too high"
    return True


@test("Design cooling channels")
def test_cooling_design():
    from src.core.cooling import design_cooling_channels, CoolantType
    design = design_cooling_channels(
        thrust=1e6,
        chamber_pressure=7e6,
        chamber_temp=3500,
        coolant=CoolantType.RP1
    )
    assert design.channels.num_channels > 0
    assert design.coolant_mass_flow > 0
    return True


# =============================================================================
# PHASE 3: Data Modules Tests
# =============================================================================

@test("Import propellant presets")
def test_propellant_imports():
    from src.data.propellant_presets import (
        PROPELLANT_PRESETS, PropellantCategory,
        get_preset, get_all_preset_names
    )
    return True


@test("Propellant preset count")
def test_propellant_count():
    from src.data.propellant_presets import get_all_preset_names
    names = get_all_preset_names()
    assert len(names) >= 15, f"Only {len(names)} presets, expected 15+"
    return True


@test("LOX/LH2 preset data")
def test_lox_lh2():
    from src.data.propellant_presets import get_preset
    preset = get_preset("LOX_LH2")
    assert preset is not None
    assert preset.isp_vacuum > 400, "LOX/LH2 Isp should be > 400s"
    assert preset.of_ratio_optimal > 4
    return True


@test("LOX/CH4 preset data")
def test_lox_ch4():
    from src.data.propellant_presets import get_preset
    preset = get_preset("LOX_LCH4")
    assert preset is not None
    assert 350 < preset.isp_vacuum < 380
    return True


@test("Get presets by category")
def test_presets_by_category():
    from src.data.propellant_presets import get_presets_by_category, PropellantCategory
    cryo = get_presets_by_category(PropellantCategory.CRYOGENIC)
    assert len(cryo) >= 2
    storable = get_presets_by_category(PropellantCategory.STORABLE)
    assert len(storable) >= 3
    return True


# =============================================================================
# PHASE 4: Utility Modules Tests
# =============================================================================

@test("Import unit conversion")
def test_units_imports():
    from src.utils.units import (
        convert, to_si, from_si,
        UnitConverter, UnitSystem, UnitCategory
    )
    return True


@test("Pressure conversion: psi to MPa")
def test_pressure_conversion():
    from src.utils.units import convert
    result = convert(1000, "psi", "MPa")
    assert abs(result - 6.89476) < 0.01, f"Wrong: {result}"
    return True


@test("Temperature conversion: F to K")
def test_temperature_conversion():
    from src.utils.units import convert
    result = convert(6000, "F", "K")
    assert 3500 < result < 3700, f"Wrong: {result}"
    return True


@test("Force conversion: lbf to kN")
def test_force_conversion():
    from src.utils.units import convert
    result = convert(225000, "lbf", "kN")
    assert abs(result - 1000.8) < 1, f"Wrong: {result}"
    return True


@test("Unit converter class")
def test_unit_converter():
    from src.utils.units import UnitConverter, UnitSystem, UnitCategory
    conv = UnitConverter(UnitSystem.IMPERIAL)
    display = conv.display(7e6, UnitCategory.PRESSURE)
    assert "psi" in display.lower() or "1015" in display
    return True


# =============================================================================
# PHASE 5: Scientific Calculations Tests
# =============================================================================

@test("C* calculation for H2/O2")
def test_cstar_h2o2():
    from src.core.propulsion import calculate_c_star
    # H2/O2: T_c ~ 3250K, gamma ~ 1.14, MW ~ 12
    R_spec = 8314.46 / 12.0  # J/(kg·K)
    c_star = calculate_c_star(gamma=1.14, R_specific=R_spec, T_chamber=3250)
    # Should be around 2300-2400 m/s
    assert 2200 < c_star < 2500, f"C* out of range: {c_star}"
    return True


@test("Thrust coefficient calculation")
def test_cf():
    from src.core.propulsion import calculate_thrust_coefficient
    # Vacuum Cf for typical nozzle
    Cf = calculate_thrust_coefficient(
        gamma=1.2,
        pressure_ratio=0.001,  # P_exit/P_chamber
        area_ratio=50,
        ambient_ratio=0.0  # Vacuum
    )
    assert 1.7 < Cf < 2.0, f"Cf out of range: {Cf}"
    return True


@test("Boundary layer loss factor")
def test_bl_loss():
    from src.core.propulsion import calculate_boundary_layer_loss_factor
    factor = calculate_boundary_layer_loss_factor(
        Re_throat=1e6,
        area_ratio=50,
        throat_radius=0.05,
        nozzle_length=0.5
    )
    assert 0.95 < factor <= 1.0, f"BL factor out of range: {factor}"
    return True


@test("Compressibility factor (ideal gas limit)")
def test_compressibility():
    from src.core.thermodynamics import calculate_compressibility_factor_virial
    Z = calculate_compressibility_factor_virial(
        T=300, P=101325, Tc=150, Pc=5e6, omega=0
    )
    assert 0.95 < Z < 1.05, f"Z should be ~1 at low pressure: {Z}"
    return True


@test("Full performance calculation")
def test_full_performance():
    from src.core.propulsion import calculate_performance, NozzleConditions
    nozzle = NozzleConditions(
        area_ratio=50,
        chamber_pressure=7e6,
        ambient_pressure=0,
        throat_area=0.01
    )
    result = calculate_performance(
        T_chamber=3500,
        P_chamber=7e6,
        gamma=1.2,
        mean_molecular_weight=18.0,
        nozzle=nozzle
    )
    assert 300 < result.isp < 500, f"Isp out of range: {result.isp}"
    assert result.thrust > 0
    return True


# =============================================================================
# PHASE 6: Integration Tests
# =============================================================================

@test("Multi-stage with propellant preset")
def test_integration_staging_propellant():
    from src.core.staging import create_falcon_9_like
    from src.data.propellant_presets import get_preset
    
    vehicle = create_falcon_9_like()
    preset = get_preset("LOX_RP1")
    
    # Verify Falcon 9 uses RP-1 (close to preset)
    s1_isp = vehicle.stages[0].engine.isp_vac
    assert abs(s1_isp - preset.isp_vacuum) < 30
    return True


@test("Optimization with unit conversion")
def test_integration_opt_units():
    from src.core.optimization import optimize_nozzle_expansion_ratio
    from src.utils.units import convert
    
    # 1000 psi in Pa
    Pc_pa = convert(1000, "psi", "Pa")
    result = optimize_nozzle_expansion_ratio(
        chamber_pressure=Pc_pa,
        ambient_pressure=101325
    )
    assert result.optimal_params['area_ratio'] > 1
    return True


# =============================================================================
# PHASE 7: File/Data Tests
# =============================================================================

@test("NASA thermo data file exists")
def test_nasa_data_exists():
    data_file = Path(__file__).parent.parent / "data" / "nasa_thermo.dat"
    assert data_file.exists(), f"File not found: {data_file}"
    return True


@test("NASA thermo data has content")
def test_nasa_data_content():
    data_file = Path(__file__).parent.parent / "data" / "nasa_thermo.dat"
    content = data_file.read_text()
    assert "H2" in content, "H2 species missing"
    assert "O2" in content, "O2 species missing"
    assert len(content) > 1000, "Data file seems too small"
    return True


@test("Example files exist")
def test_example_files():
    examples_dir = Path(__file__).parent.parent / "examples"
    if not examples_dir.exists():
        return "Examples directory not found"
    ensim_files = list(examples_dir.glob("*.ensim"))
    assert len(ensim_files) >= 1, "No .ensim example files"
    return True


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all test functions and generate report."""
    print("=" * 60)
    print("EnSim Comprehensive System Test")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Collect all test functions
    tests = [
        # Phase 1: Core
        test_core_imports,
        test_thermo_imports,
        test_propulsion_imports,
        test_constants,
        # Phase 2: New Modules
        test_staging_imports,
        test_falcon9_creation,
        test_falcon9_deltav,
        test_saturnv_creation,
        test_optimization_imports,
        test_nozzle_optimization,
        test_mass_optimization,
        test_cooling_imports,
        test_bartz,
        test_cooling_design,
        # Phase 3: Data
        test_propellant_imports,
        test_propellant_count,
        test_lox_lh2,
        test_lox_ch4,
        test_presets_by_category,
        # Phase 4: Utils
        test_units_imports,
        test_pressure_conversion,
        test_temperature_conversion,
        test_force_conversion,
        test_unit_converter,
        # Phase 5: Scientific
        test_cstar_h2o2,
        test_cf,
        test_bl_loss,
        test_compressibility,
        test_full_performance,
        # Phase 6: Integration
        test_integration_staging_propellant,
        test_integration_opt_units,
        # Phase 7: Files
        test_nasa_data_exists,
        test_nasa_data_content,
        test_example_files,
    ]
    
    # Run tests
    for test_func in tests:
        test_func()
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nPASSED: {len(RESULTS['passed'])}")
    for msg in RESULTS['passed']:
        print(f"   {msg}")
    
    if RESULTS['warnings']:
        print(f"\nWARNINGS: {len(RESULTS['warnings'])}")
        for msg in RESULTS['warnings']:
            print(f"   {msg}")
    
    if RESULTS['failed']:
        print(f"\nFAILED: {len(RESULTS['failed'])}")
        for msg in RESULTS['failed']:
            print(f"   {msg}")
    
    if RESULTS['skipped']:
        print(f"\nSKIPPED: {len(RESULTS['skipped'])}")
        for msg in RESULTS['skipped']:
            print(f"   {msg}")
    
    # Summary
    total = len(RESULTS['passed']) + len(RESULTS['failed']) + len(RESULTS['warnings'])
    success_rate = len(RESULTS['passed']) / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {total} tests | PASS RATE: {success_rate:.1f}%")
    print("=" * 60)
    
    return len(RESULTS['failed']) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

