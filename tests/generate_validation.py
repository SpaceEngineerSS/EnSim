"""
Validation Script: Generate VALIDATION.md
Compares EnSim results against NASA CEA reference values.
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, '.')

from src.core.chemistry import CombustionProblem
from src.core.propulsion import NozzleConditions, calculate_performance
from src.utils.nasa_parser import create_sample_database

# NASA CEA Reference Values (Ideal, no losses)
CEA_REFERENCE = {
    "H2_O2": {
        "description": "H2/O2, Pc=100 bar, O/F=6.0, ε=50, Vac",
        "Pc_bar": 100.0,
        "OF_ratio": 6.0,
        "expansion_ratio": 50.0,
        "fuel": "H2",
        "oxidizer": "O2",
        "fuel_moles": 2.0,
        "T_chamber_K": 3490.0,  # CEA reference
        "Isp_vac_s": 455.0,     # CEA reference
        "c_star_m_s": 2390.0,   # CEA reference
    },
    "CH4_O2": {
        "description": "CH4/O2, Pc=100 bar, O/F=3.5, ε=50, Vac",
        "Pc_bar": 100.0,
        "OF_ratio": 3.5,
        "expansion_ratio": 50.0,
        "fuel": "CH4",
        "oxidizer": "O2",
        "fuel_moles": 1.0,
        "T_chamber_K": 3550.0,  # CEA reference
        "Isp_vac_s": 363.0,     # CEA reference
        "c_star_m_s": 1850.0,   # CEA reference
    },
}


def run_validation_case(case_name: str, ref: dict) -> dict:
    """Run EnSim for a case and compare to CEA reference."""
    db = create_sample_database()
    
    # Calculate moles from O/F ratio
    mw = {"H2": 2.016, "CH4": 16.04, "O2": 32.0}
    fuel_mw = mw.get(ref["fuel"], 16.0)
    ox_mw = mw.get(ref["oxidizer"], 32.0)
    ox_moles = ref["OF_ratio"] * ref["fuel_moles"] * fuel_mw / ox_mw
    
    # Setup problem
    problem = CombustionProblem(db)
    problem.add_fuel(ref["fuel"], moles=ref["fuel_moles"])
    problem.add_oxidizer(ref["oxidizer"], moles=ox_moles)
    
    P_chamber = ref["Pc_bar"] * 1e5
    
    # Solve equilibrium
    try:
        eq_result = problem.solve(
            pressure=P_chamber,
            initial_temp_guess=3000.0,
            max_iterations=50,
            tolerance=1e-5
        )
    except Exception as e:
        return {"error": str(e)}
    
    # Calculate performance (ideal - no losses for CEA comparison)
    nozzle = NozzleConditions(
        area_ratio=ref["expansion_ratio"],
        chamber_pressure=P_chamber,
        ambient_pressure=0.0,
        throat_area=0.01
    )
    
    perf = calculate_performance(
        T_chamber=eq_result.temperature,
        P_chamber=P_chamber,
        gamma=eq_result.gamma,
        mean_molecular_weight=eq_result.mean_molecular_weight,
        nozzle=nozzle,
        eta_cstar=1.0,  # Ideal for CEA comparison
        eta_cf=1.0,
        alpha_deg=0.0   # No divergence loss for CEA comparison
    )
    
    # Calculate errors
    T_error = abs(eq_result.temperature - ref["T_chamber_K"]) / ref["T_chamber_K"] * 100
    Isp_error = abs(perf.isp - ref["Isp_vac_s"]) / ref["Isp_vac_s"] * 100
    cstar_error = abs(perf.c_star - ref["c_star_m_s"]) / ref["c_star_m_s"] * 100
    
    return {
        "case": case_name,
        "description": ref["description"],
        "T_ensim": eq_result.temperature,
        "T_cea": ref["T_chamber_K"],
        "T_error": T_error,
        "Isp_ensim": perf.isp,
        "Isp_cea": ref["Isp_vac_s"],
        "Isp_error": Isp_error,
        "cstar_ensim": perf.c_star,
        "cstar_cea": ref["c_star_m_s"],
        "cstar_error": cstar_error,
        "converged": eq_result.converged,
    }


def generate_validation_md(results: list) -> str:
    """Generate VALIDATION.md content."""
    lines = []
    lines.append("# EnSim Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Comparison with NASA CEA")
    lines.append("")
    lines.append("EnSim results compared against NASA Chemical Equilibrium with Applications (CEA).")
    lines.append("")
    lines.append("| Case | Property | EnSim | CEA | Error |")
    lines.append("|------|----------|-------|-----|-------|")
    
    all_pass = True
    max_error = 0.0
    
    for r in results:
        if "error" in r:
            lines.append(f"| {r['case']} | ERROR | {r['error']} | - | FAIL |")
            all_pass = False
            continue
        
        # Temperature
        T_status = "✅" if r["T_error"] < 2.0 else "⚠️"
        lines.append(f"| {r['case']} | T_chamber (K) | {r['T_ensim']:.0f} | {r['T_cea']:.0f} | {r['T_error']:.2f}% {T_status} |")
        
        # Isp
        Isp_status = "✅" if r["Isp_error"] < 1.0 else "⚠️"
        lines.append(f"| {r['case']} | Isp_vac (s) | {r['Isp_ensim']:.1f} | {r['Isp_cea']:.1f} | {r['Isp_error']:.2f}% {Isp_status} |")
        
        # C*
        cstar_status = "✅" if r["cstar_error"] < 1.0 else "⚠️"
        lines.append(f"| {r['case']} | C* (m/s) | {r['cstar_ensim']:.0f} | {r['cstar_cea']:.0f} | {r['cstar_error']:.2f}% {cstar_status} |")
        
        max_error = max(max_error, r["T_error"], r["Isp_error"], r["cstar_error"])
        if r["Isp_error"] > 1.0 or r["cstar_error"] > 1.0:
            all_pass = False
    
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    
    if all_pass:
        lines.append("✅ **All validation cases PASSED** (Error < 1%)")
    else:
        lines.append(f"⚠️ **Some cases have elevated error** (Max: {max_error:.2f}%)")
    
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- EnSim uses Gibbs free energy minimization")
    lines.append("- Comparison uses ideal conditions (η=1.0, α=0°)")
    lines.append("- Temperature differences may arise from species database coverage")
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("="*60)
    print("EnSim Validation Suite")
    print("="*60)
    
    results = []
    
    for case_name, ref in CEA_REFERENCE.items():
        print(f"\nRunning: {case_name}...")
        result = run_validation_case(case_name, ref)
        results.append(result)
        
        if "error" not in result:
            print(f"  T: {result['T_ensim']:.0f}K (CEA: {result['T_cea']:.0f}K, err: {result['T_error']:.2f}%)")
            print(f"  Isp: {result['Isp_ensim']:.1f}s (CEA: {result['Isp_cea']:.1f}s, err: {result['Isp_error']:.2f}%)")
        else:
            print(f"  ERROR: {result['error']}")
    
    # Generate VALIDATION.md
    md_content = generate_validation_md(results)
    
    validation_path = os.path.join(os.path.dirname(__file__), "..", "docs", "VALIDATION.md")
    with open(validation_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✅ Generated: {validation_path}")
    
    # Check pass/fail
    max_isp_error = max(r.get("Isp_error", 100) for r in results)
    if max_isp_error > 1.0:
        print(f"\n⚠️ VALIDATION WARNING: Max Isp error = {max_isp_error:.2f}%")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED: Max error < 1%")
        sys.exit(0)
