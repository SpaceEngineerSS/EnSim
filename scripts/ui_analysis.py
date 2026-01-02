#!/usr/bin/env python3
"""
EnSim UI/UX Analysis Script
============================
Analyzes UI components, widgets, and features.
Generates report of UI capabilities and integration status.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_ui():
    """Analyze UI structure and capabilities."""
    print("=" * 60)
    print("EnSim UI/UX Analysis Report")
    print("=" * 60)
    
    ui_path = Path(__file__).parent.parent / "src" / "ui"
    
    # 1. Widget Analysis
    print("\n[1] WIDGET ANALYSIS")
    print("-" * 40)
    
    widgets_dir = ui_path / "widgets"
    widgets = list(widgets_dir.glob("*.py"))
    widget_info = []
    
    for w in widgets:
        if w.name.startswith("__"):
            continue
        content = w.read_text(encoding="utf-8", errors="ignore")
        lines = len(content.splitlines())
        classes = content.count("class ")
        widget_info.append((w.stem, lines, classes))
    
    print(f"\n  Found {len(widget_info)} widget modules:\n")
    for name, lines, classes in sorted(widget_info, key=lambda x: -x[1]):
        status = "[OK]" if lines > 50 else "[MINIMAL]"
        print(f"    {status} {name:<25} ({lines:>4} lines, {classes} classes)")
    
    # 2. Window Analysis
    print("\n[2] WINDOW ANALYSIS")
    print("-" * 40)
    
    windows_dir = ui_path / "windows"
    windows = list(windows_dir.glob("*.py"))
    
    for w in windows:
        if w.name.startswith("__"):
            continue
        content = w.read_text(encoding="utf-8", errors="ignore")
        lines = len(content.splitlines())
        print(f"    [OK] {w.stem:<25} ({lines:>4} lines)")
    
    # 3. Feature Detection
    print("\n[3] UI FEATURE DETECTION")
    print("-" * 40)
    
    main_window = (windows_dir / "main_window.py").read_text(encoding="utf-8", errors="ignore")
    
    features = {
        "KPI Dashboard": "KPICard" in main_window,
        "Tab System": "QTabWidget" in main_window,
        "3D Visualization": "NozzleView3D" in main_window or "3d" in main_window.lower(),
        "Graph/Charts": "PerformanceGraph" in main_window or "Graph" in main_window,
        "Input Panel": "InputPanel" in main_window,
        "Toolbar": "QToolBar" in main_window,
        "Menu System": "QMenu" in main_window or "_setup_menus" in main_window,
        "Status Bar": "QStatusBar" in main_window,
        "Splitter Layout": "QSplitter" in main_window,
        "Monte Carlo": "MonteCarlo" in main_window,
        "Export Functions": "export" in main_window.lower(),
        "Project Save/Load": "ProjectManager" in main_window,
        "Dark Theme": "stylesheet" in main_window.lower(),
        "Keyboard Shortcuts": "QKeySequence" in main_window,
        "Workers/Threading": "QThread" in main_window or "Worker" in main_window,
    }
    
    for feature, present in features.items():
        status = "[X]" if present else "[ ]"
        print(f"    {status} {feature}")
    
    # 4. Stylesheet Analysis
    print("\n[4] STYLING ANALYSIS")
    print("-" * 40)
    
    assets_dir = Path(__file__).parent.parent / "assets"
    qss_files = list(assets_dir.glob("*.qss")) if assets_dir.exists() else []
    
    if qss_files:
        for qss in qss_files:
            content = qss.read_text(encoding="utf-8", errors="ignore")
            lines = len(content.splitlines())
            
            style_features = {
                "Neon/Glow Effects": "glow" in content.lower() or "rgba" in content,
                "Gradient Backgrounds": "gradient" in content.lower(),
                "Custom Scrollbars": "QScrollBar" in content,
                "Button Styling": "QPushButton" in content,
                "Dark Theme Colors": "#1a1a2e" in content or "#0a0a14" in content,
                "Custom Fonts": "font-family" in content,
                "Border Radius": "border-radius" in content,
                "Hover Effects": ":hover" in content,
            }
            
            print(f"\n    Stylesheet: {qss.name} ({lines} lines)")
            for feat, present in style_features.items():
                status = "[X]" if present else "[ ]"
                print(f"      {status} {feat}")
    else:
        print("    No QSS stylesheets found in assets/")
    
    # 5. New Module Integration Status
    print("\n[5] NEW MODULE UI INTEGRATION STATUS")
    print("-" * 40)
    
    all_ui_code = ""
    for py_file in ui_path.rglob("*.py"):
        all_ui_code += py_file.read_text(encoding="utf-8", errors="ignore")
    
    integrations = {
        "Multi-Stage Vehicle": ("staging" in all_ui_code.lower(), "staging.py"),
        "Optimization Tools": ("optimi" in all_ui_code.lower(), "optimization.py"),
        "Cooling Analysis": ("cooling" in all_ui_code.lower(), "cooling.py"),
        "Propellant Presets (17+)": ("preset" in all_ui_code.lower(), "propellant_presets.py"),
        "Unit Conversion": ("UnitConverter" in all_ui_code or "convert" in all_ui_code, "units.py"),
    }
    
    for feature, (integrated, module) in integrations.items():
        status = "[INTEGRATED]" if integrated else "[NOT YET]"
        print(f"    {status} {feature} ({module})")
    
    # 6. Input Panel Analysis
    print("\n[6] INPUT PANEL CONTROLS")
    print("-" * 40)
    
    input_panel_path = widgets_dir / "input_panel.py"
    if input_panel_path.exists():
        content = input_panel_path.read_text(encoding="utf-8", errors="ignore")
        
        controls = {
            "Propellant Selection": "propellant" in content.lower() or "fuel" in content.lower(),
            "Chamber Pressure": "chamber" in content.lower() and "pressure" in content.lower(),
            "O/F Ratio": "of_ratio" in content.lower() or "mixture" in content.lower(),
            "Expansion Ratio": "expansion" in content.lower() or "area_ratio" in content.lower(),
            "Mass Flow Rate": "mass_flow" in content.lower() or "mdot" in content.lower(),
            "Efficiency Inputs": "eta" in content.lower() or "efficiency" in content.lower(),
            "Ambient Pressure": "ambient" in content.lower(),
            "Spin Box/Sliders": "QSpinBox" in content or "QSlider" in content or "QDoubleSpinBox" in content,
            "Combo Box (Dropdowns)": "QComboBox" in content,
            "Run Button": "run" in content.lower() and "button" in content.lower(),
        }
        
        for control, present in controls.items():
            status = "[X]" if present else "[ ]"
            print(f"    {status} {control}")
    
    # 7. Recommendations
    print("\n[7] RECOMMENDATIONS FOR UI ENHANCEMENT")
    print("-" * 40)
    
    recommendations = []
    
    if not integrations["Multi-Stage Vehicle"][0]:
        recommendations.append("Add Multi-Stage configuration tab/widget")
    if not integrations["Optimization Tools"][0]:
        recommendations.append("Add Optimization tools panel")
    if not integrations["Cooling Analysis"][0]:
        recommendations.append("Add Thermal/Cooling analysis view")
    if not integrations["Propellant Presets (17+)"][0]:
        recommendations.append("Integrate propellant preset dropdown")
    if not integrations["Unit Conversion"][0]:
        recommendations.append("Add SI/Imperial unit toggle")
    
    if recommendations:
        print("\n  Suggested UI improvements:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")
    else:
        print("\n  All new modules appear to be integrated!")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_widgets = len(widget_info)
    total_features = sum(features.values())
    total_integrations = sum(1 for v, _ in integrations.values() if v)
    
    print(f"""
  Widgets:        {total_widgets} modules
  UI Features:    {total_features}/{len(features)} detected
  Integrations:   {total_integrations}/{len(integrations)} new modules
  
  Overall Status: {'GOOD - Minor enhancements needed' if total_integrations < len(integrations) else 'EXCELLENT'}
    """)
    
    return True


if __name__ == "__main__":
    analyze_ui()

