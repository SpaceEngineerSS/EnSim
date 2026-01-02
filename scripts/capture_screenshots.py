#!/usr/bin/env python3
"""
Automatic Screenshot Capture Script for EnSim Documentation.

Captures screenshots of each tab sequentially with proper delays.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer


def capture_screenshots():
    """Capture screenshots of all tabs."""
    
    app = QApplication(sys.argv)
    
    from src.ui.windows.main_window import MainWindow
    
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    
    # Center window
    screen = app.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        window.move((geo.width() - 1400) // 2, (geo.height() - 900) // 2)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(output_dir, exist_ok=True)
    
    step = [0]
    
    def save_screenshot(name: str):
        """Save current window as screenshot."""
        app.processEvents()
        time.sleep(0.5)
        app.processEvents()
        
        pixmap = window.grab()
        path = os.path.join(output_dir, f'{name}.png')
        pixmap.save(path, 'PNG')
        
        size_kb = os.path.getsize(path) / 1024
        print(f"[OK] {name}.png ({size_kb:.1f} KB)")
    
    def run_simulation():
        """Run a quick simulation to populate data."""
        try:
            if hasattr(window, 'input_panel'):
                p = window.input_panel
                if hasattr(p, 'fuel_combo'):
                    p.fuel_combo.setCurrentIndex(0)  # H2
                if hasattr(p, 'oxidizer_combo'):
                    p.oxidizer_combo.setCurrentIndex(0)  # O2
                if hasattr(p, 'of_ratio_spin'):
                    p.of_ratio_spin.setValue(6.0)
                if hasattr(p, 'chamber_pressure_spin'):
                    p.chamber_pressure_spin.setValue(70.0)
                if hasattr(p, 'expansion_ratio_spin'):
                    p.expansion_ratio_spin.setValue(40.0)
            
            if hasattr(window, '_run_simulation'):
                window._run_simulation()
                app.processEvents()
                time.sleep(2)  # Wait for simulation
                app.processEvents()
                print("[OK] Simulation completed")
        except Exception as e:
            print(f"[WARN] Simulation error: {e}")
    
    def next_step():
        """Execute screenshot steps sequentially."""
        s = step[0]
        
        try:
            if s == 0:
                print("\n=== EnSim Screenshot Capture v2 ===\n")
                print("Step 1: Running simulation...")
                run_simulation()
                
            elif s == 1:
                print("\nStep 2: Capturing Output tab...")
                window.tabs.setCurrentIndex(0)  # Output
                app.processEvents()
                time.sleep(0.3)
                save_screenshot('screenshot_main')
                
            elif s == 2:
                print("\nStep 3: Capturing Results tab...")
                window.tabs.setCurrentIndex(1)  # Results
                app.processEvents()
                time.sleep(0.3)
                save_screenshot('screenshot_results')
                
            elif s == 3:
                print("\nStep 4: Capturing Results/3D...")
                # Find 3D sub-tab
                results_tab = window.tabs.widget(1)
                if results_tab:
                    for child in results_tab.findChildren(type(window.tabs)):
                        if child.objectName() == "subTabs":
                            child.setCurrentIndex(1)  # 3D View
                            break
                app.processEvents()
                time.sleep(0.5)
                save_screenshot('screenshot_3d')
                
            elif s == 4:
                print("\nStep 5: Capturing Results/Graphs...")
                results_tab = window.tabs.widget(1)
                if results_tab:
                    for child in results_tab.findChildren(type(window.tabs)):
                        if child.objectName() == "subTabs":
                            child.setCurrentIndex(0)  # Graphs
                            break
                app.processEvents()
                time.sleep(0.3)
                save_screenshot('screenshot_graphs')
                
            elif s == 5:
                print("\nStep 6: Capturing Engine tab...")
                window.tabs.setCurrentIndex(2)  # Engine
                app.processEvents()
                time.sleep(0.3)
                save_screenshot('screenshot_engine')
                
            elif s == 6:
                print("\nStep 7: Capturing Vehicle tab...")
                window.tabs.setCurrentIndex(3)  # Vehicle
                app.processEvents()
                time.sleep(0.3)
                save_screenshot('screenshot_vehicle')
                
            elif s == 7:
                print("\nStep 8: Capturing Advanced tab...")
                window.tabs.setCurrentIndex(4)  # Advanced
                app.processEvents()
                time.sleep(0.3)
                save_screenshot('screenshot_advanced')
                
            elif s == 8:
                print("\n=== Complete! ===")
                print(f"Screenshots saved to: {output_dir}")
                window.close()
                app.quit()
                return
                
        except Exception as e:
            print(f"[ERROR] Step {s}: {e}")
            import traceback
            traceback.print_exc()
        
        step[0] += 1
        QTimer.singleShot(800, next_step)
    
    # Start after window is ready
    QTimer.singleShot(1500, next_step)
    
    sys.exit(app.exec())


if __name__ == '__main__':
    capture_screenshots()
