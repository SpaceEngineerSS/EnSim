#!/usr/bin/env python3
"""
Automatic Screenshot Capture Script for EnSim Documentation.

This script launches the EnSim application and captures screenshots
of all major UI components for documentation purposes.

Usage:
    python scripts/capture_screenshots.py

Output:
    Screenshots saved to docs/ folder:
    - screenshot_main.png      - Main window with simulation
    - screenshot_engine.png    - Engine tab (Thermal + Cooling)
    - screenshot_vehicle.png   - Vehicle tab (Stages)
    - screenshot_results.png   - Results tab (Graphs)
    - screenshot_3d.png        - 3D visualization
    - screenshot_advanced.png  - Advanced engineering tab
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QScreen


def capture_screenshots():
    """Main function to capture all screenshots."""
    
    app = QApplication(sys.argv)
    
    # Import after QApplication is created
    from src.ui.windows.main_window import MainWindow
    
    # Create main window
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    
    # Center on screen
    screen = app.primaryScreen()
    if screen:
        geometry = screen.availableGeometry()
        x = (geometry.width() - window.width()) // 2
        y = (geometry.height() - window.height()) // 2
        window.move(x, y)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(output_dir, exist_ok=True)
    
    screenshots = []
    current_step = [0]  # Use list for mutable in closure
    
    def take_screenshot(name: str, description: str):
        """Capture current window state."""
        # Process events to ensure UI is rendered
        app.processEvents()
        time.sleep(0.3)
        app.processEvents()
        
        # Grab window
        pixmap = window.grab()
        
        # Save
        filepath = os.path.join(output_dir, f'{name}.png')
        pixmap.save(filepath, 'PNG')
        print(f"[OK] Captured: {name}.png - {description}")
        screenshots.append(filepath)
    
    def run_simulation():
        """Run a sample simulation for screenshots."""
        try:
            # Set some example values if possible
            if hasattr(window, 'input_panel'):
                panel = window.input_panel
                # These might fail if widgets don't exist, that's OK
                try:
                    panel.fuel_combo.setCurrentText("H2")
                    panel.oxidizer_combo.setCurrentText("O2")
                    panel.of_ratio_spin.setValue(6.0)
                    panel.chamber_pressure_spin.setValue(70.0)
                    panel.expansion_ratio_spin.setValue(40.0)
                except:
                    pass
            
            # Trigger simulation
            if hasattr(window, '_run_simulation'):
                window._run_simulation()
                app.processEvents()
                time.sleep(1.5)  # Wait for simulation
                app.processEvents()
        except Exception as e:
            print(f"[WARN] Could not run simulation: {e}")
    
    def next_step():
        """Execute next screenshot step."""
        step = current_step[0]
        
        try:
            if step == 0:
                print("\n=== EnSim Screenshot Capture ===\n")
                print("Initializing application...")
                app.processEvents()
                time.sleep(0.5)
                
            elif step == 1:
                print("Running sample simulation...")
                run_simulation()
                
            elif step == 2:
                # Main window - Output tab
                if hasattr(window, 'main_tabs'):
                    window.main_tabs.setCurrentIndex(0)
                take_screenshot('screenshot_main', 'Main window with Output tab')
                
            elif step == 3:
                # Results tab
                if hasattr(window, 'main_tabs'):
                    window.main_tabs.setCurrentIndex(1)
                app.processEvents()
                time.sleep(0.3)
                take_screenshot('screenshot_results', 'Results tab with graphs')
                
            elif step == 4:
                # Engine tab
                if hasattr(window, 'main_tabs'):
                    window.main_tabs.setCurrentIndex(2)
                app.processEvents()
                time.sleep(0.3)
                take_screenshot('screenshot_engine', 'Engine tab (Thermal/Cooling)')
                
            elif step == 5:
                # Vehicle tab
                if hasattr(window, 'main_tabs'):
                    window.main_tabs.setCurrentIndex(3)
                app.processEvents()
                time.sleep(0.3)
                take_screenshot('screenshot_vehicle', 'Vehicle tab (Staging)')
                
            elif step == 6:
                # Advanced tab
                if hasattr(window, 'main_tabs'):
                    window.main_tabs.setCurrentIndex(4)
                app.processEvents()
                time.sleep(0.3)
                take_screenshot('screenshot_advanced', 'Advanced engineering tab')
                
            elif step == 7:
                # 3D View - try to find it
                # Usually in Results or separate
                if hasattr(window, 'main_tabs'):
                    window.main_tabs.setCurrentIndex(1)  # Results has 3D
                app.processEvents()
                time.sleep(0.3)
                # Try to switch to 3D sub-tab if exists
                if hasattr(window, 'results_tabs'):
                    for i in range(window.results_tabs.count()):
                        if '3D' in window.results_tabs.tabText(i):
                            window.results_tabs.setCurrentIndex(i)
                            break
                app.processEvents()
                time.sleep(0.3)
                take_screenshot('screenshot_3d', '3D nozzle visualization')
                
            elif step == 8:
                # Graphs detail
                if hasattr(window, 'main_tabs'):
                    window.main_tabs.setCurrentIndex(1)
                if hasattr(window, 'results_tabs'):
                    for i in range(window.results_tabs.count()):
                        text = window.results_tabs.tabText(i).lower()
                        if 'graph' in text or 'plot' in text or '2d' in text:
                            window.results_tabs.setCurrentIndex(i)
                            break
                app.processEvents()
                time.sleep(0.3)
                take_screenshot('screenshot_graphs', '2D performance graphs')
                
            elif step == 9:
                print(f"\n=== Screenshot Capture Complete ===")
                print(f"Total: {len(screenshots)} screenshots saved to docs/")
                print("\nFiles:")
                for f in screenshots:
                    print(f"  - {os.path.basename(f)}")
                print()
                
                # Close and exit
                window.close()
                app.quit()
                return
                
        except Exception as e:
            print(f"[ERROR] Step {step}: {e}")
        
        current_step[0] += 1
        QTimer.singleShot(500, next_step)
    
    # Start the capture sequence
    QTimer.singleShot(1000, next_step)
    
    # Run app
    sys.exit(app.exec())


if __name__ == '__main__':
    capture_screenshots()

