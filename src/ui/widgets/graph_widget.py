"""2D Performance graphs using Matplotlib."""


from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from src.core.constants import GAS_CONSTANT
from src.core.propulsion import get_nozzle_profile


class PerformanceGraph(QWidget):
    """
    Widget displaying nozzle flow property graphs.

    Shows 3 subplots:
    1. Pressure ratio vs Area ratio (log scale)
    2. Temperature vs Area ratio
    3. Mach number vs Area ratio
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the matplotlib canvas."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create figure with dark theme
        self.figure = Figure(figsize=(8, 8), facecolor='#0b0c10')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create subplots
        self._create_empty_plots()

    def _create_empty_plots(self):
        """Create the subplot structure with styling."""
        self.figure.clear()

        # 3 subplots vertically stacked
        self.ax_pressure = self.figure.add_subplot(3, 1, 1)
        self.ax_temp = self.figure.add_subplot(3, 1, 2)
        self.ax_mach = self.figure.add_subplot(3, 1, 3)

        # Apply dark theme to all axes
        for ax in [self.ax_pressure, self.ax_temp, self.ax_mach]:
            ax.set_facecolor('#1f2833')
            ax.tick_params(colors='#c5c6c7', labelcolor='#c5c6c7')
            for spine in ax.spines.values():
                spine.set_color('#45a29e')
            ax.xaxis.label.set_color('#c5c6c7')
            ax.yaxis.label.set_color('#c5c6c7')
            ax.title.set_color('#66fcf1')
            ax.grid(True, color='#45a29e', linestyle='-', linewidth=0.5, alpha=0.3)

        # Labels
        self.ax_pressure.set_ylabel('P/Pc')
        self.ax_pressure.set_title('PRESSURE PROFILE', fontsize=10, fontweight='bold')
        self.ax_pressure.set_yscale('log')

        self.ax_temp.set_ylabel('T (K)')
        self.ax_temp.set_title('TEMPERATURE PROFILE', fontsize=10, fontweight='bold')

        self.ax_mach.set_ylabel('Mach')
        self.ax_mach.set_xlabel('Area Ratio (A/A*)')
        self.ax_mach.set_title('MACH NUMBER PROFILE', fontsize=10, fontweight='bold')

        # Add placeholder text
        for ax in [self.ax_pressure, self.ax_temp, self.ax_mach]:
            ax.text(0.5, 0.5, 'INIT SIMULATION',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#45a29e', alpha=0.5)

        self.figure.tight_layout()
        self.canvas.draw()

    def update_plots(
        self,
        gamma: float,
        T_chamber: float,
        P_chamber: float,
        exit_area_ratio: float,
        mean_mw: float,
    ):
        """
        Update plots with new simulation data.

        Args:
            gamma: Ratio of specific heats
            T_chamber: Chamber temperature (K)
            P_chamber: Chamber pressure (Pa)
            exit_area_ratio: Nozzle exit area ratio
            mean_mw: Mean molecular weight (g/mol)
        """
        # Calculate specific gas constant
        R_specific = GAS_CONSTANT / (mean_mw / 1000.0)

        # Get nozzle profile data
        profile = get_nozzle_profile(
            gamma=gamma,
            T_chamber=T_chamber,
            P_chamber=P_chamber,
            exit_area_ratio=exit_area_ratio,
            R_specific=R_specific,
            n_points=100
        )

        # Clear and redraw
        for ax in [self.ax_pressure, self.ax_temp, self.ax_mach]:
            ax.clear()

        # Pressure plot (log scale)
        self.ax_pressure.semilogy(
            profile['area_ratio'], profile['P_ratio'],
            color='#66fcf1', linewidth=2, label='P/Pc'
        )
        self.ax_pressure.set_ylabel('P/Pc', color='#c5c6c7')
        self.ax_pressure.set_title('PRESSURE PROFILE', fontsize=10,
                                    fontweight='bold', color='#66fcf1')
        self.ax_pressure.axhline(y=1.0, color='#ff9f1c', linestyle='--', linewidth=1)
        self.ax_pressure.fill_between(profile['area_ratio'], profile['P_ratio'],
                                      alpha=0.2, color='#66fcf1')

        # Temperature plot
        self.ax_temp.plot(
            profile['area_ratio'], profile['temperature'],
            color='#ff9f1c', linewidth=2, label='T'
        )
        self.ax_temp.set_ylabel('Temperature (K)', color='#c5c6c7')
        self.ax_temp.set_title('TEMPERATURE PROFILE', fontsize=10,
                               fontweight='bold', color='#66fcf1')
        self.ax_temp.axhline(y=T_chamber, color='#45a29e', linestyle='--',
                             linewidth=1, label=f'Tc={T_chamber:.0f}K')
        self.ax_temp.fill_between(profile['area_ratio'], profile['temperature'],
                                  alpha=0.2, color='#ff9f1c')

        # Mach number plot
        self.ax_mach.plot(
            profile['area_ratio'], profile['mach'],
            color='#66fcf1', linewidth=2, label='Mach'
        )
        self.ax_mach.set_ylabel('Mach Number', color='#c5c6c7')
        self.ax_mach.set_xlabel('Area Ratio (A/A*)', color='#c5c6c7')
        self.ax_mach.set_title('MACH NUMBER PROFILE', fontsize=10,
                               fontweight='bold', color='#66fcf1')
        self.ax_mach.axhline(y=1.0, color='#ffffff', linestyle='--',
                             linewidth=1, label='Sonic (M=1)')
        self.ax_mach.fill_between(profile['area_ratio'], profile['mach'],
                                  alpha=0.2, color='#66fcf1')

        # Apply dark theme
        for ax in [self.ax_pressure, self.ax_temp, self.ax_mach]:
            ax.set_facecolor('#1f2833')
            ax.tick_params(colors='#c5c6c7', labelcolor='#c5c6c7')
            for spine in ax.spines.values():
                spine.set_color('#45a29e')
            ax.grid(True, color='#45a29e', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.legend(loc='upper right', facecolor='#0b0c10',
                     edgecolor='#45a29e', labelcolor='#c5c6c7')

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear all plots."""
        self._create_empty_plots()

    def contextMenuEvent(self, event):
        """Show context menu on right-click."""
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)

        save_action = QAction("Save Plot as PNG...", self)
        save_action.triggered.connect(self._save_plot_png)
        menu.addAction(save_action)

        save_svg_action = QAction("Save Plot as SVG...", self)
        save_svg_action.triggered.connect(self._save_plot_svg)
        menu.addAction(save_svg_action)

        menu.exec(event.globalPos())

    def _save_plot_png(self):
        """Save the current plot as PNG."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "nozzle_profile.png",
            "PNG Images (*.png);;All Files (*)"
        )
        if path:
            self.figure.savefig(path, dpi=300, facecolor='#1e1e1e',
                               edgecolor='none', bbox_inches='tight')

    def _save_plot_svg(self):
        """Save the current plot as SVG."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "nozzle_profile.svg",
            "SVG Images (*.svg);;All Files (*)"
        )
        if path:
            self.figure.savefig(path, format='svg', facecolor='#1e1e1e',
                               edgecolor='none', bbox_inches='tight')
