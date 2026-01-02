"""
Design Comparison Window.

Provides:
- Snapshot system to capture design states
- Side-by-side comparison table
- Overlay thrust curve plots

Phase 7: Professional UX feature.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DesignSnapshot:
    """Snapshot of a design configuration and results."""

    # Metadata
    name: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Inputs
    fuel: str = ""
    oxidizer: str = ""
    of_ratio: float = 0.0
    chamber_pressure_bar: float = 0.0
    expansion_ratio: float = 0.0
    throat_area_cm2: float = 0.0

    # Results
    isp_vacuum: float = 0.0
    isp_sea_level: float = 0.0
    thrust_kn: float = 0.0
    c_star: float = 0.0
    exit_velocity: float = 0.0
    exit_mach: float = 0.0
    gamma: float = 0.0
    temperature: float = 0.0
    mean_mw: float = 0.0

    # Thrust curve (if available)
    thrust_curve_time: np.ndarray | None = None
    thrust_curve_thrust: np.ndarray | None = None

    def format_timestamp(self) -> str:
        """Format timestamp for display."""
        return self.timestamp.strftime("%H:%M:%S")


class SnapshotManager:
    """
    Manages design snapshots.

    Singleton pattern for global access.
    """

    _instance: Optional['SnapshotManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._snapshots = []
            cls._instance._max_snapshots = 10
        return cls._instance

    @property
    def snapshots(self) -> list[DesignSnapshot]:
        """Get all snapshots."""
        return self._snapshots

    def add(self, snapshot: DesignSnapshot):
        """Add a new snapshot."""
        self._snapshots.append(snapshot)

        # Limit number of snapshots
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def remove(self, index: int):
        """Remove snapshot by index."""
        if 0 <= index < len(self._snapshots):
            self._snapshots.pop(index)

    def clear(self):
        """Clear all snapshots."""
        self._snapshots.clear()

    def get(self, index: int) -> DesignSnapshot | None:
        """Get snapshot by index."""
        if 0 <= index < len(self._snapshots):
            return self._snapshots[index]
        return None

    def count(self) -> int:
        """Get number of snapshots."""
        return len(self._snapshots)


def get_snapshot_manager() -> SnapshotManager:
    """Get the global SnapshotManager instance."""
    return SnapshotManager()


# =============================================================================
# Comparison Window
# =============================================================================

class CompareWindow(QDialog):
    """
    Design comparison dialog.

    Shows side-by-side comparison of selected snapshots.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Design Comparison")
        self.setMinimumSize(900, 600)

        self._manager = get_snapshot_manager()
        self._selected_indices = []

        self._setup_ui()
        self._load_snapshots()

    def _setup_ui(self):
        """Build the comparison UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("📊 Design Comparison")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; color: #00a8ff;")
        layout.addWidget(header)

        # Selection row
        selection_frame = QFrame()
        selection_layout = QHBoxLayout(selection_frame)

        selection_layout.addWidget(QLabel("Compare:"))

        self.snap1_combo = QComboBox()
        self.snap1_combo.setMinimumWidth(200)
        self.snap1_combo.currentIndexChanged.connect(self._update_comparison)
        selection_layout.addWidget(self.snap1_combo)

        selection_layout.addWidget(QLabel("vs"))

        self.snap2_combo = QComboBox()
        self.snap2_combo.setMinimumWidth(200)
        self.snap2_combo.currentIndexChanged.connect(self._update_comparison)
        selection_layout.addWidget(self.snap2_combo)

        selection_layout.addStretch()
        layout.addWidget(selection_frame)

        # Main content: Table + Plot
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Parameter", "Design A", "Design B"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)

        # Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 4), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        splitter.addWidget(plot_widget)
        splitter.setSizes([400, 500])

        layout.addWidget(splitter, stretch=1)

        # Buttons
        button_layout = QHBoxLayout()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _load_snapshots(self):
        """Populate snapshot combos."""
        self.snap1_combo.clear()
        self.snap2_combo.clear()

        for i, snap in enumerate(self._manager.snapshots):
            label = f"{snap.name} ({snap.format_timestamp()})"
            self.snap1_combo.addItem(label, i)
            self.snap2_combo.addItem(label, i)

        # Select last two if available
        n = self._manager.count()
        if n >= 2:
            self.snap1_combo.setCurrentIndex(n - 2)
            self.snap2_combo.setCurrentIndex(n - 1)
        elif n == 1:
            self.snap1_combo.setCurrentIndex(0)

    def _update_comparison(self):
        """Update table and plot with selected snapshots."""
        idx1 = self.snap1_combo.currentData()
        idx2 = self.snap2_combo.currentData()

        snap1 = self._manager.get(idx1) if idx1 is not None else None
        snap2 = self._manager.get(idx2) if idx2 is not None else None

        self._update_table(snap1, snap2)
        self._update_plot(snap1, snap2)

    def _update_table(self, snap1: DesignSnapshot | None,
                      snap2: DesignSnapshot | None):
        """Update comparison table."""
        parameters = [
            ("Fuel", "fuel", ""),
            ("Oxidizer", "oxidizer", ""),
            ("O/F Ratio", "of_ratio", ""),
            ("Chamber Pressure", "chamber_pressure_bar", "bar"),
            ("Expansion Ratio", "expansion_ratio", ""),
            ("Throat Area", "throat_area_cm2", "cm²"),
            ("Vacuum Isp", "isp_vacuum", "s"),
            ("Sea Level Isp", "isp_sea_level", "s"),
            ("Thrust", "thrust_kn", "kN"),
            ("C*", "c_star", "m/s"),
            ("Exit Velocity", "exit_velocity", "m/s"),
            ("Exit Mach", "exit_mach", ""),
            ("Temperature", "temperature", "K"),
            ("Gamma", "gamma", ""),
        ]

        self.table.setRowCount(len(parameters))

        for row, (name, attr, unit) in enumerate(parameters):
            # Parameter name
            self.table.setItem(row, 0, QTableWidgetItem(name))

            # Design A value
            if snap1:
                val1 = getattr(snap1, attr, "")
                text1 = f"{val1:.2f} {unit}".strip() if isinstance(val1, float) else str(val1)
            else:
                text1 = "-"
            self.table.setItem(row, 1, QTableWidgetItem(text1))

            # Design B value
            if snap2:
                val2 = getattr(snap2, attr, "")
                text2 = f"{val2:.2f} {unit}".strip() if isinstance(val2, float) else str(val2)
            else:
                text2 = "-"
            self.table.setItem(row, 2, QTableWidgetItem(text2))

            # Highlight differences
            if snap1 and snap2 and text1 != text2:
                for col in [1, 2]:
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(QColor(60, 60, 80))

    def _update_plot(self, snap1: DesignSnapshot | None,
                     snap2: DesignSnapshot | None):
        """Update comparison bar chart."""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.set_facecolor('#252525')

        if not snap1 and not snap2:
            ax.text(0.5, 0.5, "Select designs to compare",
                   ha='center', va='center', color='#666')
            self.canvas.draw()
            return

        # Metrics to compare
        metrics = ['Isp (s)', 'Thrust (kN)', 'C* (m/s)', 'T (K)']

        vals1 = []
        vals2 = []

        if snap1:
            vals1 = [snap1.isp_vacuum, snap1.thrust_kn,
                    snap1.c_star / 100, snap1.temperature / 100]
        if snap2:
            vals2 = [snap2.isp_vacuum, snap2.thrust_kn,
                    snap2.c_star / 100, snap2.temperature / 100]

        x = np.arange(len(metrics))
        width = 0.35

        if vals1:
            ax.bar(x - width/2, vals1, width, label=snap1.name if snap1 else "A",
                  color='#00a8ff', alpha=0.8)
        if vals2:
            ax.bar(x + width/2, vals2, width, label=snap2.name if snap2 else "B",
                  color='#ff6b6b', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, color='#ccc')
        ax.set_ylabel('Value (normalized)', color='#ccc')
        ax.set_title('Performance Comparison', color='white', fontweight='bold')
        ax.tick_params(colors='#888')
        ax.legend(facecolor='#2d2d2d', edgecolor='#444', labelcolor='#ccc')

        for spine in ax.spines.values():
            spine.set_color('#444')

        self.figure.tight_layout()
        self.canvas.draw()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Comparison Window...")
    print("=" * 50)

    # Create test snapshots
    manager = get_snapshot_manager()

    snap1 = DesignSnapshot(
        name="H2/O2 High Pc",
        fuel="H2",
        oxidizer="O2",
        of_ratio=6.0,
        chamber_pressure_bar=150,
        expansion_ratio=50,
        isp_vacuum=420,
        thrust_kn=100,
        c_star=2300,
        temperature=3500
    )
    manager.add(snap1)

    snap2 = DesignSnapshot(
        name="CH4/O2 Low Pc",
        fuel="CH4",
        oxidizer="O2",
        of_ratio=3.5,
        chamber_pressure_bar=80,
        expansion_ratio=30,
        isp_vacuum=350,
        thrust_kn=75,
        c_star=1800,
        temperature=3200
    )
    manager.add(snap2)

    print(f"Created {manager.count()} snapshots")
    print("✓ Comparison module ready!")
