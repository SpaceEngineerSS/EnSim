"""
Tutorial Overlay - First-time User Onboarding.

Displays an interactive step-by-step guide for new users
to understand the EnSim workflow.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class TutorialStep:
    """A single step in the tutorial."""

    def __init__(self, title: str, description: str, highlight_widget: str = None):
        self.title = title
        self.description = description
        self.highlight_widget = highlight_widget  # Object name to highlight


TUTORIAL_STEPS = [
    TutorialStep(
        "Welcome to EnSim! 🚀",
        "EnSim is a rocket engine simulation suite based on NASA CEA methodology.\n\n"
        "This tutorial will guide you through your first simulation.",
    ),
    TutorialStep(
        "Step 1: Select Propellants",
        "Choose your fuel and oxidizer from the dropdowns.\n\n"
        "• H2/O2 - Highest performance\n"
        "• CH4/O2 - Modern Raptor/Starship\n"
        "• RP-1/O2 - Classic kerosene",
        "propellantGroup"
    ),
    TutorialStep(
        "Step 2: Set Engine Parameters",
        "Configure your engine design:\n\n"
        "• Chamber Pressure - Higher = more thrust\n"
        "• Expansion Ratio - Higher = more vacuum Isp\n"
        "• O/F Ratio - Affects flame temperature",
        "engineGroup"
    ),
    TutorialStep(
        "Step 3: Run Simulation",
        "Click the RUN SIMULATION button or press F5.\n\n"
        "The simulation solves chemical equilibrium and\n"
        "calculates nozzle performance in real-time.",
        "runButton"
    ),
    TutorialStep(
        "Step 4: View Results",
        "Results appear in multiple views:\n\n"
        "• KPI Cards - Key metrics at top\n"
        "• Graphs Tab - P, T, Mach profiles\n"
        "• 3D View - Nozzle visualization\n"
        "• Output Log - Detailed calculations",
    ),
    TutorialStep(
        "You're Ready! 🎉",
        "You now know the basics of EnSim!\n\n"
        "Keyboard shortcuts:\n"
        "• F5 - Run simulation\n"
        "• Ctrl+S - Save project\n"
        "• Ctrl+E - Export report\n"
        "• Ctrl+Tab - Switch tabs\n\n"
        "Explore the Presets menu for real engine configurations!",
    ),
]


class TutorialOverlay(QWidget):
    """
    Semi-transparent overlay that guides new users.

    Signals:
        completed: Emitted when user finishes or skips tutorial
    """

    completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._current_step = 0
        self._setup_ui()

    def _setup_ui(self):
        """Build the tutorial UI."""
        self.setStyleSheet("""
            TutorialOverlay {
                background-color: rgba(0, 0, 0, 180);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Spacer to center card
        layout.addStretch(1)

        # Center container
        container = QFrame()
        container.setObjectName("tutorialCard")
        container.setStyleSheet("""
            #tutorialCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a242e, stop:1 #141b22);
                border: 2px solid #00d4ff;
                border-radius: 16px;
                max-width: 500px;
            }
        """)
        container.setMaximumWidth(500)

        card_layout = QVBoxLayout(container)
        card_layout.setContentsMargins(30, 25, 30, 25)
        card_layout.setSpacing(15)

        # Step indicator
        self.step_label = QLabel()
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_label.setStyleSheet("""
            color: #00d4ff;
            font-size: 10pt;
            font-weight: 600;
            letter-spacing: 1px;
        """)
        card_layout.addWidget(self.step_label)

        # Title
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 18pt;
            font-weight: 700;
        """)
        self.title_label.setWordWrap(True)
        card_layout.addWidget(self.title_label)

        # Description
        self.desc_label = QLabel()
        self.desc_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.desc_label.setStyleSheet("""
            color: #8899aa;
            font-size: 11pt;
            line-height: 1.5;
        """)
        self.desc_label.setWordWrap(True)
        card_layout.addWidget(self.desc_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.skip_btn = QPushButton("Skip Tutorial")
        self.skip_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #2a3a4a;
                border-radius: 8px;
                color: #8899aa;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1a242e;
                color: #ffffff;
            }
        """)
        self.skip_btn.clicked.connect(self._skip)
        btn_layout.addWidget(self.skip_btn)

        btn_layout.addStretch(1)

        self.prev_btn = QPushButton("← Back")
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background: #1a242e;
                border: 1px solid #2a3a4a;
                border-radius: 8px;
                color: #00d4ff;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2a3a4a;
            }
        """)
        self.prev_btn.clicked.connect(self._prev)
        btn_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d4ff, stop:1 #00ff9d);
                border: none;
                border-radius: 8px;
                color: #0a0e14;
                padding: 12px 24px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #ffffff;
            }
        """)
        self.next_btn.clicked.connect(self._next)
        btn_layout.addWidget(self.next_btn)

        card_layout.addLayout(btn_layout)

        # Center the card
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(container)
        h_layout.addStretch(1)
        layout.addLayout(h_layout)

        layout.addStretch(1)

        # Show first step
        self._update_step()

    def _update_step(self):
        """Update UI for current step."""
        step = TUTORIAL_STEPS[self._current_step]

        self.step_label.setText(f"STEP {self._current_step + 1} OF {len(TUTORIAL_STEPS)}")
        self.title_label.setText(step.title)
        self.desc_label.setText(step.description)

        # Update button visibility
        self.prev_btn.setVisible(self._current_step > 0)

        if self._current_step == len(TUTORIAL_STEPS) - 1:
            self.next_btn.setText("Get Started! →")
        else:
            self.next_btn.setText("Next →")

    def _next(self):
        """Go to next step or finish."""
        if self._current_step < len(TUTORIAL_STEPS) - 1:
            self._current_step += 1
            self._update_step()
        else:
            self._finish()

    def _prev(self):
        """Go to previous step."""
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step()

    def _skip(self):
        """Skip the tutorial."""
        self._finish()

    def _finish(self):
        """Complete the tutorial."""
        self.completed.emit()
        self.hide()
        self.deleteLater()

    def paintEvent(self, event):
        """Paint semi-transparent background."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 14, 20, 200))
        super().paintEvent(event)

    def showEvent(self, event):
        """Resize to match parent when shown."""
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().showEvent(event)


def show_tutorial_if_first_run(parent, settings_key: str = "tutorial_shown") -> bool:
    """
    Show tutorial if it hasn't been shown before.

    Args:
        parent: Parent widget (MainWindow)
        settings_key: QSettings key to track if tutorial was shown

    Returns:
        True if tutorial was shown, False if skipped
    """
    from PyQt6.QtCore import QSettings

    settings = QSettings("EnSim", "EnSim")

    if settings.value(settings_key, False, type=bool):
        return False  # Already shown

    overlay = TutorialOverlay(parent)
    overlay.completed.connect(lambda: settings.setValue(settings_key, True))
    overlay.show()
    overlay.raise_()

    return True
