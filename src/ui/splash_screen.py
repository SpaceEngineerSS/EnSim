"""Splash screen for EnSim application."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen


class EnSimSplashScreen(QSplashScreen):
    """
    Custom splash screen with loading progress.
    """

    def __init__(self):
        # Create a custom pixmap for the splash
        pixmap = QPixmap(500, 300)
        pixmap.fill(QColor("#1e1e1e"))

        # Draw content
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Gradient background
        gradient = QLinearGradient(0, 0, 500, 300)
        gradient.setColorAt(0, QColor("#1e1e1e"))
        gradient.setColorAt(1, QColor("#252525"))
        painter.fillRect(0, 0, 500, 300, gradient)

        # Draw nozzle shape
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#00a8ff"))

        # Simple nozzle polygon
        from PyQt6.QtCore import QPoint
        from PyQt6.QtGui import QPolygon
        nozzle = QPolygon([
            QPoint(200, 100),  # inlet top
            QPoint(200, 140),  # inlet bottom
            QPoint(230, 130),  # throat bottom
            QPoint(300, 160),  # exit bottom
            QPoint(300, 80),   # exit top
            QPoint(230, 110),  # throat top
        ])
        painter.drawPolygon(nozzle)

        # Draw exhaust flame
        painter.setBrush(QColor("#ff6b35"))
        exhaust = QPolygon([
            QPoint(300, 85),
            QPoint(300, 155),
            QPoint(380, 120),
        ])
        painter.drawPolygon(exhaust)

        # Title
        painter.setPen(QColor("#ffffff"))
        font = QFont("Segoe UI", 36, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(100, 220, "EnSim")

        # Subtitle
        painter.setPen(QColor("#888888"))
        font = QFont("Segoe UI", 12)
        painter.setFont(font)
        painter.drawText(100, 250, "Rocket Engine Simulation Suite")

        # Version
        painter.setPen(QColor("#00a8ff"))
        painter.drawText(100, 275, "v1.0.0")

        painter.end()

        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)

        self._message = "Initializing..."

    def showMessage(self, message: str):
        """Update loading message."""
        self._message = message
        super().showMessage(
            message,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            QColor("#00a8ff")
        )
        QApplication.processEvents()

    def show_with_progress(self, steps: list, callback):
        """
        Show splash and execute steps with progress.

        Args:
            steps: List of (message, callable) tuples
            callback: Function to call when done
        """
        self.show()

        for _i, (message, func) in enumerate(steps):
            self.showMessage(f"{message}...")
            QApplication.processEvents()
            try:
                func()
            except Exception as e:
                print(f"Splash step error: {e}")

        self.showMessage("Ready!")
        QTimer.singleShot(500, callback)
