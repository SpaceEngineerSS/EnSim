"""
Timeline Scrubber Widget for Flight Replay.

Provides a professional timeline control for replaying
flight recordings with play/pause, seek, and speed control.
"""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class TimelineScrubber(QWidget):
    """
    Timeline scrubber for flight replay control.

    Features:
    - Play/Pause toggle button
    - Stop button (rewind to start)
    - Seek slider with time display
    - Playback speed selector (0.25x - 4x)
    - Current time / Total duration display

    Signals:
        position_changed(float): Emitted when user seeks (0.0-1.0)
        playback_toggled(bool): Emitted on play/pause (True=playing)
        speed_changed(float): Emitted when speed changes
        stopped(): Emitted when stop is pressed
    """

    position_changed = pyqtSignal(float)  # 0.0 - 1.0
    playback_toggled = pyqtSignal(bool)   # True = playing
    speed_changed = pyqtSignal(float)     # Playback speed multiplier
    stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self._is_playing = False
        self._duration = 0.0  # seconds
        self._current_time = 0.0  # seconds
        self._speed = 1.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Build the timeline UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Play/Pause button
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(36, 36)
        self.play_btn.setObjectName("playButton")
        self.play_btn.setToolTip("Play/Pause (Space)")
        layout.addWidget(self.play_btn)

        # Stop button
        self.stop_btn = QPushButton("■")
        self.stop_btn.setFixedSize(36, 36)
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setToolTip("Stop and rewind")
        layout.addWidget(self.stop_btn)

        # Skip buttons
        self.skip_back_btn = QPushButton("⏮")
        self.skip_back_btn.setFixedSize(30, 30)
        self.skip_back_btn.setToolTip("Skip to start")
        layout.addWidget(self.skip_back_btn)

        self.rewind_btn = QPushButton("⏪")
        self.rewind_btn.setFixedSize(30, 30)
        self.rewind_btn.setToolTip("Rewind 10s")
        layout.addWidget(self.rewind_btn)

        # Current time label
        self.time_label = QLabel("00:00.0")
        self.time_label.setMinimumWidth(60)
        self.time_label.setStyleSheet("font-family: monospace; font-size: 12pt;")
        layout.addWidget(self.time_label)

        # Timeline slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(0)
        self.slider.setTracking(True)
        self.slider.setToolTip("Seek position")
        layout.addWidget(self.slider, stretch=1)

        # Duration label
        self.duration_label = QLabel("00:00.0")
        self.duration_label.setMinimumWidth(60)
        self.duration_label.setStyleSheet("font-family: monospace; font-size: 12pt;")
        layout.addWidget(self.duration_label)

        # Fast forward
        self.forward_btn = QPushButton("⏩")
        self.forward_btn.setFixedSize(30, 30)
        self.forward_btn.setToolTip("Forward 10s")
        layout.addWidget(self.forward_btn)

        self.skip_end_btn = QPushButton("⏭")
        self.skip_end_btn.setFixedSize(30, 30)
        self.skip_end_btn.setToolTip("Skip to end")
        layout.addWidget(self.skip_end_btn)

        # Speed selector
        layout.addWidget(QLabel("Speed:"))

        self.speed_combo = QComboBox()
        self.speed_combo.addItems([
            "0.25x", "0.5x", "0.75x", "1x", "1.5x", "2x", "4x"
        ])
        self.speed_combo.setCurrentIndex(3)  # 1x
        self.speed_combo.setFixedWidth(70)
        self.speed_combo.setToolTip("Playback speed")
        layout.addWidget(self.speed_combo)

        # Apply styling
        self.setStyleSheet("""
            QPushButton {
                background: #2a3a4a;
                border: 1px solid #3d4d5d;
                border-radius: 4px;
                color: #ffffff;
                font-size: 14pt;
            }
            QPushButton:hover {
                background: #3a4a5a;
                border-color: #00d4aa;
            }
            QPushButton:pressed {
                background: #4a5a6a;
            }
            QPushButton#playButton {
                background: #00aa88;
            }
            QPushButton#playButton:hover {
                background: #00ccaa;
            }
            QPushButton#stopButton {
                background: #aa4444;
            }
            QPushButton#stopButton:hover {
                background: #cc5555;
            }
            QSlider::groove:horizontal {
                background: #2a3a4a;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00d4aa;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #00aa88;
                border-radius: 4px;
            }
            QComboBox {
                background: #2a3a4a;
                border: 1px solid #3d4d5d;
                border-radius: 4px;
                padding: 4px;
                color: #ffffff;
            }
        """)

    def _connect_signals(self):
        """Connect internal signals."""
        self.play_btn.clicked.connect(self._toggle_playback)
        self.stop_btn.clicked.connect(self._on_stop)
        self.skip_back_btn.clicked.connect(self._skip_to_start)
        self.skip_end_btn.clicked.connect(self._skip_to_end)
        self.rewind_btn.clicked.connect(self._rewind_10s)
        self.forward_btn.clicked.connect(self._forward_10s)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)

        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)

    # =========================================================================
    # Public API
    # =========================================================================

    def set_duration(self, seconds: float):
        """Set the total duration of the recording."""
        self._duration = max(0.1, seconds)
        self.duration_label.setText(self._format_time(self._duration))

    def set_position(self, seconds: float):
        """Set the current playback position."""
        self._current_time = max(0, min(seconds, self._duration))
        self._update_ui()

    def get_position(self) -> float:
        """Get current position in seconds."""
        return self._current_time

    def get_normalized_position(self) -> float:
        """Get position as 0.0-1.0."""
        if self._duration <= 0:
            return 0.0
        return self._current_time / self._duration

    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._is_playing

    def play(self):
        """Start playback."""
        if not self._is_playing:
            self._toggle_playback()

    def pause(self):
        """Pause playback."""
        if self._is_playing:
            self._toggle_playback()

    def stop(self):
        """Stop and rewind."""
        self._on_stop()

    # =========================================================================
    # Internal handlers
    # =========================================================================

    def _toggle_playback(self):
        """Toggle play/pause state."""
        self._is_playing = not self._is_playing

        if self._is_playing:
            self.play_btn.setText("⏸")
            self.play_btn.setToolTip("Pause")
            interval = int(100 / self._speed)  # Update every 100ms (at 1x)
            self._timer.start(max(10, interval))
        else:
            self.play_btn.setText("▶")
            self.play_btn.setToolTip("Play")
            self._timer.stop()

        self.playback_toggled.emit(self._is_playing)

    def _on_stop(self):
        """Handle stop button."""
        self._is_playing = False
        self._current_time = 0.0
        self.play_btn.setText("▶")
        self._timer.stop()
        self._update_ui()
        self.stopped.emit()

    def _skip_to_start(self):
        """Jump to beginning."""
        self._current_time = 0.0
        self._update_ui()
        self.position_changed.emit(0.0)

    def _skip_to_end(self):
        """Jump to end."""
        self._current_time = self._duration
        self._update_ui()
        self.position_changed.emit(1.0)

    def _rewind_10s(self):
        """Rewind 10 seconds."""
        self._current_time = max(0, self._current_time - 10.0)
        self._update_ui()
        self.position_changed.emit(self.get_normalized_position())

    def _forward_10s(self):
        """Forward 10 seconds."""
        self._current_time = min(self._duration, self._current_time + 10.0)
        self._update_ui()
        self.position_changed.emit(self.get_normalized_position())

    def _on_slider_pressed(self):
        """User started dragging slider."""
        self._timer.stop()

    def _on_slider_released(self):
        """User finished dragging slider."""
        if self._is_playing:
            self._timer.start()
        self.position_changed.emit(self.get_normalized_position())

    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        if self.slider.isSliderDown():
            # User is dragging
            normalized = value / 1000.0
            self._current_time = normalized * self._duration
            self.time_label.setText(self._format_time(self._current_time))

    def _on_speed_changed(self, index: int):
        """Handle speed selection change."""
        speeds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
        self._speed = speeds[index]

        if self._is_playing:
            interval = int(100 / self._speed)
            self._timer.setInterval(max(10, interval))

        self.speed_changed.emit(self._speed)

    def _on_timer_tick(self):
        """Timer callback - advance playback."""
        delta = 0.1 * self._speed  # 100ms timer * speed
        self._current_time += delta

        if self._current_time >= self._duration:
            self._current_time = self._duration
            self._toggle_playback()  # Auto-pause at end

        self._update_ui()
        self.position_changed.emit(self.get_normalized_position())

    def _update_ui(self):
        """Update UI elements to reflect current state."""
        self.time_label.setText(self._format_time(self._current_time))

        # Update slider without triggering signals
        self.slider.blockSignals(True)
        if self._duration > 0:
            slider_value = int((self._current_time / self._duration) * 1000)
            self.slider.setValue(slider_value)
        self.slider.blockSignals(False)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS.d."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:04.1f}"


class ReplayControlBar(QFrame):
    """
    Complete replay control bar with timeline and status.

    Wraps TimelineScrubber with additional flight info display.
    """

    position_changed = pyqtSignal(float)
    playback_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("replayControlBar")
        self.setFixedHeight(60)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Timeline scrubber
        self.timeline = TimelineScrubber()
        layout.addWidget(self.timeline)

        # Forward signals
        self.timeline.position_changed.connect(self.position_changed.emit)
        self.timeline.playback_toggled.connect(self.playback_toggled.emit)

        # Styling
        self.setStyleSheet("""
            #replayControlBar {
                background: #1a2530;
                border-top: 1px solid #00d4aa;
            }
        """)

    def set_recording(self, duration: float, name: str = ""):
        """Set recording information."""
        self.timeline.set_duration(duration)

    def update_position(self, seconds: float):
        """Update current position (from external source)."""
        self.timeline.set_position(seconds)
