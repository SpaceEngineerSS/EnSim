"""UI Widgets package."""

from .graph_widget import PerformanceGraph
from .input_panel import InputPanel
from .timeline_scrubber import ReplayControlBar, TimelineScrubber
from .view3d_widget import NozzleView3D

# New widgets (v2.1)
from .staging_widget import MultiStageWidget, StageConfigCard
from .optimization_widget import OptimizationWidget
from .cooling_widget import CoolingAnalysisWidget
from .propellant_presets_widget import PropellantPresetWidget
from .unit_toggle_widget import UnitToggleWidget, UnitSystemBar, UnitDisplayLabel

__all__ = [
    # Original
    "InputPanel",
    "PerformanceGraph",
    "NozzleView3D",
    "TimelineScrubber",
    "ReplayControlBar",
    # New (v2.1)
    "MultiStageWidget",
    "StageConfigCard",
    "OptimizationWidget",
    "CoolingAnalysisWidget",
    "PropellantPresetWidget",
    "UnitToggleWidget",
    "UnitSystemBar",
    "UnitDisplayLabel",
]
