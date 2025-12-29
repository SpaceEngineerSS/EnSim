"""
Terrain Awareness and Warning System (TAWS / GPWS).

Enhanced Ground Proximity Warning System implementation for
flight simulation applications.

Modes:
    1. Excessive descent rate
    2. Excessive terrain closure rate
    3. Altitude loss after takeoff
    4. Unsafe terrain clearance
    5. Below glideslope
    6. Bank angle callouts
    7. Windshear warning

References:
    1. RTCA DO-161A - MOPS for GPWS
    2. FAA TSO-C151c - TAWS Equipment
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from enum import Enum


class WarningSeverity(Enum):
    """Warning severity level."""
    ADVISORY = "advisory"    # Informational callout
    CAUTION = "caution"      # Amber warning, crew awareness
    WARNING = "warning"      # Red warning, immediate action required


class WarningType(Enum):
    """GPWS warning types."""
    SINK_RATE = "SINK RATE"
    PULL_UP = "PULL UP"
    TERRAIN = "TERRAIN"
    TOO_LOW_TERRAIN = "TOO LOW TERRAIN"
    TOO_LOW_GEAR = "TOO LOW GEAR"
    TOO_LOW_FLAPS = "TOO LOW FLAPS"
    DONT_SINK = "DON'T SINK"
    GLIDESLOPE = "GLIDESLOPE"
    BANK_ANGLE = "BANK ANGLE"
    WINDSHEAR = "WINDSHEAR"
    ALTITUDE_CALLOUT = "ALTITUDE"


@dataclass
class TerrainWarning:
    """A terrain awareness warning."""
    type: WarningType
    severity: WarningSeverity
    message: str
    altitude_agl: float = 0.0  # feet
    closure_rate: float = 0.0  # ft/min
    priority: int = 0  # Higher = more urgent
    
    def __str__(self) -> str:
        icon = {
            WarningSeverity.ADVISORY: "ℹ️",
            WarningSeverity.CAUTION: "⚠️",
            WarningSeverity.WARNING: "🔴"
        }[self.severity]
        return f"{icon} {self.message}"


@dataclass
class AircraftState:
    """Current aircraft state for TAWS evaluation."""
    altitude_msl: float  # feet MSL
    altitude_agl: float  # feet AGL (radar altimeter)
    vertical_speed: float  # feet/min
    groundspeed: float  # knots
    heading: float  # degrees true
    bank_angle: float  # degrees
    pitch_angle: float  # degrees
    gear_down: bool = False
    flaps_position: float = 0.0  # 0.0 to 1.0
    on_ground: bool = False
    glideslope_deviation: float = 0.0  # dots (positive = above)
    localizer_deviation: float = 0.0  # dots (positive = right)
    is_approach_mode: bool = False
    time_since_takeoff: float = 0.0  # seconds


@dataclass
class TAWSConfiguration:
    """TAWS system configuration."""
    # Mode 1 - Excessive descent rate thresholds
    mode1_enabled: bool = True
    mode1_sink_rate_caution: float = -1000.0  # ft/min
    mode1_sink_rate_warning: float = -2500.0  # ft/min
    
    # Mode 2 - Terrain closure
    mode2_enabled: bool = True
    mode2_closure_rate_factor: float = 0.15
    
    # Mode 3 - Altitude loss after takeoff
    mode3_enabled: bool = True
    mode3_max_alt_loss: float = 50.0  # feet
    
    # Mode 4 - Unsafe terrain clearance
    mode4_enabled: bool = True
    mode4_gear_altitude: float = 500.0  # feet
    mode4_flaps_altitude: float = 245.0  # feet
    
    # Mode 5 - Glideslope
    mode5_enabled: bool = True
    mode5_gs_deviation_soft: float = 1.3  # dots
    mode5_gs_deviation_hard: float = 2.0  # dots
    
    # Mode 6 - Bank angle
    mode6_enabled: bool = True
    mode6_bank_angle_caution: float = 35.0  # degrees
    mode6_bank_angle_warning: float = 45.0  # degrees
    
    # Mode 7 - Windshear
    mode7_enabled: bool = True
    
    # Altitude callouts
    callouts_enabled: bool = True
    callout_altitudes: Tuple[float, ...] = (2500, 1000, 500, 400, 300, 200, 100, 50, 40, 30, 20, 10)


class TerrainAwarenessSystem:
    """
    Ground Proximity Warning System (GPWS/TAWS) implementation.
    
    Monitors aircraft state and generates appropriate warnings
    for terrain proximity and flight configuration issues.
    """
    
    def __init__(self, config: Optional[TAWSConfiguration] = None):
        """Initialize TAWS with configuration."""
        self.config = config or TAWSConfiguration()
        self._active_warnings: List[TerrainWarning] = []
        self._previous_state: Optional[AircraftState] = None
        self._max_altitude_since_takeoff: float = 0.0
        self._last_callout_altitude: float = float('inf')
        self._inhibited: bool = False
    
    def inhibit(self, inhibit: bool = True) -> None:
        """Inhibit all TAWS warnings (e.g., during intentional low flight)."""
        self._inhibited = inhibit
    
    def reset(self) -> None:
        """Reset TAWS state (call on new flight)."""
        self._active_warnings = []
        self._previous_state = None
        self._max_altitude_since_takeoff = 0.0
        self._last_callout_altitude = float('inf')
    
    def update(self, state: AircraftState) -> List[TerrainWarning]:
        """
        Update TAWS with current aircraft state.
        
        Args:
            state: Current aircraft state
            
        Returns:
            List of active warnings
        """
        if self._inhibited or state.on_ground:
            self._previous_state = state
            return []
        
        warnings = []
        
        # Track max altitude since takeoff for Mode 3
        if state.altitude_agl > self._max_altitude_since_takeoff:
            self._max_altitude_since_takeoff = state.altitude_agl
        
        # Mode 1: Excessive descent rate
        if self.config.mode1_enabled:
            mode1_warnings = self._check_mode1(state)
            warnings.extend(mode1_warnings)
        
        # Mode 2: Terrain closure rate
        if self.config.mode2_enabled and self._previous_state:
            mode2_warnings = self._check_mode2(state, self._previous_state)
            warnings.extend(mode2_warnings)
        
        # Mode 3: Altitude loss after takeoff
        if self.config.mode3_enabled:
            mode3_warnings = self._check_mode3(state)
            warnings.extend(mode3_warnings)
        
        # Mode 4: Unsafe terrain clearance
        if self.config.mode4_enabled:
            mode4_warnings = self._check_mode4(state)
            warnings.extend(mode4_warnings)
        
        # Mode 5: Below glideslope
        if self.config.mode5_enabled and state.is_approach_mode:
            mode5_warnings = self._check_mode5(state)
            warnings.extend(mode5_warnings)
        
        # Mode 6: Bank angle
        if self.config.mode6_enabled:
            mode6_warnings = self._check_mode6(state)
            warnings.extend(mode6_warnings)
        
        # Altitude callouts
        if self.config.callouts_enabled:
            callouts = self._check_altitude_callouts(state)
            warnings.extend(callouts)
        
        # Sort by priority
        warnings.sort(key=lambda w: w.priority, reverse=True)
        
        # Update state
        self._previous_state = state
        self._active_warnings = warnings
        
        return warnings
    
    def _check_mode1(self, state: AircraftState) -> List[TerrainWarning]:
        """Mode 1: Excessive descent rate."""
        warnings = []
        
        if state.vertical_speed <= 0 and state.altitude_agl < 2500:
            vs = state.vertical_speed
            agl = state.altitude_agl
            
            # Envelope depends on altitude
            warning_threshold = self.config.mode1_sink_rate_warning
            caution_threshold = self.config.mode1_sink_rate_caution
            
            # Adjust thresholds with altitude (tighter limits at low altitude)
            if agl < 1000:
                warning_threshold *= 0.8
                caution_threshold *= 0.8
            
            if vs < warning_threshold:
                warnings.append(TerrainWarning(
                    type=WarningType.PULL_UP,
                    severity=WarningSeverity.WARNING,
                    message="PULL UP! PULL UP!",
                    altitude_agl=agl,
                    closure_rate=abs(vs),
                    priority=10
                ))
            elif vs < caution_threshold:
                warnings.append(TerrainWarning(
                    type=WarningType.SINK_RATE,
                    severity=WarningSeverity.CAUTION,
                    message="SINK RATE",
                    altitude_agl=agl,
                    closure_rate=abs(vs),
                    priority=7
                ))
        
        return warnings
    
    def _check_mode2(
        self, 
        state: AircraftState, 
        prev: AircraftState
    ) -> List[TerrainWarning]:
        """Mode 2: Excessive terrain closure rate."""
        warnings = []
        
        if state.altitude_agl < 2000:
            # Terrain closure = rate of change of AGL
            dt = 0.1  # Assume 10Hz update rate
            closure_rate = (prev.altitude_agl - state.altitude_agl) / dt * 60  # ft/min
            
            if closure_rate > 0:  # Closing on terrain
                # Threshold depends on altitude and closure rate
                threshold = state.altitude_agl * self.config.mode2_closure_rate_factor * 60
                
                if closure_rate > threshold and state.altitude_agl < 1000:
                    warnings.append(TerrainWarning(
                        type=WarningType.TERRAIN,
                        severity=WarningSeverity.WARNING,
                        message="TERRAIN! TERRAIN! PULL UP!",
                        altitude_agl=state.altitude_agl,
                        closure_rate=closure_rate,
                        priority=10
                    ))
                elif closure_rate > threshold * 0.7:
                    warnings.append(TerrainWarning(
                        type=WarningType.TERRAIN,
                        severity=WarningSeverity.CAUTION,
                        message="TERRAIN",
                        altitude_agl=state.altitude_agl,
                        closure_rate=closure_rate,
                        priority=8
                    ))
        
        return warnings
    
    def _check_mode3(self, state: AircraftState) -> List[TerrainWarning]:
        """Mode 3: Altitude loss after takeoff."""
        warnings = []
        
        # Only active in initial climb (first 30 seconds)
        if state.time_since_takeoff > 0 and state.time_since_takeoff < 30:
            alt_loss = self._max_altitude_since_takeoff - state.altitude_agl
            
            if alt_loss > self.config.mode3_max_alt_loss:
                warnings.append(TerrainWarning(
                    type=WarningType.DONT_SINK,
                    severity=WarningSeverity.CAUTION,
                    message="DON'T SINK",
                    altitude_agl=state.altitude_agl,
                    priority=6
                ))
        
        return warnings
    
    def _check_mode4(self, state: AircraftState) -> List[TerrainWarning]:
        """Mode 4: Unsafe terrain clearance (gear/flaps)."""
        warnings = []
        
        # 4A: Too low gear
        if not state.gear_down and state.altitude_agl < self.config.mode4_gear_altitude:
            if not state.on_ground:
                warnings.append(TerrainWarning(
                    type=WarningType.TOO_LOW_GEAR,
                    severity=WarningSeverity.CAUTION,
                    message="TOO LOW - GEAR",
                    altitude_agl=state.altitude_agl,
                    priority=5
                ))
        
        # 4B: Too low flaps
        if state.flaps_position < 0.3 and state.altitude_agl < self.config.mode4_flaps_altitude:
            if not state.on_ground and state.is_approach_mode:
                warnings.append(TerrainWarning(
                    type=WarningType.TOO_LOW_FLAPS,
                    severity=WarningSeverity.CAUTION,
                    message="TOO LOW - FLAPS",
                    altitude_agl=state.altitude_agl,
                    priority=5
                ))
        
        # 4C: Terrain at low altitude
        if state.altitude_agl < 250 and not state.is_approach_mode:
            warnings.append(TerrainWarning(
                type=WarningType.TOO_LOW_TERRAIN,
                severity=WarningSeverity.CAUTION,
                message="TOO LOW TERRAIN",
                altitude_agl=state.altitude_agl,
                priority=6
            ))
        
        return warnings
    
    def _check_mode5(self, state: AircraftState) -> List[TerrainWarning]:
        """Mode 5: Below glideslope."""
        warnings = []
        
        if state.glideslope_deviation < 0:  # Below glideslope
            deviation = abs(state.glideslope_deviation)
            
            if deviation > self.config.mode5_gs_deviation_hard:
                warnings.append(TerrainWarning(
                    type=WarningType.GLIDESLOPE,
                    severity=WarningSeverity.WARNING,
                    message="GLIDESLOPE! GLIDESLOPE!",
                    altitude_agl=state.altitude_agl,
                    priority=7
                ))
            elif deviation > self.config.mode5_gs_deviation_soft:
                warnings.append(TerrainWarning(
                    type=WarningType.GLIDESLOPE,
                    severity=WarningSeverity.CAUTION,
                    message="GLIDESLOPE",
                    altitude_agl=state.altitude_agl,
                    priority=4
                ))
        
        return warnings
    
    def _check_mode6(self, state: AircraftState) -> List[TerrainWarning]:
        """Mode 6: Bank angle warning."""
        warnings = []
        
        bank = abs(state.bank_angle)
        
        if bank > self.config.mode6_bank_angle_warning:
            warnings.append(TerrainWarning(
                type=WarningType.BANK_ANGLE,
                severity=WarningSeverity.WARNING,
                message="BANK ANGLE! BANK ANGLE!",
                altitude_agl=state.altitude_agl,
                priority=8
            ))
        elif bank > self.config.mode6_bank_angle_caution:
            warnings.append(TerrainWarning(
                type=WarningType.BANK_ANGLE,
                severity=WarningSeverity.CAUTION,
                message="BANK ANGLE",
                altitude_agl=state.altitude_agl,
                priority=4
            ))
        
        return warnings
    
    def _check_altitude_callouts(self, state: AircraftState) -> List[TerrainWarning]:
        """Generate altitude callouts during approach."""
        callouts = []
        
        if not state.gear_down:  # Only call out with gear down (approach)
            return callouts
        
        agl = state.altitude_agl
        
        for alt in self.config.callout_altitudes:
            # Call out when passing through (hysteresis)
            if self._last_callout_altitude > alt >= agl:
                callouts.append(TerrainWarning(
                    type=WarningType.ALTITUDE_CALLOUT,
                    severity=WarningSeverity.ADVISORY,
                    message=f"{int(alt)}",
                    altitude_agl=agl,
                    priority=1
                ))
                self._last_callout_altitude = alt
                break
        
        return callouts
    
    @property
    def active_warnings(self) -> List[TerrainWarning]:
        """Get currently active warnings."""
        return self._active_warnings.copy()
    
    @property
    def highest_priority_warning(self) -> Optional[TerrainWarning]:
        """Get the highest priority active warning."""
        if not self._active_warnings:
            return None
        return self._active_warnings[0]


def create_gpws_for_flight_sim() -> TerrainAwarenessSystem:
    """
    Create a TAWS configured for flight simulation.
    
    Returns:
        Configured TerrainAwarenessSystem
    """
    config = TAWSConfiguration(
        mode1_sink_rate_caution=-1500.0,  # More permissive for sim
        mode1_sink_rate_warning=-3000.0,
        mode6_bank_angle_caution=40.0,
        mode6_bank_angle_warning=50.0,
    )
    return TerrainAwarenessSystem(config)
