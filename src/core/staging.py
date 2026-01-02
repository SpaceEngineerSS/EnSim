"""
Multi-stage rocket vehicle model with staging logic.

Provides comprehensive support for multi-stage launch vehicles including:
- Stage definition with propulsion and mass properties
- Staging event detection and execution
- Payload fairing jettison
- Coast phases between burns

References:
    - Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed., Ch. 4
    - Hill & Peterson, "Mechanics and Thermodynamics of Propulsion", 2nd ed.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import numpy as np
from numba import jit


class StageStatus(Enum):
    """Stage operational status."""
    INACTIVE = auto()     # Not yet active
    BURNING = auto()      # Engine firing
    COASTING = auto()     # Engine off, still attached
    SEPARATED = auto()    # Jettisoned from stack
    EXPENDED = auto()     # Completely expended


class StagingTrigger(Enum):
    """Conditions that trigger stage separation."""
    PROPELLANT_DEPLETION = auto()  # When propellant runs out
    TIME_BASED = auto()            # At specific mission time
    ALTITUDE_BASED = auto()        # At specific altitude
    VELOCITY_BASED = auto()        # At specific velocity
    MANUAL = auto()                # Manual trigger


@dataclass
class StageEngine:
    """
    Engine parameters for a stage.

    Attributes:
        thrust_sl: Sea-level thrust (N)
        thrust_vac: Vacuum thrust (N)
        isp_sl: Sea-level specific impulse (s)
        isp_vac: Vacuum specific impulse (s)
        num_engines: Number of engines
        throttle_min: Minimum throttle (0-1)
        throttle_max: Maximum throttle (0-1)
        gimbal_range_deg: Gimbal range (degrees)
        ignition_time: Engine ignition delay (s)
        shutdown_time: Engine shutdown time (s)
    """
    thrust_sl: float
    thrust_vac: float
    isp_sl: float
    isp_vac: float
    num_engines: int = 1
    throttle_min: float = 0.6
    throttle_max: float = 1.0
    gimbal_range_deg: float = 5.0
    ignition_time: float = 0.5
    shutdown_time: float = 0.3

    def get_thrust(self, altitude: float, throttle: float = 1.0) -> float:
        """
        Calculate thrust at given altitude and throttle.

        Uses exponential atmosphere model for interpolation.

        Args:
            altitude: Altitude above sea level (m)
            throttle: Throttle setting (0-1)

        Returns:
            Total thrust (N)
        """
        # Atmospheric pressure decay (scale height ~8.5 km)
        h_scale = 8500.0
        atm_factor = np.exp(-altitude / h_scale)

        # Interpolate between sea-level and vacuum
        thrust = self.thrust_vac - (self.thrust_vac - self.thrust_sl) * atm_factor

        # Apply throttle and engine count
        throttle_clamped = np.clip(throttle, self.throttle_min, self.throttle_max)
        return thrust * throttle_clamped * self.num_engines

    def get_isp(self, altitude: float) -> float:
        """
        Calculate Isp at given altitude.

        Args:
            altitude: Altitude above sea level (m)

        Returns:
            Specific impulse (s)
        """
        h_scale = 8500.0
        atm_factor = np.exp(-altitude / h_scale)
        return self.isp_vac - (self.isp_vac - self.isp_sl) * atm_factor

    def get_mass_flow_rate(self, altitude: float, throttle: float = 1.0) -> float:
        """
        Calculate mass flow rate.

        Args:
            altitude: Altitude (m)
            throttle: Throttle setting (0-1)

        Returns:
            Mass flow rate (kg/s)
        """
        G0 = 9.80665  # Standard gravity
        thrust = self.get_thrust(altitude, throttle)
        isp = self.get_isp(altitude)
        return thrust / (isp * G0) if isp > 0 else 0.0


@dataclass
class Stage:
    """
    Complete stage definition.

    Attributes:
        name: Stage identifier
        dry_mass: Stage dry mass without propellant (kg)
        propellant_mass: Total propellant mass (kg)
        engine: Engine parameters
        staging_trigger: What triggers separation
        trigger_value: Value for time/altitude/velocity triggers
        fairing_mass: Payload fairing mass if attached (kg)
        fairing_jettison_alt: Altitude to jettison fairing (m), None to keep
        interstage_mass: Interstage adapter mass (kg)
        separation_impulse: Separation system impulse (N·s)
        ullage_thrust: Ullage motor thrust for settling (N)
        ullage_duration: Ullage motor burn duration (s)
    """
    name: str
    dry_mass: float
    propellant_mass: float
    engine: StageEngine
    staging_trigger: StagingTrigger = StagingTrigger.PROPELLANT_DEPLETION
    trigger_value: float | None = None
    fairing_mass: float = 0.0
    fairing_jettison_alt: float | None = None
    interstage_mass: float = 0.0
    separation_impulse: float = 0.0
    ullage_thrust: float = 0.0
    ullage_duration: float = 0.5

    # Runtime state
    status: StageStatus = field(default=StageStatus.INACTIVE)
    propellant_remaining: float = field(default=-1.0)  # -1 means uninitialized
    burn_time: float = field(default=0.0)
    fairing_jettisoned: bool = field(default=False)

    def __post_init__(self):
        if self.propellant_remaining < 0:
            self.propellant_remaining = self.propellant_mass

    @property
    def total_mass(self) -> float:
        """Total stage mass including remaining propellant."""
        mass = self.dry_mass + self.propellant_remaining
        if not self.fairing_jettisoned:
            mass += self.fairing_mass
        return mass

    @property
    def propellant_fraction(self) -> float:
        """Remaining propellant as fraction of initial."""
        if self.propellant_mass > 0:
            return self.propellant_remaining / self.propellant_mass
        return 0.0

    @property
    def mass_ratio(self) -> float:
        """Initial to final mass ratio (for delta-v calculation)."""
        m_initial = self.dry_mass + self.propellant_mass
        m_final = self.dry_mass
        return m_initial / m_final if m_final > 0 else 1.0

    def get_ideal_delta_v(self, payload_mass: float = 0.0) -> float:
        """
        Calculate ideal delta-v using Tsiolkovsky equation.

        ΔV = Isp × g0 × ln(m_initial / m_final)

        Args:
            payload_mass: Additional payload mass (kg)

        Returns:
            Ideal delta-v (m/s)
        """
        G0 = 9.80665
        m_initial = self.dry_mass + self.propellant_mass + payload_mass
        m_final = self.dry_mass + payload_mass
        if m_final <= 0:
            return 0.0
        return self.engine.isp_vac * G0 * np.log(m_initial / m_final)

    def consume_propellant(self, mass_flow: float, dt: float) -> float:
        """
        Consume propellant over timestep.

        Args:
            mass_flow: Mass flow rate (kg/s)
            dt: Timestep (s)

        Returns:
            Actual mass consumed (kg)
        """
        requested = mass_flow * dt
        actual = min(requested, self.propellant_remaining)
        self.propellant_remaining -= actual
        self.burn_time += dt
        return actual

    def is_depleted(self) -> bool:
        """Check if propellant is exhausted."""
        return self.propellant_remaining <= 0.0

    def reset(self):
        """Reset stage to initial state."""
        self.propellant_remaining = self.propellant_mass
        self.status = StageStatus.INACTIVE
        self.burn_time = 0.0
        self.fairing_jettisoned = False


@dataclass
class StagingEvent:
    """Record of a staging event."""
    time: float
    stage_name: str
    event_type: str  # 'separation', 'ignition', 'shutdown', 'fairing_jettison'
    altitude: float
    velocity: float
    mass_before: float
    mass_after: float


@dataclass
class MultiStageVehicle:
    """
    Multi-stage launch vehicle model.

    Handles staging logic, mass tracking, and thrust calculations
    for vehicles with arbitrary number of stages.

    Stages are numbered from bottom (1) to top (N), with the payload
    on top of stage N.

    Attributes:
        name: Vehicle name
        stages: List of stages (index 0 = bottom stage)
        payload_mass: Payload mass on top (kg)
        launch_site_altitude: Launch site altitude MSL (m)
    """
    name: str
    stages: list[Stage]
    payload_mass: float = 0.0
    launch_site_altitude: float = 0.0

    # Runtime state
    active_stage_idx: int = field(default=0)
    mission_time: float = field(default=0.0)
    events: list[StagingEvent] = field(default_factory=list)

    def __post_init__(self):
        # Initialize first stage as active
        if self.stages:
            self.stages[0].status = StageStatus.BURNING

    @property
    def active_stage(self) -> Stage | None:
        """Get currently active stage."""
        if 0 <= self.active_stage_idx < len(self.stages):
            return self.stages[self.active_stage_idx]
        return None

    @property
    def total_mass(self) -> float:
        """Total vehicle mass including all attached stages and payload."""
        mass = self.payload_mass
        for i, stage in enumerate(self.stages):
            if stage.status not in [StageStatus.SEPARATED, StageStatus.EXPENDED]:
                mass += stage.total_mass
                # Add interstage for stages above active
                if i > self.active_stage_idx:
                    mass += stage.interstage_mass
        return mass

    @property
    def remaining_stages(self) -> int:
        """Number of stages not yet separated."""
        return sum(1 for s in self.stages
                   if s.status not in [StageStatus.SEPARATED, StageStatus.EXPENDED])

    def get_thrust(self, altitude: float, throttle: float = 1.0) -> float:
        """
        Get current total thrust.

        Args:
            altitude: Current altitude (m)
            throttle: Throttle setting (0-1)

        Returns:
            Total thrust (N)
        """
        stage = self.active_stage
        if stage is None or stage.status != StageStatus.BURNING:
            return 0.0
        return stage.engine.get_thrust(altitude, throttle)

    def get_mass_flow(self, altitude: float, throttle: float = 1.0) -> float:
        """
        Get current mass flow rate.

        Args:
            altitude: Current altitude (m)
            throttle: Throttle setting (0-1)

        Returns:
            Mass flow rate (kg/s)
        """
        stage = self.active_stage
        if stage is None or stage.status != StageStatus.BURNING:
            return 0.0
        return stage.engine.get_mass_flow_rate(altitude, throttle)

    def get_isp(self, altitude: float) -> float:
        """Get current specific impulse."""
        stage = self.active_stage
        if stage is None:
            return 0.0
        return stage.engine.get_isp(altitude)

    def step(
        self,
        dt: float,
        altitude: float,
        velocity: float,
        throttle: float = 1.0
    ) -> list[StagingEvent]:
        """
        Advance vehicle state by one timestep.

        Handles propellant consumption, staging checks, and
        fairing jettison.

        Args:
            dt: Timestep (s)
            altitude: Current altitude (m)
            velocity: Current velocity magnitude (m/s)
            throttle: Throttle setting (0-1)

        Returns:
            List of staging events that occurred
        """
        events = []
        self.mission_time += dt

        stage = self.active_stage
        if stage is None:
            return events

        # Consume propellant if burning
        if stage.status == StageStatus.BURNING:
            mass_flow = stage.engine.get_mass_flow_rate(altitude, throttle)
            stage.consume_propellant(mass_flow, dt)

        # Check fairing jettison
        if (not stage.fairing_jettisoned and
            stage.fairing_jettison_alt is not None and
            altitude >= stage.fairing_jettison_alt):

            mass_before = self.total_mass
            stage.fairing_jettisoned = True
            mass_after = self.total_mass

            event = StagingEvent(
                time=self.mission_time,
                stage_name=stage.name,
                event_type='fairing_jettison',
                altitude=altitude,
                velocity=velocity,
                mass_before=mass_before,
                mass_after=mass_after
            )
            events.append(event)
            self.events.append(event)

        # Check staging trigger
        should_stage = self._check_staging_trigger(
            stage, altitude, velocity
        )

        if should_stage:
            events.extend(self._execute_staging(altitude, velocity))

        return events

    def _check_staging_trigger(
        self,
        stage: Stage,
        altitude: float,
        velocity: float
    ) -> bool:
        """Check if staging should occur."""
        if stage.staging_trigger == StagingTrigger.PROPELLANT_DEPLETION:
            return stage.is_depleted()

        elif stage.staging_trigger == StagingTrigger.TIME_BASED:
            return (stage.trigger_value is not None and
                    self.mission_time >= stage.trigger_value)

        elif stage.staging_trigger == StagingTrigger.ALTITUDE_BASED:
            return (stage.trigger_value is not None and
                    altitude >= stage.trigger_value)

        elif stage.staging_trigger == StagingTrigger.VELOCITY_BASED:
            return (stage.trigger_value is not None and
                    velocity >= stage.trigger_value)

        return False

    def _execute_staging(
        self,
        altitude: float,
        velocity: float
    ) -> list[StagingEvent]:
        """Execute stage separation and next stage ignition."""
        events = []
        stage = self.active_stage

        if stage is None:
            return events

        # Record mass before separation
        mass_before = self.total_mass

        # Shutdown current stage
        stage.status = StageStatus.SEPARATED
        events.append(StagingEvent(
            time=self.mission_time,
            stage_name=stage.name,
            event_type='separation',
            altitude=altitude,
            velocity=velocity,
            mass_before=mass_before,
            mass_after=self.total_mass
        ))

        # Activate next stage
        self.active_stage_idx += 1
        next_stage = self.active_stage

        if next_stage is not None:
            next_stage.status = StageStatus.BURNING
            events.append(StagingEvent(
                time=self.mission_time + next_stage.engine.ignition_time,
                stage_name=next_stage.name,
                event_type='ignition',
                altitude=altitude,
                velocity=velocity,
                mass_before=self.total_mass,
                mass_after=self.total_mass
            ))

        self.events.extend(events)
        return events

    def get_total_delta_v(self) -> float:
        """
        Calculate total ideal delta-v of the vehicle.

        Uses serial staging calculation where each stage
        carries upper stages as payload.

        Returns:
            Total delta-v (m/s)
        """
        total_dv = 0.0
        payload = self.payload_mass

        # Calculate from top stage down
        for stage in reversed(self.stages):
            dv = stage.get_ideal_delta_v(payload)
            total_dv += dv
            # Current stage becomes part of payload for stage below
            payload += stage.dry_mass + stage.propellant_mass

        return total_dv

    def get_stage_delta_v_breakdown(self) -> list[tuple[str, float]]:
        """
        Get delta-v contribution of each stage.

        Returns:
            List of (stage_name, delta_v) tuples
        """
        breakdown = []
        payload = self.payload_mass

        for stage in reversed(self.stages):
            dv = stage.get_ideal_delta_v(payload)
            breakdown.append((stage.name, dv))
            payload += stage.dry_mass + stage.propellant_mass

        # Reverse to get bottom-to-top order
        return list(reversed(breakdown))

    def reset(self):
        """Reset vehicle to launch-ready state."""
        self.active_stage_idx = 0
        self.mission_time = 0.0
        self.events.clear()
        for stage in self.stages:
            stage.reset()
        if self.stages:
            self.stages[0].status = StageStatus.BURNING


# =============================================================================
# Factory Functions for Common Vehicles
# =============================================================================

def create_falcon_9_like() -> MultiStageVehicle:
    """
    Create a Falcon 9-like two-stage vehicle.

    Based on public specifications for Falcon 9 Full Thrust.

    Returns:
        MultiStageVehicle configured like Falcon 9
    """
    # First stage (9 Merlin 1D engines)
    s1_engine = StageEngine(
        thrust_sl=7_607_000,   # 7.607 MN at sea level
        thrust_vac=8_227_000,  # 8.227 MN in vacuum
        isp_sl=282,
        isp_vac=311,
        num_engines=9,
        throttle_min=0.4,
        throttle_max=1.0,
        gimbal_range_deg=5.0
    )

    stage1 = Stage(
        name="S1",
        dry_mass=25_600,       # kg
        propellant_mass=411_000,  # kg (LOX + RP-1)
        engine=s1_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION
    )

    # Second stage (1 Merlin Vacuum)
    s2_engine = StageEngine(
        thrust_sl=0,           # Vacuum-only engine
        thrust_vac=934_000,    # 934 kN
        isp_sl=0,
        isp_vac=348,
        num_engines=1,
        throttle_min=0.4,
        throttle_max=1.0,
        gimbal_range_deg=6.0
    )

    stage2 = Stage(
        name="S2",
        dry_mass=4_000,        # kg
        propellant_mass=111_500,  # kg
        engine=s2_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
        fairing_mass=1_900,    # Fairing mass
        fairing_jettison_alt=110_000  # ~110 km
    )

    return MultiStageVehicle(
        name="Falcon 9 (Full Thrust)",
        stages=[stage1, stage2],
        payload_mass=0,  # Set when needed
        launch_site_altitude=0
    )


def create_saturn_v_like() -> MultiStageVehicle:
    """
    Create a Saturn V-like three-stage vehicle.

    Based on historical Saturn V specifications.

    Returns:
        MultiStageVehicle configured like Saturn V
    """
    # S-IC First Stage (5 F-1 engines)
    s_ic_engine = StageEngine(
        thrust_sl=33_400_000,   # 33.4 MN
        thrust_vac=38_700_000,  # 38.7 MN
        isp_sl=263,
        isp_vac=304,
        num_engines=5,
        throttle_min=0.9,
        throttle_max=1.0
    )

    s_ic = Stage(
        name="S-IC",
        dry_mass=130_000,
        propellant_mass=2_160_000,
        engine=s_ic_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION
    )

    # S-II Second Stage (5 J-2 engines)
    s_ii_engine = StageEngine(
        thrust_sl=0,
        thrust_vac=5_141_000,  # 5.141 MN
        isp_sl=0,
        isp_vac=421,
        num_engines=5,
        throttle_min=0.9,
        throttle_max=1.0
    )

    s_ii = Stage(
        name="S-II",
        dry_mass=36_000,
        propellant_mass=454_000,
        engine=s_ii_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
        interstage_mass=3_600
    )

    # S-IVB Third Stage (1 J-2 engine)
    s_ivb_engine = StageEngine(
        thrust_sl=0,
        thrust_vac=1_033_000,  # 1.033 MN
        isp_sl=0,
        isp_vac=421,
        num_engines=1,
        throttle_min=0.9,
        throttle_max=1.0
    )

    s_ivb = Stage(
        name="S-IVB",
        dry_mass=13_500,
        propellant_mass=108_600,
        engine=s_ivb_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
        interstage_mass=3_200
    )

    return MultiStageVehicle(
        name="Saturn V",
        stages=[s_ic, s_ii, s_ivb],
        payload_mass=0,
        launch_site_altitude=0
    )


def create_custom_vehicle(
    stage_configs: list[dict],
    payload_mass: float = 0.0,
    name: str = "Custom Vehicle"
) -> MultiStageVehicle:
    """
    Create a custom multi-stage vehicle from configuration.

    Args:
        stage_configs: List of stage configuration dictionaries
        payload_mass: Payload mass (kg)
        name: Vehicle name

    Each stage config should have:
        - name: str
        - dry_mass: float (kg)
        - propellant_mass: float (kg)
        - thrust_sl: float (N)
        - thrust_vac: float (N)
        - isp_sl: float (s)
        - isp_vac: float (s)
        - num_engines: int (default 1)

    Returns:
        Configured MultiStageVehicle
    """
    stages = []

    for config in stage_configs:
        engine = StageEngine(
            thrust_sl=config.get('thrust_sl', 0),
            thrust_vac=config.get('thrust_vac', config.get('thrust_sl', 1e6)),
            isp_sl=config.get('isp_sl', 280),
            isp_vac=config.get('isp_vac', config.get('isp_sl', 320)),
            num_engines=config.get('num_engines', 1),
            throttle_min=config.get('throttle_min', 0.6),
            throttle_max=config.get('throttle_max', 1.0)
        )

        stage = Stage(
            name=config.get('name', f"Stage-{len(stages)+1}"),
            dry_mass=config.get('dry_mass', 1000),
            propellant_mass=config.get('propellant_mass', 10000),
            engine=engine,
            staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
            fairing_mass=config.get('fairing_mass', 0),
            fairing_jettison_alt=config.get('fairing_jettison_alt', None)
        )
        stages.append(stage)

    return MultiStageVehicle(
        name=name,
        stages=stages,
        payload_mass=payload_mass
    )

