"""
Multi-stage vehicle bookkeeping: stage masses, thrust, staging events and
ideal (Tsiolkovsky) delta-v.

Engine thrust varies linearly with ambient pressure between the sea-level and
vacuum values, F(h) = F_vac - (F_vac - F_sl) p(h) / p0, which is exact for a
fixed nozzle running full. p(h) comes from the US Standard Atmosphere 1976.

References:
    Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed., ch. 4.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .constants import G0, P_ATM
from .mission import atmosphere_properties


class StageStatus(Enum):
    """Stage operational status."""

    INACTIVE = auto()
    BURNING = auto()
    COASTING = auto()
    SEPARATED = auto()
    EXPENDED = auto()


class StagingTrigger(Enum):
    """Conditions that trigger stage separation."""

    PROPELLANT_DEPLETION = auto()
    TIME_BASED = auto()
    ALTITUDE_BASED = auto()
    VELOCITY_BASED = auto()
    MANUAL = auto()


@dataclass
class StageEngine:
    """Engine cluster of one stage.

    Thrust is **per engine**; cluster values are multiplied by
    ``num_engines``. Vacuum-only engines may leave the sea-level values at zero.

    Attributes:
        thrust_sl_per_engine: Sea-level thrust of one engine, N.
        thrust_vac_per_engine: Vacuum thrust of one engine, N.
        isp_sl: Sea-level specific impulse, s.
        isp_vac: Vacuum specific impulse, s.
        num_engines: Engines in the cluster.
        throttle_min: Deepest throttle setting while running, fraction.
        throttle_max: Highest throttle setting, fraction.
        gimbal_range_deg: Gimbal range, deg.
        ignition_time: Delay between staging and full thrust, s.
        shutdown_time: Thrust tail-off duration, s.
    """

    thrust_sl_per_engine: float
    thrust_vac_per_engine: float
    isp_sl: float
    isp_vac: float
    num_engines: int = 1
    throttle_min: float = 0.6
    throttle_max: float = 1.0
    gimbal_range_deg: float = 5.0
    ignition_time: float = 0.5
    shutdown_time: float = 0.3

    def __post_init__(self) -> None:
        if self.num_engines < 1:
            raise ValueError("A stage needs at least one engine")
        if self.thrust_vac_per_engine <= 0.0 or self.isp_vac <= 0.0:
            raise ValueError("Vacuum thrust and Isp must be positive")
        if self.thrust_sl_per_engine < 0.0 or self.isp_sl < 0.0:
            raise ValueError("Sea-level thrust and Isp cannot be negative")
        if not 0.0 <= self.throttle_min <= self.throttle_max <= 1.0:
            raise ValueError("Throttle limits must satisfy 0 <= min <= max <= 1")

    @property
    def total_thrust_sl(self) -> float:
        """Sea-level thrust of the whole cluster, N."""
        return self.thrust_sl_per_engine * self.num_engines

    @property
    def total_thrust_vac(self) -> float:
        """Vacuum thrust of the whole cluster, N."""
        return self.thrust_vac_per_engine * self.num_engines

    def _throttle(self, throttle: float) -> float:
        if throttle <= 0.0:
            return 0.0
        return float(np.clip(throttle, self.throttle_min, self.throttle_max))

    @staticmethod
    def _pressure_fraction(altitude: float) -> float:
        return atmosphere_properties(max(0.0, altitude))[1] / P_ATM

    def get_thrust(self, altitude: float, throttle: float = 1.0) -> float:
        """Cluster thrust at an altitude and throttle setting, N.

        A throttle of zero or less shuts the engines down; other values are
        clamped to the throttle limits.
        """
        fraction = self._pressure_fraction(altitude)
        per_engine = (
            self.thrust_vac_per_engine
            - (self.thrust_vac_per_engine - self.thrust_sl_per_engine) * fraction
        )
        return max(0.0, per_engine) * self._throttle(throttle) * self.num_engines

    def get_isp(self, altitude: float) -> float:
        """Delivered specific impulse at an altitude, s."""
        fraction = self._pressure_fraction(altitude)
        return self.isp_vac - (self.isp_vac - self.isp_sl) * fraction

    def get_mass_flow_rate(self, altitude: float = 0.0, throttle: float = 1.0) -> float:
        """Cluster propellant mass flow, kg/s.

        ``altitude`` is accepted for symmetry with :meth:`get_thrust`; for a
        nozzle running full the flow does not depend on ambient pressure.
        """
        del altitude
        per_engine = self.thrust_vac_per_engine / (self.isp_vac * G0)
        return per_engine * self._throttle(throttle) * self.num_engines


@dataclass
class Stage:
    """
    One stage of the stack.

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
        return sum(
            1 for s in self.stages if s.status not in [StageStatus.SEPARATED, StageStatus.EXPENDED]
        )

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
        self, dt: float, altitude: float, velocity: float, throttle: float = 1.0
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
        if (
            not stage.fairing_jettisoned
            and stage.fairing_jettison_alt is not None
            and altitude >= stage.fairing_jettison_alt
        ):
            mass_before = self.total_mass
            stage.fairing_jettisoned = True
            mass_after = self.total_mass

            event = StagingEvent(
                time=self.mission_time,
                stage_name=stage.name,
                event_type="fairing_jettison",
                altitude=altitude,
                velocity=velocity,
                mass_before=mass_before,
                mass_after=mass_after,
            )
            events.append(event)
            self.events.append(event)

        # Check staging trigger
        should_stage = self._check_staging_trigger(stage, altitude, velocity)

        if should_stage:
            events.extend(self._execute_staging(altitude, velocity))

        return events

    def _check_staging_trigger(self, stage: Stage, altitude: float, velocity: float) -> bool:
        """Check if staging should occur."""
        if stage.staging_trigger == StagingTrigger.PROPELLANT_DEPLETION:
            return stage.is_depleted()

        elif stage.staging_trigger == StagingTrigger.TIME_BASED:
            return stage.trigger_value is not None and self.mission_time >= stage.trigger_value

        elif stage.staging_trigger == StagingTrigger.ALTITUDE_BASED:
            return stage.trigger_value is not None and altitude >= stage.trigger_value

        elif stage.staging_trigger == StagingTrigger.VELOCITY_BASED:
            return stage.trigger_value is not None and velocity >= stage.trigger_value

        return False

    def _execute_staging(self, altitude: float, velocity: float) -> list[StagingEvent]:
        """Execute stage separation and next stage ignition."""
        events = []
        stage = self.active_stage

        if stage is None:
            return events

        # Record mass before separation
        mass_before = self.total_mass

        # Shutdown current stage
        stage.status = StageStatus.SEPARATED
        events.append(
            StagingEvent(
                time=self.mission_time,
                stage_name=stage.name,
                event_type="separation",
                altitude=altitude,
                velocity=velocity,
                mass_before=mass_before,
                mass_after=self.total_mass,
            )
        )

        # Activate next stage
        self.active_stage_idx += 1
        next_stage = self.active_stage

        if next_stage is not None:
            next_stage.status = StageStatus.BURNING
            events.append(
                StagingEvent(
                    time=self.mission_time + next_stage.engine.ignition_time,
                    stage_name=next_stage.name,
                    event_type="ignition",
                    altitude=altitude,
                    velocity=velocity,
                    mass_before=self.total_mass,
                    mass_after=self.total_mass,
                )
            )

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


def create_medium_lift_reference() -> MultiStageVehicle:
    """Create an editable two-stage reference vehicle."""
    # Nine kerosene gas-generator engines, 845 kN SL / 914 kN vac each
    s1_engine = StageEngine(
        thrust_sl_per_engine=845_000,
        thrust_vac_per_engine=914_000,
        isp_sl=282,
        isp_vac=311,
        num_engines=9,
        throttle_min=0.4,
        throttle_max=1.0,
        gimbal_range_deg=5.0,
    )

    stage1 = Stage(
        name="S1",
        dry_mass=25_600,  # kg
        propellant_mass=411_000,
        engine=s1_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
    )

    s2_engine = StageEngine(
        thrust_sl_per_engine=0,
        thrust_vac_per_engine=934_000,
        isp_sl=0,
        isp_vac=348,
        num_engines=1,
        throttle_min=0.4,
        throttle_max=1.0,
        gimbal_range_deg=6.0,
    )

    stage2 = Stage(
        name="S2",
        dry_mass=4_000,  # kg
        propellant_mass=111_500,  # kg
        engine=s2_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
        fairing_mass=1_900,
        fairing_jettison_alt=110_000,
    )

    return MultiStageVehicle(
        name="Medium-lift reference",
        stages=[stage1, stage2],
        payload_mass=0,
        launch_site_altitude=0,
    )


def create_heavy_lift_reference() -> MultiStageVehicle:
    """Create an editable three-stage reference vehicle."""
    # Five F-1 class engines, 6.77 MN SL / 7.77 MN vac each
    s_ic_engine = StageEngine(
        thrust_sl_per_engine=6_770_000,
        thrust_vac_per_engine=7_770_000,
        isp_sl=263,
        isp_vac=304,
        num_engines=5,
        throttle_min=0.9,
        throttle_max=1.0,
    )

    s_ic = Stage(
        name="Stage 1",
        dry_mass=130_000,
        propellant_mass=2_160_000,
        engine=s_ic_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
    )

    # Five J-2 class engines, 1.03 MN vac each
    s_ii_engine = StageEngine(
        thrust_sl_per_engine=0,
        thrust_vac_per_engine=1_033_000,
        isp_sl=0,
        isp_vac=421,
        num_engines=5,
        throttle_min=0.9,
        throttle_max=1.0,
    )

    s_ii = Stage(
        name="Stage 2",
        dry_mass=36_000,
        propellant_mass=454_000,
        engine=s_ii_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
        interstage_mass=3_600,
    )

    s_ivb_engine = StageEngine(
        thrust_sl_per_engine=0,
        thrust_vac_per_engine=1_033_000,
        isp_sl=0,
        isp_vac=421,
        num_engines=1,
        throttle_min=0.9,
        throttle_max=1.0,
    )

    s_ivb = Stage(
        name="Stage 3",
        dry_mass=13_500,
        propellant_mass=108_600,
        engine=s_ivb_engine,
        staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
        interstage_mass=3_200,
    )

    return MultiStageVehicle(
        name="Heavy-lift reference",
        stages=[s_ic, s_ii, s_ivb],
        payload_mass=0,
        launch_site_altitude=0,
    )


def create_custom_vehicle(
    stage_configs: list[dict], payload_mass: float = 0.0, name: str = "Custom Vehicle"
) -> MultiStageVehicle:
    """Build a vehicle from plain dictionaries, bottom stage first.

    Keys per stage: ``name``, ``dry_mass`` (kg), ``propellant_mass`` (kg),
    ``thrust_sl_per_engine`` and ``thrust_vac_per_engine`` (N), ``isp_sl`` and
    ``isp_vac`` (s), ``num_engines``, ``throttle_min``, ``throttle_max``,
    ``fairing_mass`` (kg) and ``fairing_jettison_alt`` (m). Masses, vacuum
    thrust and vacuum Isp are required.
    """
    stages = []
    for index, config in enumerate(stage_configs, start=1):
        required = ("dry_mass", "propellant_mass", "thrust_vac_per_engine", "isp_vac")
        missing = [key for key in required if key not in config]
        if missing:
            raise KeyError(f"Stage {index} is missing {', '.join(missing)}")
        engine = StageEngine(
            thrust_sl_per_engine=config.get("thrust_sl_per_engine", 0.0),
            thrust_vac_per_engine=config["thrust_vac_per_engine"],
            isp_sl=config.get("isp_sl", 0.0),
            isp_vac=config["isp_vac"],
            num_engines=config.get("num_engines", 1),
            throttle_min=config.get("throttle_min", 0.6),
            throttle_max=config.get("throttle_max", 1.0),
        )
        stages.append(
            Stage(
                name=config.get("name", f"Stage {index}"),
                dry_mass=config["dry_mass"],
                propellant_mass=config["propellant_mass"],
                engine=engine,
                staging_trigger=StagingTrigger.PROPELLANT_DEPLETION,
                fairing_mass=config.get("fairing_mass", 0.0),
                fairing_jettison_alt=config.get("fairing_jettison_alt"),
            )
        )
    return MultiStageVehicle(name=name, stages=stages, payload_mass=payload_mass)
