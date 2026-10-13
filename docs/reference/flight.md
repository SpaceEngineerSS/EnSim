# ensim.core.flight_6dof

Six-degree-of-freedom flight of a [`Rocket`](#ensim.core.rocket.Rocket). The
model is described under [flight dynamics](../theory/flight.md).

::: ensim.core.flight_6dof
    options:
      show_root_heading: false
      members:
        - simulate_flight_6dof
        - FlightResult6DOF
        - PhysicsViolationError
        - SimulationCancelled

## Vehicle

::: ensim.core.rocket
    options:
      heading_level: 3
      members:
        - Rocket
        - NoseShape
        - NoseCone
        - BodyTube
        - FinSet
        - EngineMount
        - create_default_rocket

## Atmosphere

::: ensim.core.mission
    options:
      heading_level: 3
      members:
        - atmosphere_properties
        - get_atmosphere
        - AtmosphereState
        - geopotential_altitude
