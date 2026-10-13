# ensim.core.staging

Multi-stage sizing with the rocket equation. Thrust is given per engine; a
stage's thrust is that times its engine count, scaled between sea level and
vacuum with the US Standard Atmosphere pressure.

::: ensim.core.staging
    options:
      show_root_heading: false
      members:
        - StageEngine
        - Stage
        - MultiStageVehicle
        - StagingEvent
        - StagingTrigger
        - StageStatus
        - create_medium_lift_reference
        - create_heavy_lift_reference
        - create_custom_vehicle
