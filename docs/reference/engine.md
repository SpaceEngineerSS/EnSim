# ensim.core.engine

The entry point for engine analysis: an operating point in, chamber state and
performance out. See the [Python API](../library/python.md) page for examples
and [nozzle expansion](../theory/nozzle.md) for the model.

::: ensim.core.engine
    options:
      show_root_heading: false
      members:
        - EngineInputs
        - EngineResult
        - analyze_engine

## Building blocks

::: ensim.core.frozen_flow
    options:
      heading_level: 3
      members:
        - FrozenStation
        - FrozenNozzle
        - frozen_nozzle
        - frozen_profile

::: ensim.core.geometry
    options:
      heading_level: 3
      members:
        - ChamberContour
        - chamber_contour
        - generate_bell_contour

::: ensim.core.analysis
    options:
      heading_level: 3
      members:
        - EngineSweep
        - sweep_engine
