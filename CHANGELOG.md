# Changelog

All notable changes to EnSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-02

### Added
- **Multi-Stage Vehicle Support** (`src/core/staging.py`)
  - Stage class with engine configuration and mass properties
  - MultiStageVehicle for complete rocket modeling
  - Presets: Falcon 9, Saturn V, custom configurations
  - Delta-V calculations with payload optimization

- **Regenerative Cooling Analysis** (`src/core/cooling.py`)
  - Bartz correlation for gas-side heat transfer
  - Dittus-Boelter coolant-side heat transfer
  - Automatic cooling channel design
  - Wall temperature and heat flux profiles
  - 5 coolant types: RP-1, LH2, LOX, LCH4, Water

- **Trajectory Optimization** (`src/core/optimization.py`)
  - Nozzle expansion ratio optimization
  - Stage mass allocation optimization
  - Mission-weighted performance metrics

- **Materials Database** (`src/core/materials.py`)
  - 10 aerospace materials with full thermal properties
  - Inconel 718, OFHC Copper, GRCop-84, Haynes 230, etc.
  - Melting points, conductivity, service temperatures

- **Mission Analysis** (`src/core/mission.py`)
  - Altitude-dependent performance simulation
  - Optimal operating altitude calculation

- **17 Propellant Presets** (`src/data/propellant_presets.py`)
  - LOX/LH2, LOX/RP-1, LOX/CH4, N2O4/UDMH, etc.
  - Pre-configured O/F ratios and properties

- **Unit Conversion System** (`src/utils/units.py`)
  - SI/Imperial toggle in UI
  - Comprehensive conversion functions

- **New UI Widgets**
  - ThermalAnalysisWidget with heat flux visualization
  - CoolingAnalysisWidget for channel design
  - MultiStageWidget for vehicle configuration
  - OptimizationWidget for trajectory optimization
  - PropellantPresetWidget for quick setup
  - UnitSystemBar for unit toggle

### Changed
- Complete UI redesign with Mission Control dark theme
- Reorganized tabs: Output, Results, Engine, Vehicle, Advanced
- Improved color scheme: cyan accents, green values, orange warnings

### Fixed
- Thermal analysis now uses proper Bartz correlation
- All module imports working correctly

---

## [1.0.2] - 2026-01-02

### Fixed
- Logo and screenshot URLs now use absolute GitHub paths for PyPI display
- All repository URLs corrected to SpaceEngineerSS/EnSim

---

## [1.0.1] - 2026-01-02

### Added
- PyPI package publishing support
- Automated release pipeline with trusted publisher

### Fixed
- Linting errors across all modules
- Updated type hints to modern Python 3.10+ syntax

---

## [1.0.0] - 2026-01-02

### Added
- **Core Physics Engine**
  - NASA 7-term polynomial thermodynamic data parser
  - Gibbs free energy minimization solver (Gordon & McBride method)
  - 1-D isentropic nozzle flow calculations
  - Numba JIT-optimized functions for real-time performance

- **Chemical Species Database**
  - H2, O2, H2O, OH, H, O (core H2/O2 combustion)
  - CH4, CO, CO2, N2, N2O (hydrocarbon and nitrous)
  - N2O4, NO2, NO, N2H4 (storable propellants)
  - RP-1 (kerosene surrogate)

- **GUI Application (PyQt6)**
  - Dark engineering theme
  - KPI dashboard with live updates
  - Input panel with all simulation parameters
  - Efficiency factor inputs (η_c*, η_Cf)
  - Tooltips explaining physics terms
  - Save/Load project (.ensim files)
  - Export to CSV and Markdown report

- **Visualization**
  - Matplotlib 2D plots (P/Pc, T, Mach vs area ratio)
  - PyVista 3D nozzle with temperature coloring
  - Interactive rotation/zoom

- **Documentation**
  - README with installation and usage
  - THEORY.md with mathematical background
  - VALIDATION.md with NASA CEA comparison
  - CITATION.cff for academic citation
  - CONTRIBUTING.md for contributors

### Technical Details
- 64 automated tests (pytest)
- Validated against NASA CEA within 3%
- Numba JIT compilation for <100ms calculation time

## [Unreleased]

### Planned
- Shifting equilibrium (non-frozen flow)
- Bell nozzle geometry (Rao optimum)
- Real gas corrections (Redlich-Kwong EOS)
- Additional propellants (H2O2, MMH variants)

