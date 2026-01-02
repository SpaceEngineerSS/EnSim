# Changelog

All notable changes to EnSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Added (Phase 6: Deep Engineering)
- **MOC Supersonic Flow Solver** (`src/core/moc_solver.py`)
  - Prandtl-Meyer function and inverse solver
  - Minimum Length Nozzle (MLN) contour generation
  - Characteristic mesh visualization
  - CSV and VTK export support
  
- **Design Optimizer** (`src/core/optimizer.py`)
  - Scipy-based engine optimization (Nelder-Mead, L-BFGS-B)
  - Monte Carlo reliability analysis with parallel execution
  - Sensitivity analysis for uncertainty quantification
  
- **Plume Visualization** (`src/ui/viz/plume_render.py`)
  - Shock diamond physics modeling
  - Mach-to-color mapping
  - Over/under-expanded plume visualization
  
- **Advanced Engineering UI Tab**
  - MOC nozzle design sub-tab with mesh plot
  - Optimization controls with convergence graph
  - Monte Carlo reliability histograms

### Planned
- Shifting equilibrium (non-frozen flow)
- Bell nozzle geometry (Rao optimum)
- Unit conversion (SI/Imperial toggle)
- Additional propellants (MMH, UDMH, H2O2)

