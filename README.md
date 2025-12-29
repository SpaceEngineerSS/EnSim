# EnSim - Open Source Rocket Engine Simulation Suite

[![Tests](https://github.com/username/ensim/actions/workflows/test.yml/badge.svg)](https://github.com/username/ensim/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/username/ensim)
[![Tests](https://img.shields.io/badge/Tests-64%20passed-brightgreen.svg)](tests/)
[![NASA CEA](https://img.shields.io/badge/Validated-NASA%20CEA%20%C2%B13%25-orange.svg)](docs/VALIDATION.md)

![EnSim Banner](docs/banner.png)

**EnSim** is a desktop application for designing liquid rocket engines. It solves complex thermochemical equilibrium equations using NASA CEA methodology to calculate theoretical performance metrics (Isp, C*, T_comb) and visualizes nozzle flow in 2D and 3D.

## Features

- 🔬 **Scientific Accuracy**: <0.1% error vs NIST reference data
- ⚡ **Real-time Analysis**: Numba JIT-compiled solvers
- 🚀 **6-DOF Simulation**: Full rigid-body flight dynamics with adaptive integration
- 🎲 **Monte Carlo**: Landing dispersion analysis (CEP, 3-sigma ellipse)
- 🎨 **Modern UI**: Dark-themed PyQt6 interface with interactive plots
- 🌡️ **Dissociation Chemistry**: Accurate high-temperature equilibrium (H, O, OH)
- 🎮 **3D Visualization**: PyVista nozzle mesh and rocket attitude display
- 📈 **Dense Output**: Fixed-rate sampling via Cubic Hermite interpolation

## Installation

```bash
# Clone repository
git clone https://github.com/SpaceEngineerSS/ensim.git
cd ensim

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate  
# Linux/Mac:
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Launch GUI
python main.py

# Run quick physics test
python main.py --test

# Run full test suite
pytest tests/ -v
```

## Scientific Validation

EnSim results are validated against NASA CEA and NIST data:

| Property | EnSim | NASA CEA | Error |
|----------|-------|----------|-------|
| T_combustion (H2/O2) | 3600 K | 3516 K | 2.4% |
| Vacuum Isp | 414.6 s | ~420 s | 1.3% |
| Cp H2O @ 1000K | 41.29 J/(mol·K) | 41.29 J/(mol·K) | <0.01% |

### 6-DOF Flight Core
- **Integrator**: RK45 (Dormand-Prince) with adaptive step size.
- **Interpolation**: Cubic Hermite Spline for smooth 0.01s dense output.
- **Orientation**: Quaternion-based (W, X, Y, Z) to avoid gimbal lock.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deep dive into the physics engine and numerical solvers.

```
EnSim/
├── src/
│   ├── core/           # Physics engine (Numba JIT)
│   │   ├── flight_6dof.py    # 6-DOF flight dynamics
│   │   ├── chemistry.py      # Gibbs equilibrium solver
│   │   ├── propulsion.py     # Nozzle flow physics
│   │   ├── math_utils.py     # Quaternion & Vector math
│   │   └── integrators.py    # RK45 & Interpolation
│   ├── ui/             # PyQt6 interface
│   │   ├── workers.py        # QThread background tasks
│   │   └── widgets/          # Flight control & Viz
│   └── utils/          # Data exporters
├── tests/              # Pytest suite
└── data/               # NASA thermo database
```


## Core Physics

### Thermodynamics
- NASA 7-term polynomial coefficients
- Automatic coefficient switching at T_mid (1000K)
- Numba JIT-optimized property calculations

### Chemical Equilibrium
- Gordon & McBride Newton-Raphson method
- G/RT normalization for numerical stability
- Full dissociation species: H, O, OH, H2, O2, H2O

### Propulsion
- Characteristic velocity: C* = √(RT)/Γ
- Exit velocity: Ve = √(2γRT/(γ-1) · [1-(Pe/Pc)^((γ-1)/γ)])
- Thrust coefficient: Cf with pressure term

## Screenshots

### Main Interface
![Main Window](docs/screenshot_main.png)

### 2D Profiles
![Graphs](docs/screenshot_graphs.png)

### 3D Nozzle
![3D View](docs/screenshot_3d.png)

## Requirements

- Python 3.10+
- PyQt6
- NumPy
- Numba
- Matplotlib
- PyVista + pyvistaqt (optional, for 3D)
- pytest (for testing)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE).

## References

1. Gordon, S. & McBride, B.J. (1994). NASA RP-1311: Computer Program for Calculation of Complex Chemical Equilibrium Compositions.
2. Sutton, G.P. & Biblarz, O. (2016). Rocket Propulsion Elements, 9th Edition.
3. NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/)

## Acknowledgments

- NASA Glenn Research Center for thermodynamic data
- The open-source scientific Python community
