# Getting Started

Welcome to EnSim! This guide will help you get up and running quickly.

## Overview

EnSim is a professional rocket engine simulation platform that provides:

- **Thermochemical Analysis**: Calculate combustion properties using NASA CEA methodology
- **Performance Metrics**: Specific impulse, characteristic velocity, thrust coefficient
- **Flight Simulation**: Full 6-DOF trajectory analysis
- **Monte Carlo**: Statistical dispersion analysis

## Installation Steps

### 1. Prerequisites

Before installing EnSim, ensure you have:

- **Python 3.10 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** - Usually included with Python
- **Git** (optional) - For cloning the repository

### 2. Clone Repository

```bash
git clone https://github.com/ensim/ensim.git
cd ensim
```

Or download the [latest release](https://github.com/ensim/ensim/releases/latest).

### 3. Create Virtual Environment

We recommend using a virtual environment:

=== "Windows"

    ```powershell
    python -m venv venv
    venv\Scripts\activate
    ```

=== "Linux/macOS"

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python main.py --test
```

You should see validation tests pass successfully.

## Quick Start

Launch EnSim:

```bash
python main.py
```

The main window will open with the Mission Control interface.

## Next Steps

- [Quick Start Guide](quickstart.md) - Your first simulation in 5 minutes
- [First Simulation](first-simulation.md) - Detailed walkthrough
- [User Guide](../user-guide/index.md) - Comprehensive feature documentation

## Troubleshooting

### Common Issues

??? question "PyQt6 import error on Linux"

    Install Qt dependencies:
    ```bash
    sudo apt-get install libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0
    ```

??? question "Numba compilation warning"

    First run may show JIT compilation messages. This is normal and improves performance on subsequent runs.

??? question "3D visualization not working"

    Ensure PyVista is installed:
    ```bash
    pip install pyvista pyvistaqt
    ```

