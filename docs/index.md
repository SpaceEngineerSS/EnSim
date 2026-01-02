# EnSim Documentation

Welcome to **EnSim** - the Professional Rocket Engine & Flight Simulation Platform.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Quick Start__

    ---

    Get up and running with EnSim in under 5 minutes.

    [:octicons-arrow-right-24: Getting Started](getting-started/index.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Learn how to use EnSim's features effectively.

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-math-integral:{ .lg .middle } __Theory__

    ---

    Understand the physics and mathematics behind EnSim.

    [:octicons-arrow-right-24: Theory](theory/index.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Detailed documentation for developers.

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

## What is EnSim?

EnSim is an open-source desktop application for rocket propulsion analysis and flight simulation. It provides:

- **Thermochemical Equilibrium**: NASA CEA methodology for combustion analysis
- **Performance Calculations**: Isp, C*, thrust coefficient, and more
- **6-DOF Flight Simulation**: Full rigid-body trajectory simulation
- **Monte Carlo Analysis**: Statistical dispersion analysis
- **Modern Interface**: User-friendly PyQt6 GUI with 3D visualization

## Key Features

### 🔬 Validated Accuracy

EnSim results are validated against NASA CEA with <2% error across all test cases.

| Property | Max Error | Status |
|----------|-----------|--------|
| Chamber Temperature | 1.76% | ✅ |
| Specific Impulse | 1.41% | ✅ |
| Characteristic Velocity | 0.96% | ✅ |

### 🚀 Comprehensive Analysis

From thermochemistry to trajectory prediction:

```mermaid
graph LR
    A[Propellant Selection] --> B[Combustion Analysis]
    B --> C[Nozzle Design]
    C --> D[Performance Metrics]
    D --> E[Flight Simulation]
    E --> F[Monte Carlo]
```

### ⚡ High Performance

- **Numba JIT Compilation**: 10-100x faster than pure Python
- **Parallel Processing**: Multi-core Monte Carlo simulations
- **Adaptive Integration**: RK45 with automatic step sizing

## Quick Example

```python
from src.core.chemistry import CombustionProblem
from src.utils.nasa_parser import create_sample_database

# Load species database
db = create_sample_database()

# Setup combustion problem
problem = CombustionProblem(db)
problem.add_fuel('H2', moles=2.0)
problem.add_oxidizer('O2', moles=1.0)

# Solve equilibrium
result = problem.solve(pressure=10e6)  # 100 bar

print(f"Chamber Temperature: {result.temperature:.0f} K")
print(f"Gamma: {result.gamma:.3f}")
print(f"Mean MW: {result.mean_molecular_weight:.2f} g/mol")
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4+ cores |
| GPU | Not required | OpenGL 3.3+ for 3D |

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/ensim/ensim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ensim/ensim/discussions)
- **Email**: support@ensim.io

## License

EnSim is released under the [MIT License](https://opensource.org/licenses/MIT).

