# EnSim Architecture

> **Engineering Simulation Core** — A High-Fidelity 6-DOF Rocket Flight Simulator

---

## 1. System Overview

### Mission Statement

EnSim is a professional-grade rocket flight simulation engine designed for aerospace engineering analysis. It provides real-time 6 Degrees of Freedom (6-DOF) trajectory simulation with adaptive numerical integration and Monte Carlo dispersion analysis.

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Core Physics** | Python 3.10+, NumPy, Numba | JIT-compiled numerical kernels |
| **UI Framework** | PyQt6 | Cross-platform desktop application |
| **3D Visualization** | PyVista | Interactive nozzle and plume rendering |
| **Parallel Computing** | multiprocessing, ProcessPoolExecutor | Monte Carlo parallelization |
| **Chemistry** | NASA CEA Method | Chemical equilibrium thermodynamics |

### Key Capabilities

- **14-State Vector Integration**: Position, velocity, quaternion, angular velocity, propellant mass
- **Adaptive RK45 Solver**: Dormand-Prince 5(4) with PI step-size control
- **Dense Output**: Fixed-rate sampling via Cubic Hermite interpolation
- **Flow Separation Physics**: Summerfield criterion for thrust loss modeling
- **Monte Carlo Dispersion**: Parallel CEP and 3-sigma ellipse computation

---

## 2. The Physics Core — "The Engine"

### 2.1 State Vector Architecture (14-DOF)

The simulation uses a 14-element state vector:

```
State = [x, y, z, vx, vy, vz, q0, q1, q2, q3, ωx, ωy, ωz, m_prop]
         ├─────┘  ├───────┘  ├────────────┘  ├────────────┘  └─ Propellant mass
         Position  Velocity   Quaternion      Angular velocity
         (ENU)     (ENU)      (Body→Inertial) (Body frame)
```

**Design Rationale:**

| Component | Choice | Reason |
|-----------|--------|--------|
| Orientation | Quaternion | Avoids gimbal lock at ±90° pitch |
| Coordinate Frame | ENU (East-North-Up) | Matches geodetic convention |
| Angular Velocity | Body Frame | Simplifies moment equations |
| Mass | Integrated State | Enables variable mass dynamics |

### 2.2 Numerical Integration

#### Dormand-Prince 5(4) — Adaptive RK45

The integrator uses embedded error estimation:

```python
# Butcher Tableau (7-stage, 5th order, 4th order error)
k1 = f(t, y)
k2 = f(t + c2*h, y + h*(a21*k1))
...
y_5th = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)  # Solution
y_4th = y + h*(d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6)  # Error estimate
```

**Step Size Control (PI Controller):**

```
err_norm = ||y_5th - y_4th|| / (atol + rtol * |y|)
h_new = h * (0.9 / err_norm^0.2) * (err_prev^0.04)
```

This provides:
- **Safety factor**: 0.9 prevents oscillation
- **I-gain (0.04)**: Smooths step size changes
- **P-gain (0.2)**: Responds to current error

#### Dense Output — Cubic Hermite Interpolation

When the solver takes large steps (h >> dt_output), we interpolate:

```
For t ∈ [t_n, t_n + h]:
  θ = (t - t_n) / h
  y(t) = (1-θ)·y_n + θ·y_{n+1} + θ(θ-1)[(1-2θ)(y_{n+1}-y_n) + (θ-1)h·f_n + θ·h·f_{n+1}]
```

This produces C¹-continuous output at fixed `output_dt` intervals regardless of internal step sizes.

### 2.3 Propulsion Model

#### Thrust Calculation with Altitude Correction

```python
P_exit = P_chamber / (1 + (γ-1)/2 * M_exit²)^(γ/(γ-1))
F_thrust = ṁ·Ve + (P_exit - P_ambient)·A_exit
```

#### Flow Separation Detection (Summerfield Criterion)

```python
separation_ratio = P_exit / P_ambient

if separation_ratio < 0.4:
    flow_regime = SEPARATED
    thrust_loss_factor = 0.5 + 0.5 * (separation_ratio / 0.4)
else:
    flow_regime = ATTACHED
    thrust_loss_factor = 1.0
```

#### 2.4 Shifting Equilibrium Model (Simplified)

EnSim implements a shifting equilibrium model that captures recombination effects:
1. **Base Expansion**: Uses chamber gamma ($\gamma_{ch}$) for isentropic ratios to ensure a fair " frozen versus shifting" comparison.
2. **Recombination**: Extent of atomic recombination (H, O, OH) is estimated based on local $T, P$.
3. **Energy Recovery**: Heat released from recombination is recovered directly into the kinetic energy of the stream: $V = \sqrt{V_{isen}^2 + 2\Delta h_{recomb}}$.

This approach ensures $Isp_{shifting} \geq Isp_{frozen}$ consistently across all expansion ratios.

**Physical Meaning:**
- Pe/Pa < 0.4: Shock wave enters nozzle, flow detaches from wall
- Thrust loss: 10-50% reduction due to asymmetric separation

---

## 3. Simulation Architecture — "The Nervous System"

### 3.1 Thread Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main GUI Thread                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ InputPanel   │  │ GraphWidget  │  │ View3DWidget │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                         ▲                                   │
│                         │ pyqtSignal(FlightResult6DOF)      │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│          FlightSimulationWorker (QThread)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  simulate_flight_6dof(rocket, params...)            │   │
│  │  ├── RK45 Integration Loop                          │   │
│  │  ├── Aerodynamic Force Calculation                  │   │
│  │  ├── Flow Separation Check                          │   │
│  │  └── Dense Output Interpolation                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Signals: log(str), progress(int), finished(result)        │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **Worker Thread**: Heavy computation in QThread prevents UI freeze
- **Signal-Slot**: Type-safe communication between threads
- **Object Copying**: Rocket is deep-copied to worker to prevent race conditions

### 3.2 Monte Carlo Parallel Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    MonteCarloWorker (QThread)                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  ProcessPoolExecutor(max_workers=CPU_COUNT)              │ │
│  │                                                          │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │ │
│  │  │ Core 1  │  │ Core 2  │  │ Core 3  │  │ Core N  │     │ │
│  │  │ Sim #1  │  │ Sim #2  │  │ Sim #3  │  │ Sim #N  │     │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │ │
│  │       │            │            │            │          │ │
│  │       └────────────┴─────┬──────┴────────────┘          │ │
│  │                          ▼                               │ │
│  │               Aggregate Results                          │ │
│  │            ├── compute_cep()                            │ │
│  │            └── compute_confidence_ellipse()             │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Result: DispersionResult(cep, ellipse, landing_points)       │
└────────────────────────────────────────────────────────────────┘
```

**GIL Bypass:**
- `multiprocessing.Process` spawns independent Python interpreters
- Each simulation runs in complete isolation
- Result aggregation happens after all processes complete

---

## 4. Directory Structure

```
EnSim/
├── main.py                      # Application entry point
├── ARCHITECTURE.md              # This document
│
├── src/
│   ├── core/                    # Physics engine (Numba-accelerated)
│   │   ├── flight_6dof.py       # 6-DOF simulator + FlightResult6DOF
│   │   ├── integrators.py       # RK45, Hermite interpolation
│   │   ├── math_utils.py        # Quaternion operations (JIT)
│   │   ├── monte_carlo.py       # Dispersion analysis + CEP
│   │   ├── rocket.py            # Vehicle model dataclasses
│   │   ├── rocket_engine.py     # Flow separation physics
│   │   ├── propulsion.py        # Isentropic flow, nozzle design
│   │   ├── chemistry.py         # NASA CEA equilibrium
│   │   └── aero.py              # Barrowman aerodynamics
│   │
│   ├── ui/                      # PyQt6 interface
│   │   ├── windows/
│   │   │   └── main_window.py   # Application shell
│   │   ├── widgets/
│   │   │   ├── vehicle_widget.py    # Rocket designer + plots
│   │   │   ├── view3d_widget.py     # PyVista 3D view
│   │   │   └── graph_widget.py      # Matplotlib plots
│   │   └── workers.py           # QThread workers
│   │
│   └── utils/                   # I/O and parsing
│       └── nasa_parser.py       # Thermodynamic data loader
│
├── data/
│   └── nasa_thermo.dat          # NASA 7-term polynomial database
│
└── tests/
    ├── unit/                    # pytest unit tests
    └── stress_test_monte_carlo.py
```

---

## 5. Future Roadmap

### 5.1 GNC Integration (Guidance, Navigation, Control)

| Feature | Description | Complexity |
|---------|-------------|------------|
| **PID Attitude Control** | Roll/pitch/yaw rate damping | Medium |
| **Thrust Vector Control** | Gimbal angle simulation | Medium |
| **LQR Optimal Control** | State-feedback trajectory tracking | High |
| **Kalman Filter** | Sensor fusion for state estimation | High |

### 5.2 Environmental Models

| Feature | Description | Reference |
|---------|-------------|-----------|
| **Wind Profile** | Weibull distribution, altitude-dependent | MIL-STD-210C |
| **Atmospheric Turbulence** | Dryden/Von Kármán spectrum | MIL-F-8785C |
| **Earth Rotation** | Coriolis and centrifugal effects | WGS-84 |

### 5.3 Terrain Integration

| Feature | Description |
|---------|-------------|
| **DEM Loading** | SRTM/ASTER elevation data |
| **Ground Collision** | Ray-mesh intersection |
| **Landing Site Analysis** | Slope and hazard detection |

---

## References

1. Stevens, B.L., Lewis, F.L. — *Aircraft Control and Simulation*, 3rd Ed.
2. Zipfel, P.H. — *Modeling and Simulation of Aerospace Vehicle Dynamics*, 3rd Ed.
3. Sutton, G.P. — *Rocket Propulsion Elements*, 9th Ed.
4. Gordon, S., McBride, B.J. — *NASA RP-1311: CEA Computer Program*
5. Dormand, J.R., Prince, P.J. — *A family of embedded Runge-Kutta formulae* (1980)

---

*Document Version: 1.0 | Last Updated: 2024-12-30*
