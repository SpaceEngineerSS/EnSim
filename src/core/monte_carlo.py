"""
Monte Carlo Dispersion Analysis for Rocket Flight Simulation.

Performs stochastic simulations to compute:
- Landing dispersion (CEP, 3-sigma ellipse)
- Performance variability (apogee, velocity, burn time)
- Sensitivity to parameter uncertainties

Uses Python multiprocessing for parallel execution on all CPU cores.

References:
    - NASA-HDBK-1001: Atmospheric Models for Aerospace Applications
    - MIL-STD-1316: Test Methods for Rocket Motors
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# =============================================================================
# Configuration & Result Dataclasses
# =============================================================================

@dataclass
class DispersionConfig:
    """
    Configuration for Monte Carlo dispersion analysis.

    All variation parameters use Gaussian distribution unless noted.
    Values are standard deviations (1-sigma) unless noted.
    """
    # Number of simulations
    num_simulations: int = 100

    # Thrust variation (multiplicative factor)
    # thrust_actual = thrust_nominal * (1 + N(0, thrust_sigma))
    thrust_sigma: float = 0.02  # 2% standard deviation

    # ISP variation (multiplicative factor)
    isp_sigma: float = 0.01  # 1% standard deviation

    # Burn time variation (multiplicative factor)
    burn_time_sigma: float = 0.01  # 1% standard deviation

    # Drag coefficient variation (multiplicative factor)
    cd_sigma: float = 0.05  # 5% standard deviation

    # Fin misalignment (degrees, uniform distribution)
    fin_misalignment_max: float = 0.5  # ±0.5 degrees

    # Launch angle variation (degrees)
    launch_angle_sigma: float = 0.5  # 0.5 degree standard deviation

    # Wind speed (m/s, Gaussian from mean)
    wind_speed_mean: float = 3.0
    wind_speed_sigma: float = 2.0

    # Wind direction (degrees, uniform 0-360 if randomize_wind_dir=True)
    wind_direction_mean: float = 0.0
    wind_direction_sigma: float = 30.0
    randomize_wind_dir: bool = True

    # Random seed (None for random, int for reproducibility)
    seed: int | None = None

    # Number of parallel workers (None = CPU count)
    n_workers: int | None = None


@dataclass
class SimulationResult:
    """Result from a single Monte Carlo simulation run."""
    run_id: int

    # Landing coordinates (ENU frame)
    landing_x: float  # East (m)
    landing_y: float  # North (m)

    # Key performance metrics
    apogee: float           # Maximum altitude (m)
    max_velocity: float     # Maximum speed (m/s)
    max_mach: float         # Maximum Mach number
    burnout_altitude: float # Altitude at burnout (m)
    flight_time: float      # Total flight time (s)

    # Applied perturbations (for analysis)
    thrust_factor: float
    isp_factor: float
    cd_factor: float
    wind_speed: float
    wind_direction: float

    # Success flag
    success: bool = True
    error_message: str = ""


@dataclass
class DispersionResult:
    """
    Complete Monte Carlo dispersion analysis results.

    Contains statistical summaries and raw data for visualization.
    """
    # Configuration used
    config: DispersionConfig

    # Timing
    total_time_seconds: float
    simulations_per_second: float

    # Landing statistics
    landing_x_mean: float
    landing_y_mean: float
    landing_x_std: float
    landing_y_std: float

    # CEP (Circular Error Probable) - radius containing 50% of landings
    cep_radius: float

    # 3-sigma ellipse parameters
    ellipse_semi_major: float  # 3-sigma major axis
    ellipse_semi_minor: float  # 3-sigma minor axis
    ellipse_angle_deg: float   # Rotation angle from East

    # Performance statistics
    apogee_mean: float
    apogee_std: float
    max_velocity_mean: float
    max_velocity_std: float
    flight_time_mean: float
    flight_time_std: float

    # Raw data for plotting
    landing_points: np.ndarray  # (N, 2) array of [x, y]
    apogees: np.ndarray         # (N,) array of apogee values

    # Individual results
    results: list[SimulationResult] = field(default_factory=list)

    # Failure count
    n_successful: int = 0
    n_failed: int = 0

    # UI-compatible aliases
    @property
    def num_simulations(self) -> int:
        """Total number of simulations run."""
        return self.n_successful + self.n_failed

    @property
    def mean_apogee(self) -> float:
        """Alias for apogee_mean for UI compatibility."""
        return self.apogee_mean

    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        total = self.n_successful + self.n_failed
        return self.n_successful / total if total > 0 else 0.0

    @property
    def ellipse_major(self) -> float:
        """Alias for ellipse_semi_major for UI."""
        return self.ellipse_semi_major

    @property
    def ellipse_minor(self) -> float:
        """Alias for ellipse_semi_minor for UI."""
        return self.ellipse_semi_minor


# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def compute_cep(points: np.ndarray, center: np.ndarray) -> float:
    """
    Compute Circular Error Probable (CEP).

    CEP is the radius of a circle centered at the mean landing point
    that contains 50% of all landing points.

    Args:
        points: (N, 2) array of landing coordinates
        center: (2,) array of center point (usually mean)

    Returns:
        CEP radius in same units as input
    """
    # Compute distances from center
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))

    # CEP is the 50th percentile distance
    return np.percentile(distances, 50)


def compute_confidence_ellipse(
    points: np.ndarray,
    confidence: float = 0.997  # 3-sigma = 99.7%
) -> tuple[float, float, float]:
    """
    Compute confidence ellipse parameters using PCA.

    Args:
        points: (N, 2) array of landing coordinates
        confidence: Confidence level (0.997 for 3-sigma)

    Returns:
        Tuple of (semi_major, semi_minor, angle_degrees)
    """
    # Center the data
    centered = points - np.mean(points, axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Chi-squared value for confidence level (2 DOF)
    # For 99.7% (3-sigma): chi2 ≈ 11.83
    # For 95%: chi2 ≈ 5.991
    # For 50%: chi2 ≈ 1.386
    from scipy.stats import chi2
    chi2_val = chi2.ppf(confidence, df=2)

    # Semi-axes lengths
    semi_major = np.sqrt(chi2_val * eigenvalues[0])
    semi_minor = np.sqrt(chi2_val * eigenvalues[1])

    # Angle of major axis from x-axis (East)
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.degrees(angle_rad)

    return semi_major, semi_minor, angle_deg


# =============================================================================
# Worker Function (runs in separate process)
# =============================================================================

def _run_single_simulation(args: tuple) -> SimulationResult:
    """
    Run a single simulation with randomized parameters.

    This function runs in a separate process via multiprocessing.

    Args:
        args: Tuple of (run_id, rocket_dict, base_params, perturbations, rng_seed)

    Returns:
        SimulationResult with outcomes and applied perturbations
    """
    run_id, rocket_dict, base_params, perturbations, rng_seed = args

    # Import here to avoid pickling issues
    from src.core.flight_6dof import simulate_flight_6dof
    from src.core.rocket import BodyTube, EngineMount, Fin, FinSet, NoseCone, Rocket

    # Set random seed for this worker
    np.random.seed(rng_seed)

    try:
        # Reconstruct Rocket from dict
        rocket = Rocket(
            name=rocket_dict['name'],
            nose=NoseCone(**rocket_dict['nose']),
            body=BodyTube(**rocket_dict['body']),
            fins=FinSet(
                fin=Fin(**rocket_dict['fins']['fin']),
                count=rocket_dict['fins']['count'],
                position=rocket_dict['fins']['position'],
                mass=rocket_dict['fins']['mass'],
                material=rocket_dict['fins']['material']
            ),
            engine=EngineMount(**rocket_dict['engine'])
        )

        # Apply perturbations
        thrust = base_params['thrust_vac'] * perturbations['thrust_factor']
        isp = base_params['isp_vac'] * perturbations['isp_factor']
        burn_time = base_params['burn_time'] * perturbations['burn_time_factor']

        # Run simulation with perturbations
        result = simulate_flight_6dof(
            rocket=rocket,
            thrust_vac=thrust,
            isp_vac=isp,
            burn_time=burn_time,
            exit_area=base_params.get('exit_area', 0.01),
            dt=base_params.get('dt', 0.02),
            max_time=base_params.get('max_time', 300.0),
            launch_angle_deg=base_params['launch_angle_deg'] + perturbations['launch_angle_offset'],
            launch_azimuth_deg=base_params.get('launch_azimuth_deg', 0.0),
            wind_speed=perturbations['wind_speed'],
            wind_direction_deg=perturbations['wind_direction'],
            use_adaptive=base_params.get('use_adaptive', False),
            # Monte Carlo perturbations
            cd_factor=perturbations['cd_factor'],
            fin_misalignment_deg=perturbations['fin_misalignment']
        )

        # Extract landing point (last position)
        landing_x = result.position_x[-1]
        landing_y = result.position_y[-1]

        return SimulationResult(
            run_id=run_id,
            landing_x=landing_x,
            landing_y=landing_y,
            apogee=result.apogee_altitude,
            max_velocity=result.max_velocity,
            max_mach=result.max_mach,
            burnout_altitude=result.burnout_altitude,
            flight_time=result.time[-1],
            thrust_factor=perturbations['thrust_factor'],
            isp_factor=perturbations['isp_factor'],
            cd_factor=perturbations['cd_factor'],
            wind_speed=perturbations['wind_speed'],
            wind_direction=perturbations['wind_direction'],
            success=True
        )

    except Exception as e:
        return SimulationResult(
            run_id=run_id,
            landing_x=0.0,
            landing_y=0.0,
            apogee=0.0,
            max_velocity=0.0,
            max_mach=0.0,
            burnout_altitude=0.0,
            flight_time=0.0,
            thrust_factor=perturbations['thrust_factor'],
            isp_factor=perturbations['isp_factor'],
            cd_factor=perturbations['cd_factor'],
            wind_speed=perturbations['wind_speed'],
            wind_direction=perturbations['wind_direction'],
            success=False,
            error_message=str(e)
        )


def _generate_perturbations(config: DispersionConfig, rng: np.random.Generator) -> dict[str, float]:
    """Generate random perturbations based on config."""

    # Thrust factor (Gaussian)
    thrust_factor = 1.0 + rng.normal(0, config.thrust_sigma)

    # ISP factor (Gaussian)
    isp_factor = 1.0 + rng.normal(0, config.isp_sigma)

    # Burn time factor (Gaussian)
    burn_time_factor = 1.0 + rng.normal(0, config.burn_time_sigma)

    # Drag factor (Gaussian)
    cd_factor = 1.0 + rng.normal(0, config.cd_sigma)

    # Launch angle offset (Gaussian)
    launch_angle_offset = rng.normal(0, config.launch_angle_sigma)

    # Fin misalignment (Uniform)
    fin_misalignment = rng.uniform(-config.fin_misalignment_max, config.fin_misalignment_max)

    # Wind speed (Gaussian, clamped to non-negative)
    wind_speed = max(0, rng.normal(config.wind_speed_mean, config.wind_speed_sigma))

    # Wind direction
    if config.randomize_wind_dir:
        wind_direction = rng.uniform(0, 360)
    else:
        wind_direction = config.wind_direction_mean + rng.normal(0, config.wind_direction_sigma)

    return {
        'thrust_factor': thrust_factor,
        'isp_factor': isp_factor,
        'burn_time_factor': burn_time_factor,
        'cd_factor': cd_factor,
        'launch_angle_offset': launch_angle_offset,
        'fin_misalignment': fin_misalignment,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
    }


def _rocket_to_dict(rocket) -> dict[str, Any]:
    """Convert Rocket object to serializable dict for multiprocessing."""
    return {
        'name': rocket.name,
        'nose': {
            'shape': rocket.nose.shape,
            'length': rocket.nose.length,
            'diameter': rocket.nose.diameter,
            'mass': rocket.nose.mass,
            'material': rocket.nose.material,
        },
        'body': {
            'length': rocket.body.length,
            'diameter': rocket.body.diameter,
            'wall_thickness': rocket.body.wall_thickness,
            'mass': rocket.body.mass,
            'material': rocket.body.material,
        },
        'fins': {
            'fin': {
                'root_chord': rocket.fins.fin.root_chord,
                'tip_chord': rocket.fins.fin.tip_chord,
                'span': rocket.fins.fin.span,
                'sweep_angle': rocket.fins.fin.sweep_angle,
                'thickness': rocket.fins.fin.thickness,
            },
            'count': rocket.fins.count,
            'position': rocket.fins.position,
            'mass': rocket.fins.mass,
            'material': rocket.fins.material,
        },
        'engine': {
            'engine_mass_dry': rocket.engine.engine_mass_dry,
            'fuel_mass': rocket.engine.fuel_mass,
            'oxidizer_mass': rocket.engine.oxidizer_mass,
            'tank_length': rocket.engine.tank_length,
            'position': rocket.engine.position,
            'thrust_vac': rocket.engine.thrust_vac,
            'isp_vac': rocket.engine.isp_vac,
            'mass_flow_rate': rocket.engine.mass_flow_rate,
            'burn_time': rocket.engine.burn_time,
        }
    }


# =============================================================================
# Main Monte Carlo Function
# =============================================================================

def run_monte_carlo(
    rocket,
    thrust_vac: float,
    isp_vac: float,
    burn_time: float,
    config: DispersionConfig | None = None,
    launch_angle_deg: float = 85.0,
    launch_azimuth_deg: float = 0.0,
    exit_area: float = 0.01,
    dt: float = 0.02,
    max_time: float = 300.0,
    use_adaptive: bool = False,
    verbose: bool = True
) -> DispersionResult:
    """
    Run Monte Carlo dispersion analysis with parallel execution.

    Performs multiple simulations with randomized parameters to compute
    landing dispersion statistics (CEP, 3-sigma ellipse).

    Args:
        rocket: Rocket object (will be copied for each simulation)
        thrust_vac: Nominal vacuum thrust (N)
        isp_vac: Nominal vacuum ISP (s)
        burn_time: Nominal burn time (s)
        config: DispersionConfig with variation parameters
        launch_angle_deg: Nominal launch angle from vertical
        launch_azimuth_deg: Nominal launch azimuth from North
        exit_area: Nozzle exit area (m²)
        dt: Time step for fixed-step integration
        max_time: Maximum simulation time (s)
        use_adaptive: Use RK45 adaptive stepping
        verbose: Print progress messages

    Returns:
        DispersionResult with statistics and raw data
    """
    if config is None:
        config = DispersionConfig()

    # Set up random number generator
    rng = np.random.default_rng(config.seed) if config.seed is not None else np.random.default_rng()

    # Prepare base parameters
    base_params = {
        'thrust_vac': thrust_vac,
        'isp_vac': isp_vac,
        'burn_time': burn_time,
        'exit_area': exit_area,
        'dt': dt,
        'max_time': max_time,
        'launch_angle_deg': launch_angle_deg,
        'launch_azimuth_deg': launch_azimuth_deg,
        'use_adaptive': use_adaptive,
    }

    # Convert rocket to dict for pickling
    rocket_dict = _rocket_to_dict(rocket)

    # Generate all perturbations upfront
    perturbations_list = []
    seeds = []
    for _i in range(config.num_simulations):
        pert = _generate_perturbations(config, rng)
        perturbations_list.append(pert)
        seeds.append(rng.integers(0, 2**31))

    # Prepare arguments for workers
    args_list = [
        (i, rocket_dict, base_params, perturbations_list[i], seeds[i])
        for i in range(config.num_simulations)
    ]

    # Determine number of workers
    n_workers = config.n_workers or multiprocessing.cpu_count()

    if verbose:
        print(f"Starting Monte Carlo analysis: {config.num_simulations} simulations")
        print(f"Using {n_workers} parallel workers")

    # Run simulations in parallel
    start_time = time.time()
    results: list[SimulationResult] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_simulation, args): args[0]
                   for args in args_list}

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if verbose and completed % max(1, config.num_simulations // 10) == 0:
                print(f"  Progress: {completed}/{config.num_simulations}")

    end_time = time.time()
    total_time = end_time - start_time

    # Sort results by run_id
    results.sort(key=lambda r: r.run_id)

    # Separate successful and failed
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    if verbose:
        print(f"Completed: {len(successful_results)} successful, {len(failed_results)} failed")
        print(f"Total time: {total_time:.1f}s ({len(results)/total_time:.1f} sims/sec)")

    # Extract landing points from successful results
    if len(successful_results) < 2:
        raise RuntimeError("Not enough successful simulations for analysis")

    landing_points = np.array([
        [r.landing_x, r.landing_y] for r in successful_results
    ])

    apogees = np.array([r.apogee for r in successful_results])
    max_velocities = np.array([r.max_velocity for r in successful_results])
    flight_times = np.array([r.flight_time for r in successful_results])

    # Compute statistics
    landing_mean = np.mean(landing_points, axis=0)
    landing_std = np.std(landing_points, axis=0)

    # CEP
    cep = compute_cep(landing_points, landing_mean)

    # 3-sigma ellipse
    try:
        semi_major, semi_minor, angle = compute_confidence_ellipse(landing_points, 0.997)
    except Exception:
        # Fallback if scipy not available
        semi_major = 3 * landing_std[0]
        semi_minor = 3 * landing_std[1]
        angle = 0.0

    return DispersionResult(
        config=config,
        total_time_seconds=total_time,
        simulations_per_second=len(results) / total_time,
        landing_x_mean=landing_mean[0],
        landing_y_mean=landing_mean[1],
        landing_x_std=landing_std[0],
        landing_y_std=landing_std[1],
        cep_radius=cep,
        ellipse_semi_major=semi_major,
        ellipse_semi_minor=semi_minor,
        ellipse_angle_deg=angle,
        apogee_mean=np.mean(apogees),
        apogee_std=np.std(apogees),
        max_velocity_mean=np.mean(max_velocities),
        max_velocity_std=np.std(max_velocities),
        flight_time_mean=np.mean(flight_times),
        flight_time_std=np.std(flight_times),
        landing_points=landing_points,
        apogees=apogees,
        results=results,
        n_successful=len(successful_results),
        n_failed=len(failed_results)
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_dispersion(result: DispersionResult, save_path: str | None = None) -> None:
    """
    Plot landing dispersion with CEP circle and confidence ellipse.

    Args:
        result: DispersionResult from run_monte_carlo
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === Left plot: Landing dispersion ===
    ax1 = axes[0]

    # Plot landing points
    ax1.scatter(
        result.landing_points[:, 0],
        result.landing_points[:, 1],
        alpha=0.5, s=10, c='blue', label='Landing Points'
    )

    # Plot mean landing point
    ax1.scatter(
        result.landing_x_mean, result.landing_y_mean,
        s=100, c='red', marker='x', linewidths=2,
        label=f'Mean ({result.landing_x_mean:.1f}, {result.landing_y_mean:.1f})'
    )

    # CEP circle
    cep_circle = Circle(
        (result.landing_x_mean, result.landing_y_mean),
        result.cep_radius,
        fill=False, color='green', linewidth=2, linestyle='--',
        label=f'CEP = {result.cep_radius:.1f} m'
    )
    ax1.add_patch(cep_circle)

    # 3-sigma ellipse
    ellipse = Ellipse(
        (result.landing_x_mean, result.landing_y_mean),
        2 * result.ellipse_semi_major,
        2 * result.ellipse_semi_minor,
        angle=result.ellipse_angle_deg,
        fill=False, color='red', linewidth=2,
        label=f'3σ Ellipse ({result.ellipse_semi_major:.1f}×{result.ellipse_semi_minor:.1f} m)'
    )
    ax1.add_patch(ellipse)

    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title(f'Landing Dispersion (N={result.n_successful})')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # === Right plot: Apogee histogram ===
    ax2 = axes[1]

    ax2.hist(result.apogees, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(
        result.apogee_mean, color='red', linewidth=2,
        label=f'Mean = {result.apogee_mean:.1f} ± {result.apogee_std:.1f} m'
    )
    ax2.axvline(
        result.apogee_mean - result.apogee_std, color='orange',
        linewidth=1, linestyle='--'
    )
    ax2.axvline(
        result.apogee_mean + result.apogee_std, color='orange',
        linewidth=1, linestyle='--', label='±1σ'
    )

    ax2.set_xlabel('Apogee (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Apogee Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

    plt.show()


def print_dispersion_summary(result: DispersionResult) -> None:
    """Print formatted summary of dispersion analysis results."""
    print("\n" + "=" * 60)
    print("MONTE CARLO DISPERSION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Simulations: {result.n_successful} successful / {result.n_failed} failed")
    print(f"Execution time: {result.total_time_seconds:.1f}s ({result.simulations_per_second:.1f} sims/sec)")
    print()
    print("LANDING DISPERSION:")
    print(f"  Mean landing: ({result.landing_x_mean:.1f}, {result.landing_y_mean:.1f}) m")
    print(f"  Std dev:      ({result.landing_x_std:.1f}, {result.landing_y_std:.1f}) m")
    print(f"  CEP (50%):    {result.cep_radius:.1f} m")
    print(f"  3σ Ellipse:   {result.ellipse_semi_major:.1f} × {result.ellipse_semi_minor:.1f} m @ {result.ellipse_angle_deg:.1f}°")
    print()
    print("PERFORMANCE:")
    print(f"  Apogee:       {result.apogee_mean:.1f} ± {result.apogee_std:.1f} m")
    print(f"  Max velocity: {result.max_velocity_mean:.1f} ± {result.max_velocity_std:.1f} m/s")
    print(f"  Flight time:  {result.flight_time_mean:.1f} ± {result.flight_time_std:.1f} s")
    print("=" * 60)
