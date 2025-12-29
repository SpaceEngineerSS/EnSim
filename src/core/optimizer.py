"""
Advanced Analysis Suite: Optimization and Monte Carlo Reliability.

Provides:
- Design optimization using scipy.optimize
- Monte Carlo reliability analysis with parallel execution
- Uncertainty quantification for rocket engine performance

References:
    - Nocedal, J. & Wright, S. "Numerical Optimization", 2nd ed.
    - NASA SP-8039 "Solid Rocket Motor Igniters" (uncertainty methods)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import optimize
import time


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OptimizationBounds:
    """Bounds for optimization variables."""
    pc_range: Tuple[float, float] = (1e6, 20e6)  # Chamber pressure (Pa)
    of_range: Tuple[float, float] = (1.0, 10.0)  # O/F ratio
    epsilon_range: Tuple[float, float] = (5.0, 100.0)  # Expansion ratio
    
    def to_scipy_bounds(self) -> optimize.Bounds:
        """Convert to scipy Bounds object."""
        return optimize.Bounds(
            lb=[self.pc_range[0], self.of_range[0], self.epsilon_range[0]],
            ub=[self.pc_range[1], self.of_range[1], self.epsilon_range[1]]
        )


@dataclass
class OptimizationResult:
    """Result from design optimization."""
    optimal_params: Dict[str, float]
    objective_value: float
    iterations: int
    success: bool
    message: str
    
    # History for convergence plots
    history: List[Dict[str, float]] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (f"OptimizationResult(Isp={self.objective_value:.1f}s, "
                f"Pc={self.optimal_params.get('Pc', 0)/1e5:.1f}bar, "
                f"ε={self.optimal_params.get('epsilon', 0):.1f})")


@dataclass
class MonteCarloInput:
    """Input parameters for Monte Carlo simulation."""
    # Nominal values
    Pc_nominal: float = 10e6  # Pa
    At_nominal: float = 0.01  # m²
    OF_nominal: float = 6.0
    gamma: float = 1.2
    T_chamber: float = 3500.0  # K
    mean_mw: float = 18.0  # g/mol
    epsilon: float = 50.0  # Expansion ratio
    
    # Uncertainties (as fraction, e.g., 0.02 = 2%)
    Pc_sigma: float = 0.02
    At_sigma: float = 0.01
    OF_sigma: float = 0.03
    gamma_sigma: float = 0.01


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo reliability analysis."""
    # Distributions
    thrust_distribution: np.ndarray
    isp_distribution: np.ndarray
    cstar_distribution: np.ndarray
    
    # Statistics
    n_samples: int
    runtime_seconds: float
    
    # Thrust statistics
    thrust_mean: float
    thrust_std: float
    thrust_p95: float  # 95th percentile
    thrust_p99: float  # 99th percentile
    
    # Isp statistics
    isp_mean: float
    isp_std: float
    isp_p95: float
    isp_p99: float
    
    # Reliability
    reliability: float = 0.0  # P(Thrust > threshold)
    threshold: float = 0.0  # Threshold used
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for key metrics."""
        alpha = (1 - confidence) / 2
        
        return {
            'thrust': (
                np.percentile(self.thrust_distribution, alpha * 100),
                np.percentile(self.thrust_distribution, (1 - alpha) * 100)
            ),
            'isp': (
                np.percentile(self.isp_distribution, alpha * 100),
                np.percentile(self.isp_distribution, (1 - alpha) * 100)
            ),
            'cstar': (
                np.percentile(self.cstar_distribution, alpha * 100),
                np.percentile(self.cstar_distribution, (1 - alpha) * 100)
            )
        }


# =============================================================================
# Core Performance Function (for optimization/MC)
# =============================================================================

def evaluate_engine_performance(
    Pc: float,
    epsilon: float,
    gamma: float,
    T_chamber: float,
    mean_mw: float,
    At: float = 0.01,
    Pa: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate rocket engine performance.
    
    This is a simplified version that doesn't require the full simulation.
    Used for optimization and Monte Carlo where speed is critical.
    
    Args:
        Pc: Chamber pressure (Pa)
        epsilon: Expansion ratio
        gamma: Ratio of specific heats
        T_chamber: Chamber temperature (K)
        mean_mw: Mean molecular weight (g/mol)
        At: Throat area (m²)
        Pa: Ambient pressure (Pa)
        
    Returns:
        Dictionary with thrust, isp, c_star, etc.
    """
    # Constants
    R = 8314.46  # J/(kmol·K)
    g0 = 9.80665  # m/s²
    
    # Specific gas constant
    R_specific = R / mean_mw  # J/(kg·K)
    
    # Characteristic velocity
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    
    term = (2.0 / gp1) ** (gp1 / (2.0 * gm1))
    capital_gamma = np.sqrt(gamma) * term
    c_star = np.sqrt(R_specific * T_chamber) / capital_gamma
    
    # Exit Mach number from area ratio (Newton-Raphson)
    M_exit = 2.0  # Initial guess
    for _ in range(20):
        term1 = 2.0 / gp1
        term2 = 1.0 + gm1 / 2.0 * M_exit * M_exit
        f = (1.0 / M_exit) * (term1 * term2) ** (gp1 / (2.0 * gm1)) - epsilon
        
        df = -(term1 * term2) ** (gp1 / (2.0 * gm1)) / (M_exit * M_exit) + \
             (gp1 / (2.0 * gm1)) * gm1 * (term1 * term2) ** (gp1 / (2.0 * gm1) - 1) * term1 / M_exit
        
        if abs(df) < 1e-12:
            break
        
        dM = -f / df
        if abs(dM) > 0.5:
            dM = 0.5 * np.sign(dM)
        
        M_exit = max(1.001, M_exit + dM)
        
        if abs(f) < 1e-8:
            break
    
    # Exit conditions
    T_ratio = 1.0 / (1.0 + gm1 / 2.0 * M_exit ** 2)
    P_ratio = T_ratio ** (gamma / gm1)
    
    T_exit = T_chamber * T_ratio
    P_exit = Pc * P_ratio
    
    # Exit velocity
    pressure_ratio = P_exit / Pc
    V_exit = np.sqrt(2.0 * gamma / gm1 * R_specific * T_chamber * 
                     (1.0 - pressure_ratio ** (gm1 / gamma)))
    
    # Thrust coefficient
    term1 = 2.0 * gamma * gamma / gm1
    term2 = (2.0 / gp1) ** (gp1 / gm1)
    term3 = 1.0 - pressure_ratio ** (gm1 / gamma)
    
    Cf_momentum = np.sqrt(term1 * term2 * term3)
    Cf_pressure = epsilon * (pressure_ratio - Pa / Pc)
    Cf = Cf_momentum + Cf_pressure
    
    # Mass flow rate
    m_dot = Pc * At / c_star
    
    # Thrust
    thrust = Cf * Pc * At
    
    # Specific impulse
    isp = c_star * Cf / g0
    
    return {
        'thrust': thrust,
        'isp': isp,
        'c_star': c_star,
        'c_f': Cf,
        'M_exit': M_exit,
        'T_exit': T_exit,
        'P_exit': P_exit,
        'V_exit': V_exit,
        'm_dot': m_dot,
    }


# =============================================================================
# Design Optimizer
# =============================================================================

class EngineOptimizer:
    """
    Rocket engine design optimizer.
    
    Optimizes for maximum Isp (default) or other objectives.
    """
    
    def __init__(
        self,
        gamma: float = 1.2,
        T_chamber: float = 3500.0,
        mean_mw: float = 18.0,
        Pa: float = 0.0,  # Ambient pressure
        At: float = 0.01,  # Throat area
    ):
        self.gamma = gamma
        self.T_chamber = T_chamber
        self.mean_mw = mean_mw
        self.Pa = Pa
        self.At = At
        
        self.history: List[Dict[str, float]] = []
    
    def _objective_isp(self, x: np.ndarray) -> float:
        """Objective function: negative Isp (for minimization)."""
        Pc, OF, epsilon = x
        
        # O/F affects mean_mw and gamma in reality
        # Simplified: use fixed values
        
        try:
            result = evaluate_engine_performance(
                Pc=Pc,
                epsilon=epsilon,
                gamma=self.gamma,
                T_chamber=self.T_chamber,
                mean_mw=self.mean_mw,
                At=self.At,
                Pa=self.Pa
            )
            
            self.history.append({
                'Pc': Pc,
                'OF': OF,
                'epsilon': epsilon,
                'isp': result['isp']
            })
            
            return -result['isp']  # Negative for minimization
            
        except Exception:
            return 1e10  # Penalty for invalid configurations
    
    def _objective_thrust_weight(self, x: np.ndarray) -> float:
        """Objective: maximize thrust-to-weight."""
        Pc, OF, epsilon = x
        
        try:
            result = evaluate_engine_performance(
                Pc=Pc,
                epsilon=epsilon,
                gamma=self.gamma,
                T_chamber=self.T_chamber,
                mean_mw=self.mean_mw,
                At=self.At,
                Pa=self.Pa
            )
            
            # Estimate engine weight (simplified)
            nozzle_length = 0.5 * np.sqrt(self.At * epsilon)  # Rough estimate
            engine_mass = 50.0 + 10.0 * nozzle_length  # kg, rough estimate
            
            t_w = result['thrust'] / (engine_mass * 9.80665)
            
            return -t_w  # Negative for minimization
            
        except Exception:
            return 1e10
    
    def optimize(
        self,
        objective: str = 'isp',
        bounds: Optional[OptimizationBounds] = None,
        x0: Optional[np.ndarray] = None,
        method: str = 'Nelder-Mead',
        maxiter: int = 200,
    ) -> OptimizationResult:
        """
        Run optimization.
        
        Args:
            objective: 'isp' or 'thrust_weight'
            bounds: Parameter bounds
            x0: Initial guess [Pc, OF, epsilon]
            method: Optimization method ('Nelder-Mead', 'Powell', 'L-BFGS-B')
            maxiter: Maximum iterations
            
        Returns:
            OptimizationResult
        """
        if bounds is None:
            bounds = OptimizationBounds()
        
        if x0 is None:
            # Start in middle of bounds
            x0 = np.array([
                (bounds.pc_range[0] + bounds.pc_range[1]) / 2,
                (bounds.of_range[0] + bounds.of_range[1]) / 2,
                (bounds.epsilon_range[0] + bounds.epsilon_range[1]) / 2
            ])
        
        self.history = []
        
        # Select objective function
        if objective == 'isp':
            obj_func = self._objective_isp
        elif objective == 'thrust_weight':
            obj_func = self._objective_thrust_weight
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Run optimization
        if method == 'Nelder-Mead':
            result = optimize.minimize(
                obj_func,
                x0,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-4}
            )
        elif method == 'L-BFGS-B':
            result = optimize.minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=[bounds.pc_range, bounds.of_range, bounds.epsilon_range],
                options={'maxiter': maxiter}
            )
        else:
            result = optimize.minimize(
                obj_func,
                x0,
                method=method,
                options={'maxiter': maxiter}
            )
        
        # Extract results
        optimal_params = {
            'Pc': result.x[0],
            'OF': result.x[1],
            'epsilon': result.x[2]
        }
        
        return OptimizationResult(
            optimal_params=optimal_params,
            objective_value=-result.fun,  # Negate back
            iterations=result.nit if hasattr(result, 'nit') else len(self.history),
            success=result.success,
            message=result.message if hasattr(result, 'message') else str(result),
            history=self.history.copy()
        )
    
    def find_optimal_expansion_ratio(
        self,
        Pc: float,
        Pa: float,
        epsilon_range: Tuple[float, float] = (5, 200)
    ) -> float:
        """
        Find optimal expansion ratio for given chamber and ambient pressure.
        
        For adapted nozzle: P_exit = P_ambient
        
        Args:
            Pc: Chamber pressure (Pa)
            Pa: Ambient pressure (Pa)
            epsilon_range: Search range for expansion ratio
            
        Returns:
            Optimal expansion ratio
        """
        def objective(epsilon):
            result = evaluate_engine_performance(
                Pc=Pc,
                epsilon=epsilon[0],
                gamma=self.gamma,
                T_chamber=self.T_chamber,
                mean_mw=self.mean_mw,
                At=self.At,
                Pa=Pa
            )
            # We want P_exit = Pa for optimal expansion
            return abs(result['P_exit'] - Pa)
        
        result = optimize.minimize(
            objective,
            x0=[(epsilon_range[0] + epsilon_range[1]) / 2],
            method='Nelder-Mead',
            bounds=[epsilon_range]
        )
        
        return result.x[0]


# =============================================================================
# Monte Carlo Reliability Analysis
# =============================================================================

def _run_single_simulation(args: Tuple) -> Dict[str, float]:
    """Run a single Monte Carlo sample (for parallel execution)."""
    (Pc, At, OF, gamma, T_chamber, mean_mw, epsilon, Pa) = args
    
    try:
        result = evaluate_engine_performance(
            Pc=Pc,
            epsilon=epsilon,
            gamma=gamma,
            T_chamber=T_chamber,
            mean_mw=mean_mw,
            At=At,
            Pa=Pa
        )
        return result
    except Exception:
        return {'thrust': np.nan, 'isp': np.nan, 'c_star': np.nan}


class MonteCarloAnalyzer:
    """
    Monte Carlo reliability analysis for rocket engines.
    
    Uses parallel execution for performance.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize analyzer.
        
        Args:
            n_workers: Number of parallel workers (None = auto)
        """
        self.n_workers = n_workers
    
    def run(
        self,
        inputs: MonteCarloInput,
        n_samples: int = 1000,
        Pa: float = 0.0,
        thrust_threshold: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            inputs: Nominal values and uncertainties
            n_samples: Number of samples
            Pa: Ambient pressure (Pa)
            thrust_threshold: Threshold for reliability calculation
            seed: Random seed for reproducibility
            
        Returns:
            MonteCarloResult with distributions and statistics
        """
        start_time = time.time()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random samples
        Pc_samples = np.random.normal(
            inputs.Pc_nominal,
            inputs.Pc_nominal * inputs.Pc_sigma,
            n_samples
        )
        At_samples = np.random.normal(
            inputs.At_nominal,
            inputs.At_nominal * inputs.At_sigma,
            n_samples
        )
        OF_samples = np.random.normal(
            inputs.OF_nominal,
            inputs.OF_nominal * inputs.OF_sigma,
            n_samples
        )
        gamma_samples = np.random.normal(
            inputs.gamma,
            inputs.gamma * inputs.gamma_sigma,
            n_samples
        )
        
        # Clamp to physical ranges
        Pc_samples = np.clip(Pc_samples, 0.1e6, 100e6)
        At_samples = np.clip(At_samples, 1e-6, 1.0)
        OF_samples = np.clip(OF_samples, 0.5, 20.0)
        gamma_samples = np.clip(gamma_samples, 1.1, 1.67)
        
        # Prepare arguments for parallel execution
        args_list = [
            (Pc_samples[i], At_samples[i], OF_samples[i], gamma_samples[i],
             inputs.T_chamber, inputs.mean_mw, inputs.epsilon, Pa)
            for i in range(n_samples)
        ]
        
        # Run simulations
        results = []
        
        if self.n_workers == 1:
            # Sequential execution (for debugging)
            for args in args_list:
                results.append(_run_single_simulation(args))
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(_run_single_simulation, args) 
                           for args in args_list]
                
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception:
                        results.append({'thrust': np.nan, 'isp': np.nan, 'c_star': np.nan})
        
        # Extract distributions
        thrust_dist = np.array([r['thrust'] for r in results])
        isp_dist = np.array([r['isp'] for r in results])
        cstar_dist = np.array([r['c_star'] for r in results])
        
        # Remove NaN values
        valid_mask = ~(np.isnan(thrust_dist) | np.isnan(isp_dist))
        thrust_dist = thrust_dist[valid_mask]
        isp_dist = isp_dist[valid_mask]
        cstar_dist = cstar_dist[valid_mask]
        
        runtime = time.time() - start_time
        
        # Calculate reliability if threshold provided
        if thrust_threshold is not None:
            reliability = np.mean(thrust_dist > thrust_threshold)
        else:
            reliability = 0.0
            thrust_threshold = 0.0
        
        return MonteCarloResult(
            thrust_distribution=thrust_dist,
            isp_distribution=isp_dist,
            cstar_distribution=cstar_dist,
            n_samples=len(thrust_dist),
            runtime_seconds=runtime,
            thrust_mean=np.mean(thrust_dist),
            thrust_std=np.std(thrust_dist),
            thrust_p95=np.percentile(thrust_dist, 95),
            thrust_p99=np.percentile(thrust_dist, 99),
            isp_mean=np.mean(isp_dist),
            isp_std=np.std(isp_dist),
            isp_p95=np.percentile(isp_dist, 95),
            isp_p99=np.percentile(isp_dist, 99),
            reliability=reliability,
            threshold=thrust_threshold
        )
    
    def run_sequential(
        self,
        inputs: MonteCarloInput,
        n_samples: int = 1000,
        Pa: float = 0.0,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo sequentially (no multiprocessing).
        
        Useful for environments where multiprocessing is problematic.
        """
        old_workers = self.n_workers
        self.n_workers = 1
        
        try:
            result = self.run(inputs, n_samples, Pa, seed=seed)
        finally:
            self.n_workers = old_workers
        
        return result


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def sensitivity_analysis(
    inputs: MonteCarloInput,
    n_samples: int = 500,
    Pa: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Perform one-at-a-time sensitivity analysis.
    
    Returns:
        Dictionary mapping input parameter to output sensitivities
    """
    # Baseline evaluation
    baseline = evaluate_engine_performance(
        Pc=inputs.Pc_nominal,
        epsilon=inputs.epsilon,
        gamma=inputs.gamma,
        T_chamber=inputs.T_chamber,
        mean_mw=inputs.mean_mw,
        At=inputs.At_nominal,
        Pa=Pa
    )
    
    sensitivities = {}
    
    # Perturb each input
    for param, (nominal, sigma) in [
        ('Pc', (inputs.Pc_nominal, inputs.Pc_sigma)),
        ('At', (inputs.At_nominal, inputs.At_sigma)),
        ('gamma', (inputs.gamma, inputs.gamma_sigma)),
    ]:
        delta = nominal * sigma
        
        if param == 'Pc':
            result_plus = evaluate_engine_performance(
                Pc=nominal + delta, epsilon=inputs.epsilon, gamma=inputs.gamma,
                T_chamber=inputs.T_chamber, mean_mw=inputs.mean_mw,
                At=inputs.At_nominal, Pa=Pa
            )
            result_minus = evaluate_engine_performance(
                Pc=nominal - delta, epsilon=inputs.epsilon, gamma=inputs.gamma,
                T_chamber=inputs.T_chamber, mean_mw=inputs.mean_mw,
                At=inputs.At_nominal, Pa=Pa
            )
        elif param == 'At':
            result_plus = evaluate_engine_performance(
                Pc=inputs.Pc_nominal, epsilon=inputs.epsilon, gamma=inputs.gamma,
                T_chamber=inputs.T_chamber, mean_mw=inputs.mean_mw,
                At=nominal + delta, Pa=Pa
            )
            result_minus = evaluate_engine_performance(
                Pc=inputs.Pc_nominal, epsilon=inputs.epsilon, gamma=inputs.gamma,
                T_chamber=inputs.T_chamber, mean_mw=inputs.mean_mw,
                At=nominal - delta, Pa=Pa
            )
        else:  # gamma
            result_plus = evaluate_engine_performance(
                Pc=inputs.Pc_nominal, epsilon=inputs.epsilon, gamma=nominal + delta,
                T_chamber=inputs.T_chamber, mean_mw=inputs.mean_mw,
                At=inputs.At_nominal, Pa=Pa
            )
            result_minus = evaluate_engine_performance(
                Pc=inputs.Pc_nominal, epsilon=inputs.epsilon, gamma=nominal - delta,
                T_chamber=inputs.T_chamber, mean_mw=inputs.mean_mw,
                At=inputs.At_nominal, Pa=Pa
            )
        
        # Central difference
        sensitivities[param] = {
            'thrust': (result_plus['thrust'] - result_minus['thrust']) / (2 * delta),
            'isp': (result_plus['isp'] - result_minus['isp']) / (2 * delta),
            'c_star': (result_plus['c_star'] - result_minus['c_star']) / (2 * delta),
        }
    
    return sensitivities


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Optimizer Module...")
    print("=" * 50)
    
    # Test performance evaluation
    print("\n1. Performance Evaluation Test:")
    result = evaluate_engine_performance(
        Pc=10e6,  # 100 bar
        epsilon=50,
        gamma=1.2,
        T_chamber=3500,
        mean_mw=18,
        At=0.01,
        Pa=0
    )
    print(f"   Thrust: {result['thrust']/1000:.1f} kN")
    print(f"   Isp: {result['isp']:.1f} s")
    print(f"   C*: {result['c_star']:.1f} m/s")
    
    # Test optimization
    print("\n2. Optimization Test:")
    optimizer = EngineOptimizer(gamma=1.2, T_chamber=3500, mean_mw=18)
    opt_result = optimizer.optimize(
        objective='isp',
        bounds=OptimizationBounds(
            pc_range=(5e6, 20e6),
            of_range=(3.0, 8.0),
            epsilon_range=(20, 100)
        ),
        maxiter=50
    )
    print(f"   Optimal Isp: {opt_result.objective_value:.1f} s")
    print(f"   Optimal Pc: {opt_result.optimal_params['Pc']/1e6:.1f} MPa")
    print(f"   Optimal ε: {opt_result.optimal_params['epsilon']:.1f}")
    
    # Test Monte Carlo (small sample for speed)
    print("\n3. Monte Carlo Test (100 samples):")
    mc = MonteCarloAnalyzer(n_workers=1)  # Sequential for test
    mc_inputs = MonteCarloInput(
        Pc_nominal=10e6,
        At_nominal=0.01,
        gamma=1.2,
        T_chamber=3500,
        mean_mw=18,
        epsilon=50
    )
    mc_result = mc.run_sequential(mc_inputs, n_samples=100, seed=42)
    print(f"   Thrust: {mc_result.thrust_mean/1000:.1f} ± {mc_result.thrust_std/1000:.1f} kN")
    print(f"   Isp: {mc_result.isp_mean:.1f} ± {mc_result.isp_std:.1f} s")
    print(f"   Runtime: {mc_result.runtime_seconds:.2f} s")
    
    print("\n✓ Optimizer module test complete!")
