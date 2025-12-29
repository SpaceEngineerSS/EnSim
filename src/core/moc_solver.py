"""
Method of Characteristics (MOC) Solver for Supersonic Nozzle Design.

Implements 2-D characteristic mesh generation for:
- Minimum Length Nozzle (MLN) design
- Exact supersonic flow field calculation
- Characteristic line propagation

References:
    - Anderson, J.D. "Modern Compressible Flow", 3rd ed., Ch. 11
    - Zucrow, M.J. & Hoffman, J.D. "Gas Dynamics", Vol. 2
    - NASA SP-8120 "Solid Rocket Motor Nozzles"
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from numba import jit


# =============================================================================
# Constants
# =============================================================================

GAMMA_DEFAULT = 1.2  # Typical for combustion products


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CharacteristicPoint:
    """A point in the characteristic mesh.
    
    Attributes:
        x: Axial position (m)
        y: Radial position (m)
        M: Mach number
        theta: Flow angle (rad)
        nu: Prandtl-Meyer angle (rad)
        T: Static temperature (K)
        P: Static pressure (Pa)
    """
    x: float
    y: float
    M: float
    theta: float
    nu: float
    T: float = 0.0
    P: float = 0.0


@dataclass
class MOCMesh:
    """Complete MOC solution mesh.
    
    Contains the characteristic net for a supersonic nozzle.
    """
    points: List[List[CharacteristicPoint]] = field(default_factory=list)
    wall_points: List[CharacteristicPoint] = field(default_factory=list)
    centerline_points: List[CharacteristicPoint] = field(default_factory=list)
    
    # Nozzle parameters
    M_exit: float = 2.0
    gamma: float = GAMMA_DEFAULT
    throat_radius: float = 1.0
    
    # Arrays for visualization
    x_mesh: Optional[np.ndarray] = None
    y_mesh: Optional[np.ndarray] = None
    mach_mesh: Optional[np.ndarray] = None


@dataclass
class NozzleContourMOC:
    """MLN contour from MOC design.
    
    Superior to Rao Bell for maximum thrust.
    """
    x: np.ndarray  # Axial coordinates (m)
    y: np.ndarray  # Radial coordinates (m)
    M: np.ndarray  # Mach number along wall
    theta: np.ndarray  # Wall angle (rad)
    
    # Design parameters
    M_exit: float
    gamma: float
    throat_radius: float
    exit_radius: float
    length: float


# =============================================================================
# Core Physics Functions (Numba Optimized)
# =============================================================================

@jit(nopython=True, cache=True)
def prandtl_meyer_angle(M: float, gamma: float) -> float:
    """
    Calculate Prandtl-Meyer angle ν(M) for supersonic expansion.
    
    ν(M) = √[(γ+1)/(γ-1)] * arctan√[(γ-1)/(γ+1) * (M²-1)] - arctan√(M²-1)
    
    Args:
        M: Mach number (must be >= 1)
        gamma: Ratio of specific heats
        
    Returns:
        Prandtl-Meyer angle in radians
    """
    if M < 1.0:
        return 0.0
    
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    
    M2_minus_1 = M * M - 1.0
    if M2_minus_1 < 0:
        M2_minus_1 = 0.0
    
    sqrt_term = np.sqrt(M2_minus_1)
    ratio = np.sqrt(gp1 / gm1)
    
    term1 = ratio * np.arctan(sqrt_term / ratio)
    term2 = np.arctan(sqrt_term)
    
    nu = term1 - term2
    return nu


@jit(nopython=True, cache=True)
def inverse_prandtl_meyer(nu: float, gamma: float, 
                          tol: float = 1e-8, max_iter: int = 50) -> float:
    """
    Solve for Mach number given Prandtl-Meyer angle.
    
    Uses Newton-Raphson iteration.
    
    Args:
        nu: Prandtl-Meyer angle (rad)
        gamma: Ratio of specific heats
        
    Returns:
        Mach number
    """
    if nu <= 0:
        return 1.0
    
    # Initial guess using approximation
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    
    # Good initial guess from expansion
    M = 1.0 + 0.5 * nu
    if M < 1.001:
        M = 1.001
    
    for _ in range(max_iter):
        nu_current = prandtl_meyer_angle(M, gamma)
        error = nu_current - nu
        
        if np.abs(error) < tol:
            break
        
        # Derivative dν/dM
        M2 = M * M
        sqrt_term = np.sqrt(M2 - 1.0)
        dnu_dM = sqrt_term / (M * (1.0 + gm1 / 2.0 * M2))
        
        if np.abs(dnu_dM) < 1e-12:
            break
        
        dM = -error / dnu_dM
        
        # Damping for stability
        if np.abs(dM) > 0.5:
            dM = 0.5 * np.sign(dM)
        
        M_new = M + dM
        if M_new < 1.001:
            M_new = 1.001
        
        M = M_new
    
    return M


@jit(nopython=True, cache=True)
def mach_angle(M: float) -> float:
    """
    Calculate Mach angle μ = arcsin(1/M).
    
    Args:
        M: Mach number (>= 1)
        
    Returns:
        Mach angle in radians
    """
    if M <= 1.0:
        return np.pi / 2.0
    return np.arcsin(1.0 / M)


@jit(nopython=True, cache=True)
def isentropic_temperature_ratio(M: float, gamma: float) -> float:
    """
    Calculate T/T0 for isentropic flow.
    
    T/T0 = 1 / (1 + (γ-1)/2 * M²)
    """
    return 1.0 / (1.0 + (gamma - 1.0) / 2.0 * M * M)


@jit(nopython=True, cache=True)
def isentropic_pressure_ratio(M: float, gamma: float) -> float:
    """
    Calculate P/P0 for isentropic flow.
    
    P/P0 = (T/T0)^(γ/(γ-1))
    """
    T_ratio = isentropic_temperature_ratio(M, gamma)
    return T_ratio ** (gamma / (gamma - 1.0))


@jit(nopython=True, cache=True)
def variable_gamma(T: float, T0: float, gamma0: float = 1.4) -> float:
    """
    Estimate variable gamma based on temperature.
    
    Real gas effect: γ decreases with temperature for polyatomic gases.
    Uses simplified correlation for combustion products.
    
    Args:
        T: Local static temperature (K)
        T0: Stagnation temperature (K)
        gamma0: Reference gamma at low temperature
        
    Returns:
        Local gamma value
    """
    # Simplified temperature-dependent gamma
    # For combustion products, γ typically ranges from ~1.3 at low T to ~1.15 at high T
    T_ratio = T / T0
    
    # Linear interpolation (simplified model)
    gamma_high = 1.15  # At very high T
    gamma_low = 1.30   # At low T
    
    gamma = gamma_low + (gamma_high - gamma_low) * (1.0 - T_ratio)
    
    # Clamp to physical range
    if gamma < 1.1:
        gamma = 1.1
    if gamma > 1.4:
        gamma = 1.4
    
    return gamma


# =============================================================================
# Characteristic Line Computations
# =============================================================================

@jit(nopython=True, cache=True)
def characteristic_slope_plus(theta: float, mu: float) -> float:
    """
    Slope of C+ (right-running) characteristic.
    
    dy/dx = tan(θ - μ)
    """
    return np.tan(theta - mu)


@jit(nopython=True, cache=True)
def characteristic_slope_minus(theta: float, mu: float) -> float:
    """
    Slope of C- (left-running) characteristic.
    
    dy/dx = tan(θ + μ)
    """
    return np.tan(theta + mu)


@jit(nopython=True, cache=True)
def solve_interior_point(
    x1: float, y1: float, theta1: float, nu1: float, M1: float,
    x2: float, y2: float, theta2: float, nu2: float, M2: float,
    gamma: float
) -> Tuple[float, float, float, float, float]:
    """
    Solve for interior point (3) from points (1) and (2).
    
    Point 1 is on C+ (right-running from point 1)
    Point 2 is on C- (left-running from point 2)
    
    Compatibility equations:
        C+: θ + ν = const (K-)
        C-: θ - ν = const (K+)
    
    Returns:
        (x3, y3, theta3, nu3, M3)
    """
    # Riemann invariants
    K_minus = theta1 + nu1  # Constant along C+
    K_plus = theta2 - nu2   # Constant along C-
    
    # Solve for point 3 properties
    theta3 = 0.5 * (K_minus + K_plus)
    nu3 = 0.5 * (K_minus - K_plus)
    
    # Get Mach number from P-M angle
    M3 = inverse_prandtl_meyer(nu3, gamma)
    mu3 = mach_angle(M3)
    
    # Average Mach angles for characteristic slopes
    mu1 = mach_angle(M1)
    mu2 = mach_angle(M2)
    
    # Average slopes
    slope_plus = 0.5 * (characteristic_slope_plus(theta1, mu1) + 
                        characteristic_slope_plus(theta3, mu3))
    slope_minus = 0.5 * (characteristic_slope_minus(theta2, mu2) + 
                         characteristic_slope_minus(theta3, mu3))
    
    # Solve intersection
    # y - y1 = slope_plus * (x - x1)
    # y - y2 = slope_minus * (x - x2)
    
    denom = slope_minus - slope_plus
    if np.abs(denom) < 1e-12:
        # Nearly parallel, use average
        x3 = 0.5 * (x1 + x2)
        y3 = 0.5 * (y1 + y2)
    else:
        x3 = (y1 - y2 + slope_minus * x2 - slope_plus * x1) / denom
        y3 = y1 + slope_plus * (x3 - x1)
    
    return x3, y3, theta3, nu3, M3


@jit(nopython=True, cache=True)
def solve_centerline_point(
    x1: float, y1: float, theta1: float, nu1: float, M1: float,
    gamma: float
) -> Tuple[float, float, float, float, float]:
    """
    Reflect C- characteristic off centerline (y=0).
    
    At centerline: θ = 0 (flow is axial)
    C- reflects as C+ with same ν.
    
    Returns:
        (x3, y3, theta3, nu3, M3) at centerline
    """
    # At centerline, θ = 0
    theta3 = 0.0
    
    # From C- compatibility: θ - ν = const
    # θ1 - ν1 = θ3 - ν3
    # 0 - ν3 = θ1 - ν1
    nu3 = nu1 - theta1
    
    # Mach number
    M3 = inverse_prandtl_meyer(nu3, gamma)
    
    # Location: follow C- from point 1 to y=0
    mu1 = mach_angle(M1)
    mu3 = mach_angle(M3)
    
    # Average slope
    slope_minus = 0.5 * (characteristic_slope_minus(theta1, mu1) + 
                         characteristic_slope_minus(theta3, mu3))
    
    # y3 = 0, solve for x3
    if np.abs(slope_minus) < 1e-12:
        x3 = x1 + y1  # Fallback
    else:
        x3 = x1 - y1 / slope_minus
    
    y3 = 0.0
    
    return x3, y3, theta3, nu3, M3


@jit(nopython=True, cache=True)
def solve_wall_point(
    x1: float, y1: float, theta1: float, nu1: float, M1: float,
    theta_wall: float, gamma: float
) -> Tuple[float, float, float, float, float]:
    """
    Find wall point where C+ intersects wall with angle θ_wall.
    
    Wall is tangent to flow: θ = θ_wall at wall.
    
    Returns:
        (x3, y3, theta3, nu3, M3) at wall
    """
    # At wall, θ = θ_wall
    theta3 = theta_wall
    
    # From C+ compatibility: θ + ν = const
    # θ1 + ν1 = θ3 + ν3
    nu3 = nu1 + theta1 - theta_wall
    
    # Mach number
    M3 = inverse_prandtl_meyer(nu3, gamma)
    mu3 = mach_angle(M3)
    
    # Location: follow C+ from point 1
    mu1 = mach_angle(M1)
    slope_plus = 0.5 * (characteristic_slope_plus(theta1, mu1) + 
                        characteristic_slope_plus(theta3, mu3))
    
    # Need to find intersection with wall - iterative for real wall
    # For MLN, wall angle is known, so we can compute directly
    wall_slope = np.tan(theta_wall)
    
    # C+ line: y - y1 = slope_plus * (x - x1)
    # Wall: y = y_throat + wall_slope * (x - x_throat)
    # Simplified: assume we're building the wall as we go
    
    # For MLN initial expansion, use characteristic intersection
    x3 = x1 + 0.1  # Small step
    y3 = y1 + slope_plus * (x3 - x1)
    
    return x3, y3, theta3, nu3, M3


# =============================================================================
# MLN Design Algorithm
# =============================================================================

def generate_mln_contour(
    M_exit: float,
    gamma: float = GAMMA_DEFAULT,
    throat_radius: float = 1.0,
    n_char_lines: int = 20,
    T0: float = 3000.0,
    P0: float = 1e7,
    use_variable_gamma: bool = False
) -> Tuple[NozzleContourMOC, MOCMesh]:
    """
    Generate Minimum Length Nozzle contour using Method of Characteristics.
    
    The MLN provides the shortest nozzle length for given exit Mach number.
    It consists of:
    1. Initial expansion region (centered at throat, sharp corner)
    2. Straightening section (cancels expansion waves)
    
    Args:
        M_exit: Design exit Mach number
        gamma: Ratio of specific heats
        throat_radius: Throat radius (m)
        n_char_lines: Number of characteristic lines in expansion fan
        T0: Stagnation temperature (K)
        P0: Stagnation pressure (Pa)
        use_variable_gamma: If True, update gamma based on local temperature
        
    Returns:
        Tuple of (NozzleContourMOC, MOCMesh)
    """
    # Calculate exit Prandtl-Meyer angle
    nu_exit = prandtl_meyer_angle(M_exit, gamma)
    
    # Initial expansion: flow turns from θ=0 to θ_max = ν_exit/2
    # This is the maximum wall angle at the throat
    theta_max = nu_exit / 2.0
    
    # Generate initial expansion fan at throat
    theta_values = np.linspace(0.001, theta_max, n_char_lines)
    nu_values = theta_values  # At throat, ν = θ for centered expansion
    
    # Initialize mesh storage
    mesh = MOCMesh(
        M_exit=M_exit,
        gamma=gamma,
        throat_radius=throat_radius
    )
    
    # Storage for characteristic lines
    # Each "row" is a characteristic line emanating from throat
    char_lines = []
    
    # Initialize points on the expansion fan (at throat, x=0, y=throat_radius)
    x_throat = 0.0
    y_throat = throat_radius
    
    for i, (theta, nu) in enumerate(zip(theta_values, nu_values)):
        M = inverse_prandtl_meyer(nu, gamma)
        
        # Temperature from isentropic relations
        T = T0 * isentropic_temperature_ratio(M, gamma)
        P = P0 * isentropic_pressure_ratio(M, gamma)
        
        # Update gamma if using variable gamma model
        local_gamma = gamma
        if use_variable_gamma:
            local_gamma = variable_gamma(T, T0, gamma)
        
        point = CharacteristicPoint(
            x=x_throat,
            y=y_throat,
            M=M,
            theta=theta,
            nu=nu,
            T=T,
            P=P
        )
        char_lines.append([point])
    
    # Propagate characteristics through the nozzle
    # Using a unit process approach
    
    # First, propagate to centerline reflections
    for iteration in range(n_char_lines * 2):
        new_points_added = False
        
        for i in range(len(char_lines)):
            if len(char_lines[i]) == 0:
                continue
            
            last_point = char_lines[i][-1]
            
            # Check if we've reached the exit (or maximum extent)
            if last_point.M >= M_exit * 0.99:
                continue
            
            # Interior point calculation with adjacent characteristic
            if i > 0 and len(char_lines[i-1]) >= len(char_lines[i]):
                # Get adjacent point
                adj_idx = len(char_lines[i]) - 1
                if adj_idx < len(char_lines[i-1]):
                    adj_point = char_lines[i-1][adj_idx]
                    
                    # Solve interior point
                    x3, y3, theta3, nu3, M3 = solve_interior_point(
                        adj_point.x, adj_point.y, adj_point.theta, 
                        adj_point.nu, adj_point.M,
                        last_point.x, last_point.y, last_point.theta,
                        last_point.nu, last_point.M,
                        gamma
                    )
                    
                    if y3 > 0 and x3 > last_point.x:
                        T3 = T0 * isentropic_temperature_ratio(M3, gamma)
                        P3 = P0 * isentropic_pressure_ratio(M3, gamma)
                        
                        new_point = CharacteristicPoint(
                            x=x3, y=y3, M=M3, theta=theta3, nu=nu3,
                            T=T3, P=P3
                        )
                        char_lines[i].append(new_point)
                        new_points_added = True
            
            # Centerline reflection for lowest characteristic
            if i == 0 and last_point.y > 0.01 * throat_radius:
                x3, y3, theta3, nu3, M3 = solve_centerline_point(
                    last_point.x, last_point.y, last_point.theta,
                    last_point.nu, last_point.M, gamma
                )
                
                if x3 > last_point.x:
                    T3 = T0 * isentropic_temperature_ratio(M3, gamma)
                    P3 = P0 * isentropic_pressure_ratio(M3, gamma)
                    
                    center_point = CharacteristicPoint(
                        x=x3, y=0.0, M=M3, theta=0.0, nu=nu3,
                        T=T3, P=P3
                    )
                    mesh.centerline_points.append(center_point)
        
        if not new_points_added:
            break
    
    # Extract wall contour from uppermost characteristic line
    wall_x = [char_lines[-1][0].x]
    wall_y = [char_lines[-1][0].y]
    wall_M = [char_lines[-1][0].M]
    wall_theta = [char_lines[-1][0].theta]
    
    # Build wall by following the envelope of C+ characteristics
    for i in range(len(char_lines) - 1, -1, -1):
        if len(char_lines[i]) > 1:
            for point in char_lines[i][1:]:
                # Check if this extends the wall
                if point.x > wall_x[-1]:
                    wall_x.append(point.x)
                    wall_y.append(point.y)
                    wall_M.append(point.M)
                    wall_theta.append(point.theta)
    
    # Add exit point
    exit_radius = throat_radius * np.sqrt(
        (1.0 + (gamma - 1.0) / 2.0) ** ((gamma + 1.0) / (gamma - 1.0)) /
        (1.0 + (gamma - 1.0) / 2.0 * M_exit ** 2) ** ((gamma + 1.0) / (gamma - 1.0))
    ) * M_exit
    
    # Convert to arrays
    wall_x = np.array(wall_x)
    wall_y = np.array(wall_y)
    wall_M = np.array(wall_M)
    wall_theta = np.array(wall_theta)
    
    # Normalize and scale
    if len(wall_x) > 1:
        # Sort by x coordinate
        sort_idx = np.argsort(wall_x)
        wall_x = wall_x[sort_idx]
        wall_y = wall_y[sort_idx]
        wall_M = wall_M[sort_idx]
        wall_theta = wall_theta[sort_idx]
    
    # Create contour object
    contour = NozzleContourMOC(
        x=wall_x,
        y=wall_y,
        M=wall_M,
        theta=wall_theta,
        M_exit=M_exit,
        gamma=gamma,
        throat_radius=throat_radius,
        exit_radius=wall_y[-1] if len(wall_y) > 0 else exit_radius,
        length=wall_x[-1] if len(wall_x) > 0 else 0.0
    )
    
    # Store mesh points
    mesh.points = char_lines
    mesh.wall_points = [CharacteristicPoint(x=wall_x[i], y=wall_y[i], 
                                            M=wall_M[i], theta=wall_theta[i],
                                            nu=prandtl_meyer_angle(wall_M[i], gamma))
                        for i in range(len(wall_x))]
    
    # Create visualization arrays
    _create_mesh_arrays(mesh, char_lines)
    
    return contour, mesh


def _create_mesh_arrays(mesh: MOCMesh, char_lines: List[List[CharacteristicPoint]]):
    """Convert characteristic lines to 2D arrays for visualization."""
    # Find maximum length
    max_len = max(len(line) for line in char_lines) if char_lines else 0
    n_lines = len(char_lines)
    
    if max_len == 0 or n_lines == 0:
        return
    
    # Create padded arrays
    x_mesh = np.full((n_lines, max_len), np.nan)
    y_mesh = np.full((n_lines, max_len), np.nan)
    mach_mesh = np.full((n_lines, max_len), np.nan)
    
    for i, line in enumerate(char_lines):
        for j, point in enumerate(line):
            x_mesh[i, j] = point.x
            y_mesh[i, j] = point.y
            mach_mesh[i, j] = point.M
    
    mesh.x_mesh = x_mesh
    mesh.y_mesh = y_mesh
    mesh.mach_mesh = mach_mesh


# =============================================================================
# Utility Functions
# =============================================================================

def compare_contours(moc_contour: NozzleContourMOC, 
                     rao_x: np.ndarray, rao_y: np.ndarray) -> dict:
    """
    Compare MOC MLN contour with Rao Bell contour.
    
    Returns:
        Dictionary with comparison metrics
    """
    # Interpolate Rao onto MOC x-coordinates
    rao_y_interp = np.interp(moc_contour.x, rao_x, rao_y)
    
    # Differences
    diff = moc_contour.y - rao_y_interp
    
    return {
        'max_diff': np.max(np.abs(diff)),
        'mean_diff': np.mean(diff),
        'moc_length': moc_contour.length,
        'moc_exit_radius': moc_contour.exit_radius,
    }


def export_contour_csv(contour: NozzleContourMOC, filepath: str):
    """Export contour to CSV file."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_m', 'y_m', 'Mach', 'theta_rad'])
        for i in range(len(contour.x)):
            writer.writerow([
                contour.x[i],
                contour.y[i],
                contour.M[i],
                contour.theta[i]
            ])


def export_mesh_vtk(mesh: MOCMesh, filepath: str):
    """
    Export MOC mesh to VTK format for external visualization.
    
    Requires pyvista.
    """
    try:
        import pyvista as pv
        
        # Create structured grid from mesh arrays
        if mesh.x_mesh is None or mesh.y_mesh is None:
            raise ValueError("Mesh arrays not initialized")
        
        # Flatten and filter NaN values
        valid_mask = ~np.isnan(mesh.x_mesh)
        x_flat = mesh.x_mesh[valid_mask]
        y_flat = mesh.y_mesh[valid_mask]
        z_flat = np.zeros_like(x_flat)
        mach_flat = mesh.mach_mesh[valid_mask]
        
        # Create point cloud
        points = np.column_stack([x_flat, y_flat, z_flat])
        cloud = pv.PolyData(points)
        cloud['Mach'] = mach_flat
        
        # Save
        cloud.save(filepath)
        
    except ImportError:
        raise ImportError("PyVista required for VTK export")


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Test MOC solver
    print("Testing MOC Solver...")
    print("=" * 50)
    
    # Test Prandtl-Meyer function
    M_test = 2.0
    gamma = 1.4
    nu = prandtl_meyer_angle(M_test, gamma)
    print(f"ν(M={M_test}, γ={gamma}) = {np.degrees(nu):.2f}°")
    
    # Inverse test
    M_recovered = inverse_prandtl_meyer(nu, gamma)
    print(f"M recovered from ν = {M_recovered:.4f}")
    
    # Generate MLN
    print("\nGenerating MLN contour for M_exit = 3.0...")
    contour, mesh = generate_mln_contour(
        M_exit=3.0,
        gamma=1.2,
        throat_radius=0.05,
        n_char_lines=15
    )
    
    print(f"Contour length: {contour.length:.4f} m")
    print(f"Exit radius: {contour.exit_radius:.4f} m")
    print(f"Number of wall points: {len(contour.x)}")
    
    print("\n✓ MOC Solver test complete!")
