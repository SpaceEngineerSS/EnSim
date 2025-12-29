"""
Numerical Integration Methods for ODE Solving.

Provides Numba-optimized adaptive step-size integrators for the
6-DOF flight simulator. Uses Dormand-Prince 5(4) method (ode45)
for high accuracy with automatic step-size control.

Key Features:
    - Dormand-Prince 5(4) RK method (7 stages, FSAL property)
    - Error estimation for adaptive stepping
    - Step-size control with PI controller
    - Numba JIT compilation for performance

References:
    - Dormand, J.R. & Prince, P.J. (1980). "A family of embedded 
      Runge-Kutta formulae." J. Comp. Appl. Math. 6(1): 19-26.
    - Hairer, Nørsett & Wanner (1993). "Solving ODEs I: Nonstiff Problems"
"""

import numpy as np
from numba import jit


# =============================================================================
# Dormand-Prince 5(4) Butcher Tableau Coefficients
# =============================================================================

# Time nodes (c_i): fraction of step at each stage
DP5_C = np.array([
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    4.0 / 5.0,
    8.0 / 9.0,
    1.0,
    1.0
], dtype=np.float64)

# Coefficient matrix (a_ij): weights for computing k_i from previous k's
# Row i gives coefficients for computing k_{i+1}
DP5_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
    [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
    [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
    [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
], dtype=np.float64)

# 5th order solution weights (b_i): for the accepted solution
DP5_B5 = np.array([
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0
], dtype=np.float64)

# 4th order solution weights (b*_i): for error estimation
DP5_B4 = np.array([
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0
], dtype=np.float64)

# Error coefficients: E = b5 - b4 (difference for error estimate)
DP5_E = DP5_B5 - DP5_B4


# =============================================================================
# Adaptive Step-Size Control Constants
# =============================================================================

# Safety factor for step-size adjustment (prevents oscillation)
SAFETY = 0.9

# PI controller exponents for step-size adjustment
# New step = h * SAFETY * (tol/err)^ALPHA * (prev_err/tol)^BETA
ALPHA = 0.2   # Main error exponent (1/5 for 5th order)
BETA = 0.04   # Smoothing exponent for step history

# Step-size limits (relative to current step)
MAX_FACTOR = 5.0   # Maximum step increase factor
MIN_FACTOR = 0.2   # Minimum step decrease factor


# =============================================================================
# Core RK45 Functions (Numba JIT)
# =============================================================================

@jit(nopython=True, cache=True)
def rk45_compute_stages(
    y: np.ndarray,
    k: np.ndarray,
    h: float,
    n: int
) -> np.ndarray:
    """
    Compute intermediate state for RK45 stage evaluation.
    
    Args:
        y: Current state vector (n elements)
        k: Array of k values computed so far, shape (7, n)
        h: Step size
        n: Dimension of state vector
        
    Returns:
        State vector at current stage position
    """
    # Pre-allocate result
    y_stage = y.copy()
    return y_stage


@jit(nopython=True, cache=True)
def rk45_error_norm(
    y_new: np.ndarray,
    y_old: np.ndarray,
    error: np.ndarray,
    atol: float,
    rtol: float
) -> float:
    """
    Compute scaled error norm for step acceptance.
    
    Uses mixed absolute/relative tolerance scaling:
        scale_i = atol + rtol * max(|y_new_i|, |y_old_i|)
        err_scaled = sqrt(mean((error_i / scale_i)^2))
    
    Args:
        y_new: New state estimate
        y_old: Previous state
        error: Error estimate (difference between 5th and 4th order)
        atol: Absolute tolerance
        rtol: Relative tolerance
        
    Returns:
        Scaled RMS error norm
    """
    n = len(error)
    sum_sq = 0.0
    
    for i in range(n):
        # Scale factor: tolerances weighted by solution magnitude
        scale = atol + rtol * max(abs(y_new[i]), abs(y_old[i]))
        if scale < 1e-30:
            scale = 1e-30
        
        # Add scaled squared error
        sum_sq += (error[i] / scale) ** 2
    
    # RMS norm
    return np.sqrt(sum_sq / n)


@jit(nopython=True, cache=True)
def compute_new_step_size(
    h: float,
    err_norm: float,
    prev_err_norm: float,
    h_min: float,
    h_max: float
) -> float:
    """
    Compute optimal new step size using PI controller.
    
    Uses standard step-size formula with smoothing:
        h_new = h * SAFETY * (1/err)^ALPHA * (prev_err)^BETA
    
    Clamped to [h_min, h_max] and limited by factor constraints.
    
    Args:
        h: Current step size
        err_norm: Current scaled error norm
        prev_err_norm: Previous step's error norm (for smoothing)
        h_min: Minimum allowed step size
        h_max: Maximum allowed step size
        
    Returns:
        New step size
    """
    if err_norm < 1e-30:
        # Error essentially zero - use maximum increase
        factor = MAX_FACTOR
    else:
        # PI controller formula
        factor = SAFETY * (1.0 / err_norm) ** ALPHA
        
        # Add smoothing if we have previous error
        if prev_err_norm > 1e-30:
            factor *= (prev_err_norm / err_norm) ** BETA
    
    # Clamp factor to prevent extreme changes
    if factor > MAX_FACTOR:
        factor = MAX_FACTOR
    if factor < MIN_FACTOR:
        factor = MIN_FACTOR
    
    # Compute new step
    h_new = h * factor
    
    # Clamp to absolute limits
    if h_new < h_min:
        h_new = h_min
    if h_new > h_max:
        h_new = h_max
    
    return h_new


@jit(nopython=True, cache=True)
def cubic_hermite_interpolate(
    t_target: float,
    t0: float,
    t1: float,
    y0: np.ndarray,
    y1: np.ndarray,
    dy0: np.ndarray,
    dy1: np.ndarray
) -> np.ndarray:
    """
    Cubic Hermite interpolation for dense output.
    
    Interpolates state between two known points using cubic polynomials
    that match both value and derivative at endpoints.
    
    Args:
        t_target: Time to interpolate to (t0 <= t_target <= t1)
        t0: Start time
        t1: End time
        y0: State at t0
        y1: State at t1
        dy0: Derivative at t0
        dy1: Derivative at t1
        
    Returns:
        Interpolated state at t_target
    """
    h = t1 - t0
    if h < 1e-30:
        return y0.copy()
    
    # Normalized time [0, 1]
    s = (t_target - t0) / h
    
    # Hermite basis functions
    h00 = 2*s*s*s - 3*s*s + 1      # y0 coefficient
    h10 = s*s*s - 2*s*s + s        # dy0 coefficient (scaled by h)
    h01 = -2*s*s*s + 3*s*s         # y1 coefficient
    h11 = s*s*s - s*s              # dy1 coefficient (scaled by h)
    
    # Interpolated value
    n = len(y0)
    y_interp = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        y_interp[i] = h00 * y0[i] + h10 * h * dy0[i] + \
                      h01 * y1[i] + h11 * h * dy1[i]
    
    return y_interp


@jit(nopython=True, cache=True)
def linear_interpolate(
    t_target: float,
    t0: float,
    t1: float,
    y0: np.ndarray,
    y1: np.ndarray
) -> np.ndarray:
    """
    Simple linear interpolation for dense output.
    
    Faster than cubic Hermite but less accurate.
    Use when derivatives are expensive to compute.
    
    Args:
        t_target: Time to interpolate to
        t0, t1: Endpoint times
        y0, y1: Endpoint states
        
    Returns:
        Linearly interpolated state
    """
    h = t1 - t0
    if h < 1e-30:
        return y0.copy()
    
    alpha = (t_target - t0) / h
    
    n = len(y0)
    y_interp = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        y_interp[i] = (1.0 - alpha) * y0[i] + alpha * y1[i]
    
    return y_interp
