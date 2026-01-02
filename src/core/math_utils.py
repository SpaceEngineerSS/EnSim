"""
Quaternion Mathematics for 6-DOF Rigid Body Dynamics.

Implements quaternion operations using the Hamilton convention:
    q = [w, x, y, z] = w + xi + yj + zk

where:
    i² = j² = k² = ijk = -1

Key properties:
    - Unit quaternions represent rotations in 3D space
    - Quaternion multiplication is non-commutative
    - No gimbal lock (unlike Euler angles)

All functions are Numba JIT-compiled for performance.

References:
    - Diebel, J. "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors"
    - NASA RP-1207 "Euler Angles, Quaternions, and Transformation Matrices"
"""

import numpy as np
from numba import jit

# =============================================================================
# Quaternion Basic Operations
# =============================================================================

@jit(nopython=True, cache=True)
def q_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication (Hamilton product).

    Given q1 = [w1, x1, y1, z1] and q2 = [w2, x2, y2, z2]:

    q1 ⊗ q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
               w1*x2 + x1*w2 + y1*z2 - z1*y2,
               w1*y2 - x1*z2 + y1*w2 + z1*x2,
               w1*z2 + x1*y2 - y1*x2 + z1*w2]

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]

    Note:
        Quaternion multiplication is NOT commutative: q1⊗q2 ≠ q2⊗q1
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


@jit(nopython=True, cache=True)
def q_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate.

    For q = [w, x, y, z]:
        q* = [w, -x, -y, -z]

    For unit quaternions, the conjugate equals the inverse:
        q* = q⁻¹ (when ||q|| = 1)

    Args:
        q: Input quaternion [w, x, y, z]

    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


@jit(nopython=True, cache=True)
def q_norm(q: np.ndarray) -> float:
    """
    Quaternion norm (magnitude).

    ||q|| = sqrt(w² + x² + y² + z²)

    Args:
        q: Input quaternion [w, x, y, z]

    Returns:
        Scalar norm value
    """
    return np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])


@jit(nopython=True, cache=True)
def q_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.

    q_unit = q / ||q||

    This is CRITICAL after numerical integration to prevent
    drift from the unit quaternion constraint.

    Args:
        q: Input quaternion [w, x, y, z]

    Returns:
        Unit quaternion [w, x, y, z] with ||q|| = 1
    """
    norm = q_norm(q)
    if norm < 1e-12:
        # Return identity quaternion if input is degenerate
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


@jit(nopython=True, cache=True)
def q_inverse(q: np.ndarray) -> np.ndarray:
    """
    Quaternion inverse.

    q⁻¹ = q* / ||q||²

    For unit quaternions (||q|| = 1), this simplifies to q⁻¹ = q*.

    Args:
        q: Input quaternion [w, x, y, z]

    Returns:
        Inverse quaternion such that q ⊗ q⁻¹ = [1, 0, 0, 0]
    """
    norm_sq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    if norm_sq < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q_conjugate(q) / norm_sq


# =============================================================================
# Rotation Operations
# =============================================================================

@jit(nopython=True, cache=True)
def q_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a 3D vector by a quaternion.

    Uses the formula:
        v' = q ⊗ [0, v] ⊗ q*

    Optimized direct calculation (avoiding intermediate quaternion products):
        v' = v + 2w(ω × v) + 2(ω × (ω × v))

    where q = [w, ω] and ω = [x, y, z] is the vector part.

    Args:
        q: Unit quaternion [w, x, y, z] representing rotation
        v: 3D vector [x, y, z] to rotate

    Returns:
        Rotated 3D vector [x', y', z']

    Reference:
        Efficient formula from "Quaternion Rotation" by Kavan et al.
    """
    # Extract quaternion components
    w = q[0]
    qx, qy, qz = q[1], q[2], q[3]

    # Vector part of quaternion (axis × sin(θ/2))
    # ω = [qx, qy, qz]

    # Cross product: ω × v
    cx = qy * v[2] - qz * v[1]
    cy = qz * v[0] - qx * v[2]
    cz = qx * v[1] - qy * v[0]

    # Cross product: ω × (ω × v)
    ccx = qy * cz - qz * cy
    ccy = qz * cx - qx * cz
    ccz = qx * cy - qy * cx

    # v' = v + 2w(ω × v) + 2(ω × (ω × v))
    return np.array([
        v[0] + 2.0 * (w * cx + ccx),
        v[1] + 2.0 * (w * cy + ccy),
        v[2] + 2.0 * (w * cz + ccz)
    ])


@jit(nopython=True, cache=True)
def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create quaternion from axis-angle representation.

    q = [cos(θ/2), sin(θ/2) * axis]

    Args:
        axis: Unit vector [x, y, z] defining rotation axis
        angle: Rotation angle in radians (right-hand rule)

    Returns:
        Unit quaternion [w, x, y, z]
    """
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    # Normalize axis (safety)
    axis_norm = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if axis_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    ax = axis[0] / axis_norm
    ay = axis[1] / axis_norm
    az = axis[2] / axis_norm

    return np.array([c, s * ax, s * ay, s * az])


@jit(nopython=True, cache=True)
def q_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Extract axis-angle representation from quaternion.

    Args:
        q: Unit quaternion [w, x, y, z]

    Returns:
        Array [ax, ay, az, angle] where (ax, ay, az) is the axis and angle is in radians
    """
    # Ensure unit quaternion (inline to avoid re-assignment issues)
    norm = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    w = q[0] / norm
    x = q[1] / norm
    y = q[2] / norm
    z = q[3] / norm

    # Clamp w to [-1, 1] for arccos (manual clamp for Numba)
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0

    angle = 2.0 * np.arccos(w)

    s = np.sqrt(1.0 - w * w)
    if s < 1e-8:
        # Angle is ~0, axis is arbitrary
        return np.array([1.0, 0.0, 0.0, 0.0])

    ax = x / s
    ay = y / s
    az = z / s
    return np.array([ax, ay, az, angle])


# =============================================================================
# Euler Angle Conversions (ZYX Convention - Aerospace Standard)
# =============================================================================

@jit(nopython=True, cache=True)
def q_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create quaternion from Euler angles (ZYX convention).

    The ZYX convention applies rotations in order:
        1. Yaw (ψ) around Z-axis
        2. Pitch (θ) around Y-axis
        3. Roll (φ) around X-axis

    This is the aerospace/robotics standard (Tait-Bryan angles).

    Args:
        roll: Rotation around X-axis (φ) in radians
        pitch: Rotation around Y-axis (θ) in radians
        yaw: Rotation around Z-axis (ψ) in radians

    Returns:
        Unit quaternion [w, x, y, z]

    Reference:
        Diebel, "Representing Attitude", Eq. 290
    """
    # Half angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # ZYX quaternion composition
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


@jit(nopython=True, cache=True)
def q_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Extract Euler angles from quaternion (ZYX convention).

    WARNING: Euler angles have singularities at pitch = ±90°.
    Use quaternions directly for numerical integration.

    Args:
        q: Unit quaternion [w, x, y, z]

    Returns:
        Array [roll, pitch, yaw] in radians
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (φ) - rotation around X
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (θ) - rotation around Y
    sinp = 2.0 * (w * y - z * x)
    # Clamp to avoid NaN from arcsin (manual clamp for Numba)
    if sinp > 1.0:
        sinp = 1.0
    elif sinp < -1.0:
        sinp = -1.0
    pitch = np.arcsin(sinp)

    # Yaw (ψ) - rotation around Z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


# =============================================================================
# Quaternion Kinematics (for dynamics integration)
# =============================================================================

@jit(nopython=True, cache=True)
def q_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute quaternion time derivative from angular velocity.

    The kinematic equation is:
        q̇ = ½ ω ⊗ q  (body-fixed angular velocity)

    or equivalently using the Ω matrix:
        q̇ = ½ Ω(ω) q

    where ω = [ωx, ωy, ωz] is the angular velocity in body frame.

    Args:
        q: Current orientation quaternion [w, x, y, z]
        omega: Angular velocity in body frame [ωx, ωy, ωz] (rad/s)

    Returns:
        Quaternion derivative [ẇ, ẋ, ẏ, ż]

    Reference:
        Crassidis & Junkins, "Optimal Estimation of Dynamic Systems", Eq. 2.85
    """
    # Create quaternion from angular velocity: [0, ωx, ωy, ωz]
    omega_q = np.array([0.0, omega[0], omega[1], omega[2]])

    # q̇ = ½ ω_q ⊗ q
    q_dot = 0.5 * q_mult(omega_q, q)

    return q_dot


@jit(nopython=True, cache=True)
def q_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate quaternion over time step using exponential map.

    For small rotation Δθ = ω × dt:
        q(t+dt) = exp(½ Δθ) ⊗ q(t)

    This is more accurate than simple Euler integration
    and preserves the unit quaternion constraint better.

    Args:
        q: Current orientation quaternion [w, x, y, z]
        omega: Angular velocity [ωx, ωy, ωz] (rad/s)
        dt: Time step (s)

    Returns:
        Updated quaternion (normalized)
    """
    # Rotation angle magnitude
    theta = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * dt

    if theta < 1e-10:
        # Small angle approximation
        return q_normalize(q)

    # Half angle for quaternion
    half_theta = theta * 0.5

    # Axis (normalized omega)
    omega_norm = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    ax = omega[0] / omega_norm
    ay = omega[1] / omega_norm
    az = omega[2] / omega_norm

    # Incremental rotation quaternion
    s = np.sin(half_theta)
    c = np.cos(half_theta)
    dq = np.array([c, s * ax, s * ay, s * az])

    # Compose rotations: q_new = dq ⊗ q
    q_new = q_mult(dq, q)

    return q_normalize(q_new)


# =============================================================================
# Rotation Matrix Conversion (for reference/debugging)
# =============================================================================

@jit(nopython=True, cache=True)
def q_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    This is the Direction Cosine Matrix (DCM) from body to inertial frame.

    Args:
        q: Unit quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix (body → inertial)

    Reference:
        NASA RP-1207, Eq. 54
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Precompute products
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.zeros((3, 3))

    R[0, 0] = 1.0 - 2.0 * (yy + zz)
    R[0, 1] = 2.0 * (xy - wz)
    R[0, 2] = 2.0 * (xz + wy)

    R[1, 0] = 2.0 * (xy + wz)
    R[1, 1] = 1.0 - 2.0 * (xx + zz)
    R[1, 2] = 2.0 * (yz - wx)

    R[2, 0] = 2.0 * (xz - wy)
    R[2, 1] = 2.0 * (yz + wx)
    R[2, 2] = 1.0 - 2.0 * (xx + yy)

    return R


# =============================================================================
# Vector Utilities (Numba-compatible)
# =============================================================================

@jit(nopython=True, cache=True)
def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    3D cross product: a × b

    Args:
        a: First vector [x, y, z]
        b: Second vector [x, y, z]

    Returns:
        Cross product vector [x, y, z]
    """
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])


@jit(nopython=True, cache=True)
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    3D dot product: a · b

    Args:
        a: First vector [x, y, z]
        b: Second vector [x, y, z]

    Returns:
        Scalar dot product
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit(nopython=True, cache=True)
def vector_norm(v: np.ndarray) -> float:
    """
    3D vector magnitude: ||v||

    Args:
        v: Input vector [x, y, z]

    Returns:
        Scalar norm
    """
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


@jit(nopython=True, cache=True)
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize 3D vector to unit length.

    Args:
        v: Input vector [x, y, z]

    Returns:
        Unit vector [x, y, z] with ||v|| = 1
    """
    norm = vector_norm(v)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return v / norm


# =============================================================================
# Identity/Constant Quaternions
# =============================================================================

@jit(nopython=True, cache=True)
def q_identity() -> np.ndarray:
    """
    Return identity quaternion (no rotation).

    Returns:
        Identity quaternion [1, 0, 0, 0]
    """
    return np.array([1.0, 0.0, 0.0, 0.0])
