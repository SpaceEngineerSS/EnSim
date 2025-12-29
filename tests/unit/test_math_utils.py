"""
Unit tests for quaternion mathematics module.

Validates quaternion operations against known mathematical properties
and analytical solutions.

Reference:
    Diebel, J. "Representing Attitude: Euler Angles, Unit Quaternions,
    and Rotation Vectors"
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.math_utils import (
    q_mult,
    q_conjugate,
    q_normalize,
    q_inverse,
    q_norm,
    q_rotate_vector,
    q_from_axis_angle,
    q_to_axis_angle,
    q_from_euler,
    q_to_euler,
    q_derivative,
    q_integrate,
    q_to_rotation_matrix,
    q_identity,
    cross_product,
    dot_product,
    vector_norm,
    normalize_vector,
)


# =============================================================================
# Quaternion Basic Operations
# =============================================================================

class TestQuaternionBasics:
    """Test basic quaternion operations."""
    
    def test_q_identity(self):
        """Identity quaternion should be [1, 0, 0, 0]."""
        q = q_identity()
        assert_allclose(q, [1.0, 0.0, 0.0, 0.0])
    
    def test_q_norm_unit(self):
        """Unit quaternion should have norm = 1."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(q_norm(q) - 1.0) < 1e-10
    
    def test_q_normalize(self):
        """Normalization should produce unit quaternion."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_unit = q_normalize(q)
        assert abs(q_norm(q_unit) - 1.0) < 1e-10
        assert_allclose(q_unit, [1.0, 0.0, 0.0, 0.0])
    
    def test_q_conjugate(self):
        """Conjugate should negate vector part."""
        q = np.array([0.5, 0.1, 0.2, 0.3])
        q_conj = q_conjugate(q)
        assert_allclose(q_conj, [0.5, -0.1, -0.2, -0.3])


class TestQuaternionMultiplication:
    """Test quaternion multiplication (Hamilton product)."""
    
    def test_q_mult_identity(self):
        """q ⊗ I = q (identity is multiplicative identity)."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        I = q_identity()
        result = q_mult(q, I)
        assert_allclose(result, q, atol=1e-10)
    
    def test_q_mult_identity_left(self):
        """I ⊗ q = q."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        I = q_identity()
        result = q_mult(I, q)
        assert_allclose(result, q, atol=1e-10)
    
    def test_q_mult_inverse(self):
        """q ⊗ q⁻¹ = I (inverse property)."""
        q = q_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q_inv = q_inverse(q)
        result = q_mult(q, q_inv)
        assert_allclose(result, q_identity(), atol=1e-10)
    
    def test_q_mult_non_commutative(self):
        """q1 ⊗ q2 ≠ q2 ⊗ q1 (non-commutative)."""
        q1 = q_from_axis_angle(np.array([1., 0., 0.]), np.pi/4)
        q2 = q_from_axis_angle(np.array([0., 1., 0.]), np.pi/4)
        
        r1 = q_mult(q1, q2)
        r2 = q_mult(q2, q1)
        
        # Should NOT be equal
        assert not np.allclose(r1, r2)
    
    def test_q_mult_associative(self):
        """(q1 ⊗ q2) ⊗ q3 = q1 ⊗ (q2 ⊗ q3) (associative)."""
        q1 = q_from_axis_angle(np.array([1., 0., 0.]), np.pi/6)
        q2 = q_from_axis_angle(np.array([0., 1., 0.]), np.pi/4)
        q3 = q_from_axis_angle(np.array([0., 0., 1.]), np.pi/3)
        
        left = q_mult(q_mult(q1, q2), q3)
        right = q_mult(q1, q_mult(q2, q3))
        
        assert_allclose(left, right, atol=1e-10)


# =============================================================================
# Vector Rotation
# =============================================================================

class TestVectorRotation:
    """Test quaternion vector rotation."""
    
    def test_rotate_identity(self):
        """Identity quaternion should not rotate vector."""
        v = np.array([1.0, 2.0, 3.0])
        q = q_identity()
        v_rot = q_rotate_vector(q, v)
        assert_allclose(v_rot, v, atol=1e-10)
    
    def test_rotate_90deg_x(self):
        """Rotate [0, 1, 0] by 90° about X → [0, 0, 1]."""
        v = np.array([0.0, 1.0, 0.0])
        q = q_from_axis_angle(np.array([1., 0., 0.]), np.pi/2)
        v_rot = q_rotate_vector(q, v)
        expected = np.array([0.0, 0.0, 1.0])
        assert_allclose(v_rot, expected, atol=1e-10)
    
    def test_rotate_90deg_y(self):
        """Rotate [1, 0, 0] by 90° about Y → [0, 0, -1]."""
        v = np.array([1.0, 0.0, 0.0])
        q = q_from_axis_angle(np.array([0., 1., 0.]), np.pi/2)
        v_rot = q_rotate_vector(q, v)
        expected = np.array([0.0, 0.0, -1.0])
        assert_allclose(v_rot, expected, atol=1e-10)
    
    def test_rotate_90deg_z(self):
        """Rotate [1, 0, 0] by 90° about Z → [0, 1, 0]."""
        v = np.array([1.0, 0.0, 0.0])
        q = q_from_axis_angle(np.array([0., 0., 1.]), np.pi/2)
        v_rot = q_rotate_vector(q, v)
        expected = np.array([0.0, 1.0, 0.0])
        assert_allclose(v_rot, expected, atol=1e-10)
    
    def test_rotate_180deg(self):
        """Rotate [1, 0, 0] by 180° about Z → [-1, 0, 0]."""
        v = np.array([1.0, 0.0, 0.0])
        q = q_from_axis_angle(np.array([0., 0., 1.]), np.pi)
        v_rot = q_rotate_vector(q, v)
        expected = np.array([-1.0, 0.0, 0.0])
        assert_allclose(v_rot, expected, atol=1e-10)
    
    def test_rotate_preserves_magnitude(self):
        """Rotation should preserve vector magnitude."""
        v = np.array([1.0, 2.0, 3.0])
        q = q_from_axis_angle(np.array([1., 1., 1.]), np.pi/3)
        v_rot = q_rotate_vector(q, v)
        
        assert abs(vector_norm(v_rot) - vector_norm(v)) < 1e-10


# =============================================================================
# Euler Angle Conversions
# =============================================================================

class TestEulerConversion:
    """Test Euler angle ↔ quaternion conversions."""
    
    def test_euler_zero(self):
        """Zero Euler angles → Identity quaternion."""
        q = q_from_euler(0.0, 0.0, 0.0)
        assert_allclose(q, q_identity(), atol=1e-10)
    
    def test_euler_roundtrip_roll(self):
        """Roll only: Euler → Q → Euler should match."""
        roll = np.pi / 6
        q = q_from_euler(roll, 0.0, 0.0)
        euler = q_to_euler(q)
        assert abs(euler[0] - roll) < 1e-10
        assert abs(euler[1]) < 1e-10
        assert abs(euler[2]) < 1e-10
    
    def test_euler_roundtrip_pitch(self):
        """Pitch only: Euler → Q → Euler should match."""
        pitch = np.pi / 4
        q = q_from_euler(0.0, pitch, 0.0)
        euler = q_to_euler(q)
        assert abs(euler[0]) < 1e-10
        assert abs(euler[1] - pitch) < 1e-10
        assert abs(euler[2]) < 1e-10
    
    def test_euler_roundtrip_yaw(self):
        """Yaw only: Euler → Q → Euler should match."""
        yaw = np.pi / 3
        q = q_from_euler(0.0, 0.0, yaw)
        euler = q_to_euler(q)
        assert abs(euler[0]) < 1e-10
        assert abs(euler[1]) < 1e-10
        assert abs(euler[2] - yaw) < 1e-10
    
    def test_euler_roundtrip_combined(self):
        """Combined angles: Euler → Q → Euler should match."""
        roll, pitch, yaw = np.pi/6, np.pi/8, np.pi/5
        q = q_from_euler(roll, pitch, yaw)
        euler = q_to_euler(q)
        assert abs(euler[0] - roll) < 1e-10
        assert abs(euler[1] - pitch) < 1e-10
        assert abs(euler[2] - yaw) < 1e-10


# =============================================================================
# Axis-Angle Conversion
# =============================================================================

class TestAxisAngleConversion:
    """Test axis-angle ↔ quaternion conversions."""
    
    def test_axis_angle_identity(self):
        """Zero angle → Identity quaternion."""
        q = q_from_axis_angle(np.array([1., 0., 0.]), 0.0)
        assert_allclose(q, q_identity(), atol=1e-10)
    
    def test_axis_angle_roundtrip(self):
        """Axis-angle → Q → Axis-angle should match."""
        axis = normalize_vector(np.array([1.0, 2.0, 3.0]))
        angle = np.pi / 3
        
        q = q_from_axis_angle(axis, angle)
        result = q_to_axis_angle(q)
        axis_out = result[:3]
        angle_out = result[3]
        
        assert abs(angle_out - angle) < 1e-10
        assert_allclose(axis_out, axis, atol=1e-10)



# =============================================================================
# Quaternion Kinematics
# =============================================================================

class TestQuaternionKinematics:
    """Test quaternion derivative and integration."""
    
    def test_derivative_zero_omega(self):
        """Zero angular velocity → Zero quaternion derivative."""
        q = q_from_euler(0.1, 0.2, 0.3)
        omega = np.array([0.0, 0.0, 0.0])
        q_dot = q_derivative(q, omega)
        assert_allclose(q_dot, np.zeros(4), atol=1e-10)
    
    def test_integrate_preserves_norm(self):
        """Integration should preserve unit quaternion constraint."""
        q = q_from_euler(0.1, 0.2, 0.3)
        omega = np.array([0.1, 0.2, 0.3])  # rad/s
        dt = 0.01  # s
        
        q_new = q_integrate(q, omega, dt)
        
        assert abs(q_norm(q_new) - 1.0) < 1e-10


# =============================================================================
# Rotation Matrix
# =============================================================================

class TestRotationMatrix:
    """Test quaternion to rotation matrix conversion."""
    
    def test_identity_rotation_matrix(self):
        """Identity quaternion → Identity matrix."""
        q = q_identity()
        R = q_to_rotation_matrix(q)
        assert_allclose(R, np.eye(3), atol=1e-10)
    
    def test_rotation_matrix_orthogonal(self):
        """Rotation matrix should be orthogonal (R^T R = I)."""
        q = q_from_euler(0.1, 0.2, 0.3)
        R = q_to_rotation_matrix(q)
        RtR = R.T @ R
        assert_allclose(RtR, np.eye(3), atol=1e-10)
    
    def test_rotation_matrix_det_one(self):
        """Rotation matrix should have determinant = 1."""
        q = q_from_euler(0.1, 0.2, 0.3)
        R = q_to_rotation_matrix(q)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10
    
    def test_rotation_matrix_vs_quaternion(self):
        """Rotation via matrix should match quaternion rotation."""
        v = np.array([1.0, 2.0, 3.0])
        q = q_from_euler(0.3, 0.4, 0.5)
        
        # Rotate via quaternion
        v_q = q_rotate_vector(q, v)
        
        # Rotate via matrix
        R = q_to_rotation_matrix(q)
        v_R = R @ v
        
        assert_allclose(v_q, v_R, atol=1e-10)


# =============================================================================
# Vector Utilities
# =============================================================================

class TestVectorUtilities:
    """Test basic vector operations."""
    
    def test_cross_product(self):
        """Cross product of X and Y should be Z."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = cross_product(x, y)
        assert_allclose(z, [0.0, 0.0, 1.0], atol=1e-10)
    
    def test_dot_product(self):
        """Dot product of orthogonal vectors is zero."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        assert abs(dot_product(x, y)) < 1e-10
    
    def test_vector_norm(self):
        """Vector [3, 4, 0] has norm 5."""
        v = np.array([3.0, 4.0, 0.0])
        assert abs(vector_norm(v) - 5.0) < 1e-10
    
    def test_normalize_vector(self):
        """Normalized vector has unit length."""
        v = np.array([3.0, 4.0, 0.0])
        v_unit = normalize_vector(v)
        assert abs(vector_norm(v_unit) - 1.0) < 1e-10
