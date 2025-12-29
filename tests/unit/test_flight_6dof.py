"""
Unit tests for 6-DOF flight simulator.

Tests the 6-DOF rigid body dynamics solver against known analytical
solutions and basic physics validation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.flight_6dof import (
    FlightResult6DOF,
    PhysicsViolationError,
    calculate_inertia_tensor_cylindrical,
    calculate_inertia_principal,
    calculate_gravity_vector,
    calculate_aero_angles,
    calculate_aerodynamic_moment,
    derivatives_6dof,
    rk4_step_6dof,
    simulate_flight_6dof,
    G0,
    R_EARTH,
)
from src.core.math_utils import q_identity, q_from_euler, q_normalize
from src.core.rocket import Rocket


# =============================================================================
# Inertia Tensor Tests
# =============================================================================

class TestInertiaTensor:
    """Test inertia tensor calculations."""
    
    def test_cylindrical_inertia_formula(self):
        """Test cylindrical inertia against analytical formula."""
        mass = 100.0  # kg
        length = 2.0  # m
        radius = 0.1  # m
        
        I = calculate_inertia_tensor_cylindrical(mass, length, radius)
        
        # Expected values
        Ixx_expected = 0.5 * mass * radius**2  # Roll
        Iyy_expected = mass * (3*radius**2 + length**2) / 12.0  # Pitch
        
        assert abs(I[0, 0] - Ixx_expected) < 1e-10
        assert abs(I[1, 1] - Iyy_expected) < 1e-10
        assert abs(I[2, 2] - Iyy_expected) < 1e-10  # Symmetric
    
    def test_inertia_principal_matches_tensor(self):
        """Principal moments should match tensor diagonal."""
        mass = 50.0
        length = 1.5
        radius = 0.08
        
        I_tensor = calculate_inertia_tensor_cylindrical(mass, length, radius)
        I_principal = calculate_inertia_principal(mass, length, radius)
        
        assert_allclose([I_tensor[0,0], I_tensor[1,1], I_tensor[2,2]], 
                       I_principal, atol=1e-10)
    
    def test_inertia_scales_with_mass(self):
        """Inertia should scale linearly with mass."""
        length = 2.0
        radius = 0.1
        
        I1 = calculate_inertia_principal(10.0, length, radius)
        I2 = calculate_inertia_principal(20.0, length, radius)
        
        assert_allclose(I2, 2.0 * I1, atol=1e-10)
    
    def test_inertia_decreases_with_fuel_burn(self):
        """Simulates inertia decrease as fuel depletes."""
        length = 2.0
        radius = 0.1
        
        I_full = calculate_inertia_principal(100.0, length, radius)
        I_empty = calculate_inertia_principal(30.0, length, radius)  # 70% fuel burned
        
        # All moments should decrease
        assert all(I_empty < I_full)


# =============================================================================
# Gravity Model Tests
# =============================================================================

class TestGravityModel:
    """Test gravity vector calculations."""
    
    def test_gravity_at_sea_level(self):
        """Gravity at sea level should be g0."""
        g = calculate_gravity_vector(0.0)
        assert abs(g[2] + G0) < 1e-6  # Negative Z (pointing down)
        assert abs(g[0]) < 1e-10  # No horizontal component
        assert abs(g[1]) < 1e-10
    
    def test_gravity_decreases_with_altitude(self):
        """Gravity magnitude should decrease with altitude."""
        g_0 = calculate_gravity_vector(0.0)
        g_100km = calculate_gravity_vector(100000.0)
        g_400km = calculate_gravity_vector(400000.0)  # ISS altitude
        
        assert abs(g_0[2]) > abs(g_100km[2])
        assert abs(g_100km[2]) > abs(g_400km[2])
    
    def test_gravity_inverse_square_law(self):
        """Verify inverse square law."""
        g0 = calculate_gravity_vector(0.0)
        
        # At altitude h, g = g0 * (R/(R+h))^2
        h = 1000000.0  # 1000 km
        g_h = calculate_gravity_vector(h)
        
        ratio = (R_EARTH / (R_EARTH + h))**2
        expected = abs(g0[2]) * ratio
        
        assert abs(abs(g_h[2]) - expected) < 1e-6


# =============================================================================
# Aerodynamic Angle Tests
# =============================================================================

class TestAeroAngles:
    """Test angle of attack and sideslip calculations."""
    
    def test_zero_aoa_forward_flight(self):
        """Forward flight should have zero AoA."""
        v_body = np.array([100.0, 0.0, 0.0])  # Pure forward velocity
        alpha, beta = calculate_aero_angles(v_body)
        
        assert abs(alpha) < 1e-10
        assert abs(beta) < 1e-10
    
    def test_positive_aoa(self):
        """Downward velocity component creates positive AoA."""
        v_body = np.array([100.0, 0.0, -10.0])  # Velocity below nose
        alpha, beta = calculate_aero_angles(v_body)
        
        # Positive AoA when velocity is below the nose
        assert alpha > 0
    
    def test_sideslip(self):
        """Sideward velocity creates sideslip."""
        v_body = np.array([100.0, 10.0, 0.0])  # Velocity to the right
        alpha, beta = calculate_aero_angles(v_body)
        
        # Positive sideslip when velocity is to the right
        assert beta > 0


# =============================================================================
# Aerodynamic Moment Tests
# =============================================================================

class TestAeroMoments:
    """Test aerodynamic moment calculations."""
    
    def test_moment_direction(self):
        """CP ahead of CG should create restoring moment."""
        cp_pos = 0.5  # CP closer to nose
        cg_pos = 1.0  # CG further back
        
        # Lift force in +Y direction (body frame)
        F_aero = np.array([0.0, 100.0, 0.0])
        
        moment = calculate_aerodynamic_moment(cp_pos, cg_pos, F_aero)
        
        # Moment should be about Z-axis (yaw)
        assert abs(moment[2]) > 0
    
    def test_zero_moment_aligned_cp_cg(self):
        """No moment when CP = CG."""
        cp_pos = 1.0
        cg_pos = 1.0
        F_aero = np.array([0.0, 100.0, 50.0])
        
        moment = calculate_aerodynamic_moment(cp_pos, cg_pos, F_aero)
        
        assert_allclose(moment, [0.0, 0.0, 0.0], atol=1e-10)


# =============================================================================
# RK4 Integration Tests
# =============================================================================

class TestRK4Integration:
    """Test RK4 integrator."""
    
    def test_free_fall_analytical(self):
        """Test free fall against analytical solution: z = -½gt²."""
        # Initial state: stationary at 1000m
        state = np.array([
            0.0, 0.0, 1000.0,  # Position
            0.0, 0.0, 0.0,      # Velocity
            1.0, 0.0, 0.0, 0.0, # Identity quaternion
            0.0, 0.0, 0.0       # Angular velocity
        ])
        
        mass = 10.0
        I_principal = np.array([0.1, 1.0, 1.0])
        F_thrust = np.array([0.0, 0.0, 0.0])
        F_aero = np.array([0.0, 0.0, 0.0])
        M_aero = np.array([0.0, 0.0, 0.0])
        M_thrust = np.array([0.0, 0.0, 0.0])
        
        dt = 0.001  # Small time step for accuracy
        t_total = 1.0  # 1 second
        n_steps = int(t_total / dt)
        
        for _ in range(n_steps):
            state = rk4_step_6dof(
                state, dt, mass, I_principal,
                F_thrust, F_aero, M_aero, M_thrust, state[2]
            )
        
        # Analytical solution: z = z0 - ½gt²
        z_expected = 1000.0 - 0.5 * G0 * t_total**2
        
        # Should be within 0.1% of analytical
        error = abs(state[2] - z_expected) / z_expected
        assert error < 0.001, f"Free fall error: {error*100:.2f}%"
    
    def test_quaternion_normalized_after_step(self):
        """Quaternion should remain normalized after RK4 step."""
        state = np.array([
            0.0, 0.0, 1000.0,
            100.0, 0.0, 50.0,
            0.707, 0.707, 0.0, 0.0,  # Non-trivial quaternion
            0.1, 0.2, 0.3            # Non-zero angular velocity
        ])
        
        mass = 10.0
        I_principal = np.array([0.1, 1.0, 1.0])
        F_thrust = np.array([1000.0, 0.0, 0.0])
        F_aero = np.array([-50.0, 10.0, 5.0])
        M_aero = np.array([0.1, 0.2, 0.1])
        M_thrust = np.array([0.0, 0.0, 0.0])
        
        # Run several steps
        for _ in range(100):
            state = rk4_step_6dof(
                state, 0.01, mass, I_principal,
                F_thrust, F_aero, M_aero, M_thrust, state[2]
            )
        
        # Check quaternion is still normalized
        q = state[6:10]
        q_norm = np.sqrt(sum(q**2))
        assert abs(q_norm - 1.0) < 1e-10


# =============================================================================
# Physics Violation Tests
# =============================================================================

class TestPhysicsViolations:
    """Test physics violation error handling."""
    
    def test_negative_thrust_raises(self):
        """Negative thrust should raise PhysicsViolationError."""
        rocket = Rocket()
        
        with pytest.raises(PhysicsViolationError, match="Negative thrust"):
            simulate_flight_6dof(
                rocket,
                thrust_vac=-1000.0,  # Invalid
                isp_vac=300.0,
                burn_time=10.0
            )
    
    def test_zero_isp_raises(self):
        """Zero Isp should raise PhysicsViolationError."""
        rocket = Rocket()
        
        with pytest.raises(PhysicsViolationError, match="Invalid Isp"):
            simulate_flight_6dof(
                rocket,
                thrust_vac=1000.0,
                isp_vac=0.0,  # Invalid
                burn_time=10.0
            )


# =============================================================================
# Full Simulation Tests
# =============================================================================

class TestFullSimulation:
    """Integration tests for complete simulation."""
    
    @pytest.fixture
    def basic_rocket(self):
        """Create a simple rocket for testing."""
        rocket = Rocket()
        # Set up engine mount with propellant
        rocket.engine.fuel_mass = 5.0
        rocket.engine.oxidizer_mass = 15.0
        return rocket
    
    def test_simulation_returns_result(self, basic_rocket):
        """Simulation should return FlightResult6DOF object."""
        result = simulate_flight_6dof(
            basic_rocket,
            thrust_vac=5000.0,
            isp_vac=250.0,
            burn_time=5.0,
            max_time=30.0,
            dt=0.1
        )
        
        assert isinstance(result, FlightResult6DOF)
        assert result.success
    
    def test_simulation_altitude_positive(self, basic_rocket):
        """Rocket should gain altitude."""
        result = simulate_flight_6dof(
            basic_rocket,
            thrust_vac=5000.0,
            isp_vac=250.0,
            burn_time=5.0,
            max_time=30.0,
            dt=0.1
        )
        
        assert result.apogee_altitude > 0, "Rocket should reach positive apogee"
    
    def test_quaternion_always_normalized(self, basic_rocket):
        """Quaternion should stay normalized throughout flight."""
        result = simulate_flight_6dof(
            basic_rocket,
            thrust_vac=5000.0,
            isp_vac=250.0,
            burn_time=5.0,
            max_time=30.0,
            dt=0.1
        )
        
        # Check norm of quaternion at each time step
        for i in range(len(result.time)):
            q_norm = np.sqrt(
                result.quaternion_w[i]**2 +
                result.quaternion_x[i]**2 +
                result.quaternion_y[i]**2 +
                result.quaternion_z[i]**2
            )
            assert abs(q_norm - 1.0) < 1e-6, f"Quaternion not normalized at t={result.time[i]:.2f}s"
    
    def test_mass_decreases_during_burn(self, basic_rocket):
        """Mass should decrease during burn."""
        result = simulate_flight_6dof(
            basic_rocket,
            thrust_vac=5000.0,
            isp_vac=250.0,
            burn_time=5.0,
            max_time=30.0,
            dt=0.1
        )
        
        initial_mass = result.mass[0]
        burnout_idx = int(5.0 / 0.1)  # Approximate burnout index
        burnout_mass = result.mass[min(burnout_idx, len(result.mass)-1)]
        
        assert burnout_mass < initial_mass, "Mass should decrease during burn"
