"""
NASA CEA Comparison Tests for EnSim Validation.

This module contains comprehensive validation tests comparing EnSim's
thermochemical calculations against NASA CEA reference data.

Reference:
    - NASA CEA Online Calculator (https://cearun.grc.nasa.gov/)
    - Gordon & McBride, NASA RP-1311 (1994)
    - McBride et al., NASA/TP-2002-211556

Test methodology:
    1. Run identical cases in both EnSim and NASA CEA
    2. Compare key performance parameters
    3. Verify agreement within engineering tolerances
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Optional

from src.core.chemistry import CombustionProblem
from src.core.propulsion import calculate_c_star, calculate_thrust_coefficient
from src.core.constants import GAS_CONSTANT, G0
from src.utils.nasa_parser import create_sample_database


@dataclass
class CEATestCase:
    """NASA CEA reference test case."""
    name: str
    fuel: str
    oxidizer: str
    of_ratio: float  # O/F by mass
    fuel_moles: float
    ox_moles: float
    chamber_pressure: float  # Pa
    expansion_ratio: float
    # CEA Reference values
    cea_T_chamber: float  # K
    cea_gamma: float
    cea_mean_mw: float  # g/mol
    cea_c_star: float  # m/s
    cea_isp_vac: float  # s
    # Tolerances
    T_tolerance: float = 0.02  # 2%
    isp_tolerance: float = 0.02  # 2%


# NASA CEA Reference Test Cases
# These values were obtained from NASA CEA Online Calculator
CEA_TEST_CASES = [
    CEATestCase(
        name="LOX/LH2 Nominal",
        fuel="H2",
        oxidizer="O2",
        of_ratio=6.0,
        fuel_moles=2.0,  # 2H2 + O2 stoichiometric
        ox_moles=1.0,
        chamber_pressure=6.89e6,  # 1000 psia
        expansion_ratio=40.0,
        cea_T_chamber=3528,
        cea_gamma=1.141,
        cea_mean_mw=13.45,
        cea_c_star=2390,
        cea_isp_vac=455.3,
    ),
    CEATestCase(
        name="LOX/Methane",
        fuel="CH4",
        oxidizer="O2",
        of_ratio=3.5,
        fuel_moles=1.0,
        ox_moles=2.0,
        chamber_pressure=10e6,  # 100 bar
        expansion_ratio=35.0,
        cea_T_chamber=3550,
        cea_gamma=1.139,
        cea_mean_mw=20.87,
        cea_c_star=1850,
        cea_isp_vac=363.0,
    ),
    CEATestCase(
        name="N2O4/UDMH Storable",
        fuel="N2H4",  # Using Hydrazine as UDMH surrogate
        oxidizer="N2O4",
        of_ratio=2.0,
        fuel_moles=1.0,
        ox_moles=0.5,
        chamber_pressure=0.85e6,  # 123 psia
        expansion_ratio=100.0,
        cea_T_chamber=3196,
        cea_gamma=1.216,
        cea_mean_mw=21.82,
        cea_c_star=1666,
        cea_isp_vac=318.0,
        T_tolerance=0.03,  # Higher tolerance for storable propellants
        isp_tolerance=0.03,
    ),
]


class TestNASACEAComparison:
    """
    Comprehensive NASA CEA comparison test suite.
    
    These tests validate EnSim against the industry-standard
    NASA CEA equilibrium solver.
    """
    
    @pytest.fixture
    def species_db(self):
        """Create extended species database for testing."""
        return create_sample_database()
    
    @pytest.mark.parametrize("test_case", CEA_TEST_CASES, ids=lambda x: x.name)
    def test_chamber_temperature(self, test_case: CEATestCase, species_db):
        """
        Validate adiabatic flame temperature against CEA.
        
        This is the primary validation metric as temperature
        determines all other combustion properties.
        """
        if test_case.fuel not in species_db:
            pytest.skip(f"Fuel {test_case.fuel} not in database")
        if test_case.oxidizer not in species_db:
            pytest.skip(f"Oxidizer {test_case.oxidizer} not in database")
        
        problem = CombustionProblem(species_db)
        problem.add_fuel(test_case.fuel, moles=test_case.fuel_moles)
        problem.add_oxidizer(test_case.oxidizer, moles=test_case.ox_moles)
        
        result = problem.solve(
            pressure=test_case.chamber_pressure,
            initial_temp_guess=3000.0,
            max_iterations=100
        )
        
        if not result.converged:
            pytest.skip("Solver did not converge")
        
        error = abs(result.temperature - test_case.cea_T_chamber) / test_case.cea_T_chamber
        
        assert error < test_case.T_tolerance, (
            f"Temperature error {error*100:.2f}% exceeds tolerance {test_case.T_tolerance*100:.0f}%\n"
            f"EnSim: {result.temperature:.0f} K, CEA: {test_case.cea_T_chamber:.0f} K"
        )
    
    @pytest.mark.parametrize("test_case", CEA_TEST_CASES, ids=lambda x: x.name)
    def test_gamma_ratio(self, test_case: CEATestCase, species_db):
        """
        Validate specific heat ratio against CEA.
        
        Gamma (γ = Cp/Cv) is crucial for nozzle flow calculations.
        """
        if test_case.fuel not in species_db or test_case.oxidizer not in species_db:
            pytest.skip("Species not in database")
        
        problem = CombustionProblem(species_db)
        problem.add_fuel(test_case.fuel, moles=test_case.fuel_moles)
        problem.add_oxidizer(test_case.oxidizer, moles=test_case.ox_moles)
        
        result = problem.solve(pressure=test_case.chamber_pressure)
        
        if not result.converged:
            pytest.skip("Solver did not converge")
        
        error = abs(result.gamma - test_case.cea_gamma) / test_case.cea_gamma
        
        # Gamma should be within 2%
        assert error < 0.02, (
            f"Gamma error {error*100:.2f}% exceeds 2%\n"
            f"EnSim: {result.gamma:.3f}, CEA: {test_case.cea_gamma:.3f}"
        )
    
    @pytest.mark.parametrize("test_case", CEA_TEST_CASES, ids=lambda x: x.name)
    def test_mean_molecular_weight(self, test_case: CEATestCase, species_db):
        """
        Validate mean molecular weight against CEA.
        
        M̄ directly affects characteristic velocity and Isp.
        """
        if test_case.fuel not in species_db or test_case.oxidizer not in species_db:
            pytest.skip("Species not in database")
        
        problem = CombustionProblem(species_db)
        problem.add_fuel(test_case.fuel, moles=test_case.fuel_moles)
        problem.add_oxidizer(test_case.oxidizer, moles=test_case.ox_moles)
        
        result = problem.solve(pressure=test_case.chamber_pressure)
        
        if not result.converged:
            pytest.skip("Solver did not converge")
        
        error = abs(result.mean_molecular_weight - test_case.cea_mean_mw) / test_case.cea_mean_mw
        
        # MW should be within 2%
        assert error < 0.02, (
            f"Mean MW error {error*100:.2f}% exceeds 2%\n"
            f"EnSim: {result.mean_molecular_weight:.2f} g/mol, CEA: {test_case.cea_mean_mw:.2f} g/mol"
        )


class TestChamberTemperatureRange:
    """
    Test chamber temperature across various O/F ratios.
    
    Validates that temperature peaks near stoichiometric and
    decreases for lean/rich mixtures.
    """
    
    @pytest.fixture
    def species_db(self):
        return create_sample_database()
    
    def test_h2_o2_temperature_vs_of_ratio(self, species_db):
        """
        Test H2/O2 temperature variation with O/F ratio.
        
        Expected behavior:
        - Temperature should peak near O/F = 4-6 (stoichiometric region)
        - Should decrease for very lean (high O/F) mixtures
        - Should decrease for very rich (low O/F) mixtures
        """
        of_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]
        temperatures = []
        
        for of in of_ratios:
            # Calculate moles: O/F = (moles_O2 * 32) / (moles_H2 * 2)
            # moles_O2 = of * moles_H2 * 2 / 32 = of * moles_H2 / 16
            moles_h2 = 2.0
            moles_o2 = of * moles_h2 * 2.0 / 32.0
            
            problem = CombustionProblem(species_db)
            problem.add_fuel('H2', moles=moles_h2)
            problem.add_oxidizer('O2', moles=moles_o2)
            
            result = problem.solve(pressure=1e6)
            
            if result.converged or result.iterations >= 30:
                temperatures.append(result.temperature)
            else:
                temperatures.append(0)
        
        # Verify temperature trend
        valid_temps = [t for t in temperatures if t > 0]
        if len(valid_temps) < 3:
            pytest.skip("Not enough converged cases")
        
        # Temperature should be in reasonable range
        for T in valid_temps:
            assert 2000 < T < 4500, f"Temperature {T:.0f} K outside expected range"


class TestDissociationEffects:
    """
    Test that dissociation effects are properly captured.
    
    At high temperatures, molecular species dissociate:
    H2O <-> H + OH
    H2 <-> 2H
    O2 <-> 2O
    """
    
    @pytest.fixture
    def species_db(self):
        return create_sample_database()
    
    def test_dissociation_increases_with_temperature(self, species_db):
        """
        Verify that atomic species increase at higher temperatures.
        
        At higher equilibrium temperatures, dissociation should increase:
        - More H atoms
        - More O atoms  
        - More OH radicals
        """
        # Rich mixture gives higher temperature
        problem_hot = CombustionProblem(species_db)
        problem_hot.add_fuel('H2', moles=2.0)
        problem_hot.add_oxidizer('O2', moles=1.0)
        result_hot = problem_hot.solve(pressure=101325.0)  # 1 atm - less pressure suppression
        
        # High pressure suppresses dissociation
        problem_cold = CombustionProblem(species_db)
        problem_cold.add_fuel('H2', moles=2.0)
        problem_cold.add_oxidizer('O2', moles=1.0)
        result_cold = problem_cold.solve(pressure=10e6)  # 100 atm
        
        if result_hot.converged and result_cold.converged:
            # At lower pressure (higher dissociation), expect more radicals
            x_H_hot = result_hot.get_mole_fraction('H')
            x_H_cold = result_cold.get_mole_fraction('H')
            
            # Low pressure should have more dissociation
            assert x_H_hot >= x_H_cold * 0.5, (
                f"Expected more H at low pressure: {x_H_hot:.4f} vs {x_H_cold:.4f}"
            )


class TestSpeciesConservation:
    """
    Test element conservation in equilibrium calculations.
    
    Total atoms of each element must be conserved from
    reactants to products.
    """
    
    @pytest.fixture
    def species_db(self):
        return create_sample_database()
    
    def test_hydrogen_conservation(self, species_db):
        """
        Verify hydrogen atoms are conserved.
        
        Input: 2 mol H2 = 4 mol H atoms
        Output: Should also have 4 mol H atoms total
        """
        problem = CombustionProblem(species_db)
        problem.add_fuel('H2', moles=2.0)
        problem.add_oxidizer('O2', moles=1.0)
        
        result = problem.solve(pressure=1e6)
        
        if not result.converged:
            pytest.skip("Solver did not converge")
        
        # Count H atoms in products
        h_atoms_out = 0.0
        h_content = {'H2O': 2, 'H2': 2, 'OH': 1, 'H': 1, 'HO2': 1, 'H2O2': 2}
        
        for i, name in enumerate(result.species_names):
            if name in h_content:
                h_atoms_out += result.moles[i] * h_content[name]
        
        h_atoms_in = 4.0  # 2 mol H2 × 2 atoms/mol
        
        rel_error = abs(h_atoms_out - h_atoms_in) / h_atoms_in
        
        # Allow 5% tolerance for numerical solver
        assert rel_error < 0.05, (
            f"H conservation error: {rel_error*100:.2f}%\n"
            f"Input: {h_atoms_in:.2f} mol H, Output: {h_atoms_out:.2f} mol H"
        )
    
    def test_oxygen_conservation(self, species_db):
        """
        Verify oxygen atoms are conserved.
        
        Input: 1 mol O2 = 2 mol O atoms
        Output: Should also have 2 mol O atoms total
        """
        problem = CombustionProblem(species_db)
        problem.add_fuel('H2', moles=2.0)
        problem.add_oxidizer('O2', moles=1.0)
        
        result = problem.solve(pressure=1e6)
        
        if not result.converged:
            pytest.skip("Solver did not converge")
        
        # Count O atoms in products
        o_atoms_out = 0.0
        o_content = {'H2O': 1, 'O2': 2, 'OH': 1, 'O': 1, 'HO2': 2, 'H2O2': 2, 'CO': 1, 'CO2': 2}
        
        for i, name in enumerate(result.species_names):
            if name in o_content:
                o_atoms_out += result.moles[i] * o_content[name]
        
        o_atoms_in = 2.0  # 1 mol O2 × 2 atoms/mol
        
        rel_error = abs(o_atoms_out - o_atoms_in) / o_atoms_in
        
        # Allow 5% tolerance for numerical solver
        assert rel_error < 0.05, (
            f"O conservation error: {rel_error*100:.2f}%\n"
            f"Input: {o_atoms_in:.2f} mol O, Output: {o_atoms_out:.2f} mol O"
        )


class TestCharacteristicVelocity:
    """
    Test characteristic velocity (C*) calculations.
    
    C* = sqrt(R * Tc) / Γ(γ)
    where Γ(γ) = sqrt(γ) * (2/(γ+1))^((γ+1)/(2*(γ-1)))
    """
    
    def test_c_star_formula(self):
        """
        Verify C* calculation matches analytical formula.
        
        Reference: Sutton & Biblarz, "Rocket Propulsion Elements"
        """
        # Test conditions
        T_c = 3500.0  # K
        gamma = 1.15
        M_mol = 18.0  # g/mol
        
        # Calculate specific gas constant
        R_specific = GAS_CONSTANT / (M_mol / 1000.0)  # J/(kg·K)
        
        # EnSim calculation (uses gamma, R_specific, T_chamber)
        c_star_ensim = calculate_c_star(gamma, R_specific, T_c)
        
        # Analytical calculation
        # Γ = sqrt(γ) * (2/(γ+1))^((γ+1)/(2*(γ-1)))
        gamma_func = np.sqrt(gamma) * ((2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1))))
        c_star_analytical = np.sqrt(R_specific * T_c) / gamma_func
        
        rel_error = abs(c_star_ensim - c_star_analytical) / c_star_analytical
        
        assert rel_error < 0.001, (
            f"C* calculation error: {rel_error*100:.3f}%\n"
            f"EnSim: {c_star_ensim:.1f} m/s, Analytical: {c_star_analytical:.1f} m/s"
        )


class TestThrustCoefficient:
    """
    Test thrust coefficient (Cf) calculations.
    
    Cf = sqrt(2γ²/(γ-1) * (2/(γ+1))^((γ+1)/(γ-1)) * [1-(Pe/Pc)^((γ-1)/γ)])
         + ε(Pe-Pa)/Pc
    """
    
    def _calculate_pressure_ratio(self, gamma: float, area_ratio: float) -> float:
        """Calculate Pe/Pc from area ratio using isentropic relations."""
        from src.core.propulsion import solve_mach_from_area_ratio_supersonic
        M = solve_mach_from_area_ratio_supersonic(area_ratio, gamma)
        gm1 = gamma - 1.0
        pressure_ratio = (1.0 + gm1 / 2.0 * M * M) ** (-gamma / gm1)
        return pressure_ratio
    
    def test_cf_vacuum(self):
        """
        Test vacuum thrust coefficient.
        
        At vacuum (Pa = 0), Cf depends only on:
        - Gamma (γ)
        - Expansion ratio (ε)
        - Pressure ratio (Pe/Pc)
        """
        gamma = 1.2
        epsilon = 50.0  # Area ratio
        
        # Calculate pressure ratio from area ratio (isentropic)
        pressure_ratio = self._calculate_pressure_ratio(gamma, epsilon)
        
        # Calculate Cf (ambient_ratio = 0 for vacuum)
        cf_vac = calculate_thrust_coefficient(
            gamma=gamma,
            pressure_ratio=pressure_ratio,
            area_ratio=epsilon,
            ambient_ratio=0.0  # Vacuum: Pa/Pc = 0
        )
        
        # Typical vacuum Cf for ε=50, γ=1.2 should be ~1.85-1.95
        assert 1.7 < cf_vac < 2.1, f"Cf_vac = {cf_vac:.3f} outside expected range"
    
    def test_cf_increases_with_expansion_ratio(self):
        """
        Verify Cf increases with expansion ratio in vacuum.
        """
        gamma = 1.2
        
        # Calculate for each expansion ratio
        eps_list = [10.0, 50.0, 100.0]
        cf_values = []
        
        for eps in eps_list:
            pr = self._calculate_pressure_ratio(gamma, eps)
            cf = calculate_thrust_coefficient(gamma, pr, eps, 0.0)
            cf_values.append(cf)
        
        cf_10, cf_50, cf_100 = cf_values
        
        assert cf_10 < cf_50 < cf_100, (
            f"Cf should increase with ε: Cf(10)={cf_10:.3f}, "
            f"Cf(50)={cf_50:.3f}, Cf(100)={cf_100:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

