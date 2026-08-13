import numpy as np
import pytest
from numpy.testing import assert_allclose

from ensim.core.chemistry import (
    GAS_CONSTANT,
    bilinear_interpolate,
    create_combustion_lookup_table,
    load_nasa_thermo_dat,
    lookup_combustion_properties,
    nasa_get_cp_r,
)
from ensim.utils.nasa_parser import create_sample_database


def test_packaged_thermodynamic_data_are_available():
    database = load_nasa_thermo_dat()
    assert {"H2", "O2", "H2O"} <= database.keys()
    h2 = database["H2"]
    cp = nasa_get_cp_r(1000.0, h2["coeffs_low"], h2["coeffs_high"], h2["T_mid"])
    assert cp * GAS_CONSTANT == pytest.approx(30.16, rel=0.02)


def test_bilinear_interpolation_reproduces_nodes_and_midpoint():
    x_grid = np.array([0.0, 1.0])
    y_grid = np.array([0.0, 1.0])
    values = np.array([[0.0, 2.0], [2.0, 4.0]])
    assert bilinear_interpolate(0.0, 0.0, x_grid, y_grid, values) == 0.0
    assert bilinear_interpolate(0.5, 0.5, x_grid, y_grid, values) == pytest.approx(2.0)


@pytest.fixture(scope="module")
def equilibrium_table():
    return create_combustion_lookup_table(
        np.array([4.0, 6.0]),
        np.array([1e6, 5e6]),
        species_db=create_sample_database(),
    )


def test_lookup_table_is_built_from_converged_equilibrium(equilibrium_table):
    assert equilibrium_table["T_chamber"].shape == (2, 2)
    assert np.all(
        (equilibrium_table["T_chamber"] > 2000.0) & (equilibrium_table["T_chamber"] < 4500.0)
    )
    assert np.all((equilibrium_table["gamma"] > 1.0) & (equilibrium_table["gamma"] < 1.5))


def test_lookup_interpolation_reproduces_a_grid_point(equilibrium_table):
    result = lookup_combustion_properties(
        4.0,
        1e6,
        equilibrium_table["of_grid"],
        equilibrium_table["Pc_grid"],
        equilibrium_table["T_chamber"],
        equilibrium_table["gamma"],
        equilibrium_table["M_mol"],
    )
    expected = (
        equilibrium_table["T_chamber"][0, 0],
        equilibrium_table["gamma"][0, 0],
        equilibrium_table["M_mol"][0, 0],
    )
    assert_allclose(result, expected)


def test_empirical_lookup_fallback_is_rejected():
    with pytest.raises(ValueError, match="empirical"):
        create_combustion_lookup_table(
            np.array([4.0, 6.0]),
            np.array([1e6, 5e6]),
            use_equilibrium_solver=False,
        )
