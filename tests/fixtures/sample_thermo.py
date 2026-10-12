"""Hand-entered NASA Glenn coefficients for a handful of species, used by tests
that must not depend on the packaged data file."""

import numpy as np

from ensim.core.types import SpeciesData, SpeciesDatabase


def create_sample_database() -> SpeciesDatabase:
    """
    Create a small sample database with verified coefficients.

    These coefficients match NASA CEA reference data for common species.
    Useful for testing when full database file is not available.

    Reference: NASA/TP-2002-211556
    """
    db: SpeciesDatabase = {}

    # H2O (Water vapor) - from NASA Glenn database
    db["H2O"] = SpeciesData(
        name="H2O",
        molecular_weight=18.01528,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                2.67703787e00,  # a1
                2.97318329e-03,  # a2
                -7.73769690e-07,  # a3
                9.44336689e-11,  # a4
                -4.26900959e-15,  # a5
                -2.98858938e04,  # a6 (H/R integration constant)
                6.88255571e00,  # a7 (S/R integration constant)
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                4.19864056e00,  # a1
                -2.03643410e-03,  # a2
                6.52040211e-06,  # a3
                -5.48797062e-09,  # a4
                1.77197817e-12,  # a5
                -3.02937267e04,  # a6
                -8.49032208e-01,  # a7
            ],
            dtype=np.float64,
        ),
        h_formation_298=-241826.0,  # J/mol
    )

    # O2 (Oxygen) - from NASA Glenn database
    db["O2"] = SpeciesData(
        name="O2",
        molecular_weight=31.9988,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                3.66096065e00,
                6.56365811e-04,
                -1.41149627e-07,
                2.05797935e-11,
                -1.29913436e-15,
                -1.21597718e03,
                3.41536279e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.78245636e00,
                -2.99673416e-03,
                9.84730201e-06,
                -9.68129509e-09,
                3.24372837e-12,
                -1.06394356e03,
                3.65767573e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=0.0,
    )

    # H2 (Hydrogen) - from NASA Glenn database
    db["H2"] = SpeciesData(
        name="H2",
        molecular_weight=2.01588,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                2.93286575e00,
                8.26608026e-04,
                -1.46402364e-07,
                1.54100414e-11,
                -6.88804800e-16,
                -8.13065581e02,
                -1.02432865e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                2.34433112e00,
                7.98052075e-03,
                -1.94781510e-05,
                2.01572094e-08,
                -7.37611761e-12,
                -9.17935173e02,
                6.83010238e-01,
            ],
            dtype=np.float64,
        ),
        h_formation_298=0.0,
    )

    # N2 (Nitrogen) - from NASA Glenn database
    db["N2"] = SpeciesData(
        name="N2",
        molecular_weight=28.0134,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                2.95257637e00,
                1.39690040e-03,
                -4.92631603e-07,
                7.86010195e-11,
                -4.60755204e-15,
                -9.23948688e02,
                5.87188762e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.53100528e00,
                -1.23660988e-04,
                -5.02999433e-07,
                2.43530612e-09,
                -1.40881235e-12,
                -1.04697628e03,
                2.96747038e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=0.0,
    )

    # CO2 (Carbon Dioxide) - from NASA Glenn database
    db["CO2"] = SpeciesData(
        name="CO2",
        molecular_weight=44.0095,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                4.63659493e00,
                2.74131991e-03,
                -9.95828531e-07,
                1.60373011e-10,
                -9.16103468e-15,
                -4.90249341e04,
                -1.93534855e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                2.35677352e00,
                8.98459677e-03,
                -7.12356269e-06,
                2.45919022e-09,
                -1.43699548e-13,
                -4.83719697e04,
                9.90105222e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=-393510.0,  # J/mol
    )

    # CO (Carbon Monoxide) - from NASA Glenn database
    db["CO"] = SpeciesData(
        name="CO",
        molecular_weight=28.0101,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                3.04848583e00,
                1.35172818e-03,
                -4.85794075e-07,
                7.88536486e-11,
                -4.69807489e-15,
                -1.42661171e04,
                6.01709790e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.57953347e00,
                -6.10353680e-04,
                1.01681433e-06,
                9.07005884e-10,
                -9.04424499e-13,
                -1.43440860e04,
                3.50840928e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=-110530.0,  # J/mol
    )

    # OH (Hydroxyl radical) - from NASA Glenn database
    db["OH"] = SpeciesData(
        name="OH",
        molecular_weight=17.00734,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                2.83864607e00,
                1.10725586e-03,
                -2.93914978e-07,
                4.20524247e-11,
                -2.42169092e-15,
                3.69780808e03,
                5.84452662e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.99198424e00,
                -2.40106655e-03,
                4.61664033e-06,
                -3.87916306e-09,
                1.36319502e-12,
                3.36889836e03,
                -1.03998477e-01,
            ],
            dtype=np.float64,
        ),
        h_formation_298=38987.0,  # J/mol
    )

    # CH4 (Methane) - from NASA Glenn database
    db["CH4"] = SpeciesData(
        name="CH4",
        molecular_weight=16.04246,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                1.65326226e00,
                1.00263099e-02,
                -3.31661238e-06,
                5.36483138e-10,
                -3.14696758e-14,
                -1.00095936e04,
                9.90506283e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                5.14987613e00,
                -1.36709788e-02,
                4.91800599e-05,
                -4.84743026e-08,
                1.66693956e-11,
                -1.02466476e04,
                -4.64130376e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=-74600.0,  # J/mol
    )

    # H (Atomic Hydrogen) - from NASA Glenn database
    # Critical for high-temperature dissociation
    db["H"] = SpeciesData(
        name="H",
        molecular_weight=1.00794,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                2.50000286e00,  # a1
                -5.65334214e-09,  # a2
                3.63251723e-12,  # a3
                -9.19949720e-16,  # a4
                7.95260746e-20,  # a5
                2.54736589e04,  # a6
                -4.46698494e-01,  # a7
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                2.50000000e00,  # a1
                0.00000000e00,  # a2
                0.00000000e00,  # a3
                0.00000000e00,  # a4
                0.00000000e00,  # a5
                2.54736599e04,  # a6 (H/R = 25474 K means Hf = 218 kJ/mol)
                -4.46682853e-01,  # a7
            ],
            dtype=np.float64,
        ),
        h_formation_298=217998.0,  # J/mol (218 kJ/mol)
    )

    # O (Atomic Oxygen) - from NASA Glenn database
    # Critical for high-temperature dissociation
    db["O"] = SpeciesData(
        name="O",
        molecular_weight=15.9994,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                2.54363697e00,  # a1
                -2.73162486e-05,  # a2
                -4.19029520e-09,  # a3
                4.95481845e-12,  # a4
                -4.79553694e-16,  # a5
                2.92260120e04,  # a6
                4.92229457e00,  # a7
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.16826710e00,  # a1
                -3.27931884e-03,  # a2
                6.64306396e-06,  # a3
                -6.12806624e-09,  # a4
                2.11265971e-12,  # a5
                2.91222592e04,  # a6 (H/R = 29122 K means Hf = 249 kJ/mol)
                2.05193346e00,  # a7
            ],
            dtype=np.float64,
        ),
        h_formation_298=249175.0,  # J/mol (249 kJ/mol)
    )

    # Additional Propellants for Extended Simulation
    # N2O4 (Nitrogen Tetroxide) - Storable oxidizer
    # From NASA CEA database
    db["N2O4"] = SpeciesData(
        name="N2O4",
        molecular_weight=92.011,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                1.15752932e01,
                4.01615532e-03,
                -1.57178022e-06,
                2.68273657e-10,
                -1.66921538e-14,
                -1.08238125e03,
                -3.12993556e01,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.02002933e00,
                2.95904306e-02,
                -3.01342572e-05,
                1.42360485e-08,
                -2.42361443e-12,
                -6.79238789e02,
                1.18695821e01,
            ],
            dtype=np.float64,
        ),
        h_formation_298=9160.0,  # J/mol
    )

    # NO2 (Nitrogen Dioxide) - Decomposition product
    db["NO2"] = SpeciesData(
        name="NO2",
        molecular_weight=46.0055,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                4.88475400e00,
                2.17239555e-03,
                -8.28069088e-07,
                1.57475020e-10,
                -1.05108950e-14,
                2.31649722e03,
                -1.17416951e-01,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.94403120e00,
                -1.58542900e-03,
                1.66578120e-05,
                -2.04754260e-08,
                7.83505040e-12,
                2.89661800e03,
                6.31199190e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=33100.0,  # J/mol
    )

    # NO (Nitric Oxide) - Combustion product
    db["NO"] = SpeciesData(
        name="NO",
        molecular_weight=30.0061,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                3.26071234e00,
                1.19101135e-03,
                -4.29122646e-07,
                6.94481463e-11,
                -4.03295681e-15,
                9.92143132e03,
                6.36900518e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                4.21859896e00,
                -4.63988124e-03,
                1.10443049e-05,
                -9.34055507e-09,
                2.80554874e-12,
                9.84509964e03,
                2.28061001e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=91290.0,  # J/mol
    )

    # N2H4 (Hydrazine) - Storable fuel
    db["N2H4"] = SpeciesData(
        name="N2H4",
        molecular_weight=32.04516,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                4.93957047e00,
                5.44619867e-03,
                -1.81867996e-06,
                2.86416949e-10,
                -1.68234315e-14,
                9.28248610e03,
                -2.93660548e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                3.83471652e00,
                -6.49892297e-04,
                2.55942828e-05,
                -3.05458668e-08,
                1.22543904e-11,
                1.00893925e04,
                5.75272614e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=95350.0,  # J/mol
    )

    # C12H24 (RP-1 surrogate - n-Dodecane approximation)
    # RP-1 is approximated as C12H24 (average kerosene)
    db["RP1"] = SpeciesData(
        name="RP1",
        molecular_weight=168.319,  # C12H24 approx
        phase="L",
        temp_ranges=[(298.0, 1000.0, 5000.0)],
        coeffs_high=np.array(
            [
                2.63340726e01,
                5.15951294e-02,
                -1.76158656e-05,
                2.75303318e-09,
                -1.60259334e-13,
                -4.37463200e04,
                -1.06582054e02,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                -2.21757100e00,
                1.43567380e-01,
                -9.15679800e-05,
                2.99573500e-08,
                -3.95011800e-12,
                -3.53167100e04,
                4.68196500e01,
            ],
            dtype=np.float64,
        ),
        h_formation_298=-290900.0,  # J/mol (liquid)
    )

    # C (Solid carbon for soot calculations)
    db["C(s)"] = SpeciesData(
        name="C(s)",
        molecular_weight=12.0107,
        phase="S",
        temp_ranges=[(200.0, 1000.0, 5000.0)],
        coeffs_high=np.array(
            [
                1.45571870e00,
                1.71702470e-03,
                -6.97562390e-07,
                1.35277160e-10,
                -1.00328830e-14,
                -6.95137900e02,
                -8.52583350e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                -3.10872240e-01,
                4.40353550e-03,
                1.90394100e-06,
                -6.38546880e-09,
                2.98964460e-12,
                -1.08650140e02,
                1.11382480e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=0.0,  # J/mol (reference element)
    )
    # MMH, UDMH, H2O2, N2O - storable propellants
    # Reference: NASA CEA database, NIST, Thermodynamics of Organic Compounds
    # MMH (Monomethyl Hydrazine) - CH6N2
    # Used in Soyuz, Dragon, many spacecraft RCS
    db["MMH"] = SpeciesData(
        name="MMH",
        molecular_weight=46.0717,  # CH3NHNH2
        phase="L",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                5.97286300e00,
                1.08568840e-02,
                -3.90853780e-06,
                6.33853540e-10,
                -3.82653260e-14,
                1.10987100e04,
                -7.03399280e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                2.83575980e00,
                1.13608450e-02,
                1.59989500e-05,
                -2.65791360e-08,
                1.08673640e-11,
                1.23653780e04,
                1.00936700e01,
            ],
            dtype=np.float64,
        ),
        h_formation_298=54840.0,  # J/mol (liquid)
    )

    # UDMH (Unsymmetrical Dimethyl Hydrazine) - C2H8N2
    # Used in Proton, Long March, historical US rockets
    db["UDMH"] = SpeciesData(
        name="UDMH",
        molecular_weight=60.0986,  # (CH3)2NNH2
        phase="L",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                7.63116200e00,
                1.47316050e-02,
                -5.27679720e-06,
                8.51247420e-10,
                -5.12137580e-14,
                5.17621100e03,
                -1.60548670e01,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                1.55206330e00,
                2.42217360e-02,
                1.11891880e-05,
                -3.22019830e-08,
                1.42818850e-11,
                7.08247380e03,
                1.73696330e01,
            ],
            dtype=np.float64,
        ),
        h_formation_298=48300.0,  # J/mol (liquid)
    )

    # H2O2 (Hydrogen Peroxide) - High concentration (90%+)
    # Used in RCS, hybrid rockets, monopropellant
    db["H2O2"] = SpeciesData(
        name="H2O2",
        molecular_weight=34.01468,
        phase="L",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                4.57316685e00,
                4.33613639e-03,
                -1.47468882e-06,
                2.34890357e-10,
                -1.43165356e-14,
                -1.80069609e04,
                6.64961940e-01,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                4.31515149e00,
                -8.47390622e-04,
                1.76404323e-05,
                -2.26762944e-08,
                9.08950158e-12,
                -1.77067437e04,
                3.27373319e00,
            ],
            dtype=np.float64,
        ),
        h_formation_298=-187780.0,  # J/mol (liquid)
    )

    # N2O (Nitrous Oxide) - "Laughing Gas"
    # Popular hybrid oxidizer for amateur rocketry
    db["N2O"] = SpeciesData(
        name="N2O",
        molecular_weight=44.0128,
        phase="G",
        temp_ranges=[(200.0, 1000.0, 6000.0)],
        coeffs_high=np.array(
            [
                4.82318400e00,
                2.62693690e-03,
                -9.58508320e-07,
                1.60007330e-10,
                -9.77535890e-15,
                8.07340920e03,
                -2.20172070e00,
            ],
            dtype=np.float64,
        ),
        coeffs_low=np.array(
            [
                2.25715020e00,
                1.13047280e-02,
                -1.36713190e-05,
                9.68198020e-09,
                -2.93071820e-12,
                8.74177280e03,
                1.07579920e01,
            ],
            dtype=np.float64,
        ),
        h_formation_298=82050.0,  # J/mol (gas)
    )

    return db
