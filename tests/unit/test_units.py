import pytest

from ensim.utils.units import (
    DISPLAY_UNITS,
    SI_UNITS,
    UNITS,
    Quantity,
    UnitSystem,
    convert,
    display_unit,
    from_si,
    to_si,
)


@pytest.mark.parametrize(
    ("quantity", "value", "unit", "si"),
    [
        (Quantity.LENGTH, 1.0, "ft", 0.3048),
        (Quantity.MASS, 1.0, "lb", 0.45359237),
        (Quantity.MASS, 250.0, "g", 0.25),
        (Quantity.ACCELERATION, 1.0, "g", 9.80665),
        (Quantity.FORCE, 1.0, "lbf", 4.4482216152605),
        (Quantity.PRESSURE, 1.0, "psi", 6894.757293168361),
        (Quantity.PRESSURE, 1.0, "atm", 101325.0),
        (Quantity.TEMPERATURE, 25.0, "°C", 298.15),
        (Quantity.TEMPERATURE, 32.0, "°F", 273.15),
        (Quantity.TEMPERATURE, 491.67, "°R", 273.15),
        (Quantity.VELOCITY, 36.0, "km/h", 10.0),
        (Quantity.SPECIFIC_IMPULSE, 9806.65, "m/s", 1000.0),
    ],
)
def test_reference_conversions(quantity, value, unit, si):
    assert to_si(value, quantity, unit) == pytest.approx(si, rel=1e-12)
    assert from_si(si, quantity, unit) == pytest.approx(value, rel=1e-12)


def test_round_trip_every_unit():
    for quantity, units in UNITS.items():
        for unit in units:
            assert convert(
                convert(123.456, quantity, SI_UNITS[quantity], unit),
                quantity,
                unit,
                SI_UNITS[quantity],
            ) == pytest.approx(123.456, rel=1e-12)


def test_every_quantity_has_si_and_display_units():
    for quantity in Quantity:
        assert SI_UNITS[quantity] in UNITS[quantity]
        for system in UnitSystem:
            assert display_unit(quantity, system) in UNITS[quantity]
    assert set(DISPLAY_UNITS) == set(UnitSystem)


def test_unknown_unit_names_the_alternatives():
    with pytest.raises(ValueError, match="lbf"):
        to_si(1.0, Quantity.FORCE, "pound")
