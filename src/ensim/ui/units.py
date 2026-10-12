"""Display units for the interface. Values stay SI everywhere else."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from ensim.utils.units import Quantity, UnitSystem, display_unit, from_si, to_si


class Units(QObject):
    """The unit system the user chose; widgets listen to :attr:`changed`."""

    changed = pyqtSignal(object)

    def __init__(self, system: UnitSystem = UnitSystem.SI):
        super().__init__()
        self.system = system

    def set_system(self, system: UnitSystem) -> None:
        if system != self.system:
            self.system = system
            self.changed.emit(system)

    def unit(self, quantity: Quantity, overrides: dict[UnitSystem, str] | None = None) -> str:
        if overrides and self.system in overrides:
            return overrides[self.system]
        return display_unit(quantity, self.system)

    def show(self, value_si: float, quantity: Quantity, unit: str | None = None) -> float:
        return from_si(value_si, quantity, unit or self.unit(quantity))

    def read(self, value: float, quantity: Quantity, unit: str | None = None) -> float:
        return to_si(value, quantity, unit or self.unit(quantity))


UNITS = Units()


def fmt(value: float | None, digits: int = 4) -> str:
    """Engineering-friendly number: about ``digits`` significant figures, no exponent below 1e6."""
    if value is None or value != value:
        return "–"
    magnitude = abs(value)
    if magnitude == 0.0:
        return "0"
    if magnitude >= 1e6 or magnitude < 1e-3:
        return f"{value:.{max(digits - 1, 1)}e}"
    integer_digits = len(str(int(magnitude))) if magnitude >= 1 else 0
    decimals = max(0, digits - integer_digits)
    if magnitude < 1:
        decimals = digits
    return f"{value:,.{decimals}f}".replace(",", " ")
