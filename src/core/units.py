"""
Unit Conversion System for Engineering Calculations.

Provides:
- Global SI/Imperial unit system toggle
- Automatic conversion for all engineering quantities
- Unit-aware formatting for display

Phase 7: Professional UX feature.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
from enum import Enum
import numpy as np


class UnitSystem(Enum):
    """Available unit systems."""
    SI = "SI"
    IMPERIAL = "Imperial"


# =============================================================================
# Conversion Constants
# =============================================================================

# Pressure: Pa <-> psi
PA_PER_PSI = 6894.757
PSI_PER_PA = 1 / PA_PER_PSI

# Force: N <-> lbf
N_PER_LBF = 4.44822
LBF_PER_N = 1 / N_PER_LBF

# Length: m <-> in, ft
M_PER_IN = 0.0254
IN_PER_M = 1 / M_PER_IN
M_PER_FT = 0.3048
FT_PER_M = 1 / M_PER_FT

# Area: m² <-> in², ft²
M2_PER_IN2 = M_PER_IN ** 2
IN2_PER_M2 = IN_PER_M ** 2
M2_PER_FT2 = M_PER_FT ** 2
FT2_PER_M2 = FT_PER_M ** 2

# Temperature: K <-> °R (Rankine)
# T(R) = T(K) * 1.8
K_TO_R = 1.8
R_TO_K = 1 / K_TO_R

# Mass: kg <-> lbm
KG_PER_LBM = 0.453592
LBM_PER_KG = 1 / KG_PER_LBM

# Mass flow: kg/s <-> lbm/s
# Same ratio as mass

# Velocity: m/s <-> ft/s
MPS_TO_FPS = FT_PER_M  # m/s -> ft/s

# Specific impulse: s (same in both systems!)
# Isp is dimensionless when written as F/(m_dot * g0)

# Molecular weight: g/mol (same in both systems)


# =============================================================================
# Unit Definitions
# =============================================================================

@dataclass
class UnitDef:
    """Definition of a unit type with conversions."""
    name: str
    si_unit: str
    imperial_unit: str
    to_imperial: float  # SI * to_imperial = Imperial
    decimals_si: int = 2
    decimals_imperial: int = 2


UNIT_DEFINITIONS: Dict[str, UnitDef] = {
    'pressure': UnitDef(
        name='Pressure',
        si_unit='bar',
        imperial_unit='psi',
        to_imperial=14.5038,  # bar -> psi
        decimals_si=1,
        decimals_imperial=0
    ),
    'pressure_pa': UnitDef(
        name='Pressure',
        si_unit='Pa',
        imperial_unit='psi',
        to_imperial=PSI_PER_PA,
        decimals_si=0,
        decimals_imperial=2
    ),
    'force': UnitDef(
        name='Force',
        si_unit='kN',
        imperial_unit='klbf',
        to_imperial=LBF_PER_N,  # N -> lbf, but we use kN -> klbf
        decimals_si=2,
        decimals_imperial=2
    ),
    'thrust': UnitDef(
        name='Thrust',
        si_unit='kN',
        imperial_unit='klbf',
        to_imperial=0.2248,  # kN -> klbf
        decimals_si=2,
        decimals_imperial=2
    ),
    'length': UnitDef(
        name='Length',
        si_unit='m',
        imperial_unit='in',
        to_imperial=IN_PER_M,
        decimals_si=3,
        decimals_imperial=2
    ),
    'length_cm': UnitDef(
        name='Length',
        si_unit='cm',
        imperial_unit='in',
        to_imperial=0.3937,  # cm -> in
        decimals_si=2,
        decimals_imperial=2
    ),
    'area': UnitDef(
        name='Area',
        si_unit='cm²',
        imperial_unit='in²',
        to_imperial=0.155,  # cm² -> in²
        decimals_si=2,
        decimals_imperial=3
    ),
    'temperature': UnitDef(
        name='Temperature',
        si_unit='K',
        imperial_unit='°R',
        to_imperial=K_TO_R,
        decimals_si=0,
        decimals_imperial=0
    ),
    'velocity': UnitDef(
        name='Velocity',
        si_unit='m/s',
        imperial_unit='ft/s',
        to_imperial=MPS_TO_FPS,
        decimals_si=1,
        decimals_imperial=0
    ),
    'mass': UnitDef(
        name='Mass',
        si_unit='kg',
        imperial_unit='lbm',
        to_imperial=LBM_PER_KG,
        decimals_si=2,
        decimals_imperial=2
    ),
    'mass_flow': UnitDef(
        name='Mass Flow',
        si_unit='kg/s',
        imperial_unit='lbm/s',
        to_imperial=LBM_PER_KG,
        decimals_si=3,
        decimals_imperial=3
    ),
    'isp': UnitDef(
        name='Specific Impulse',
        si_unit='s',
        imperial_unit='s',
        to_imperial=1.0,  # Same in both systems
        decimals_si=1,
        decimals_imperial=1
    ),
    'c_star': UnitDef(
        name='C*',
        si_unit='m/s',
        imperial_unit='ft/s',
        to_imperial=MPS_TO_FPS,
        decimals_si=1,
        decimals_imperial=0
    ),
    'mach': UnitDef(
        name='Mach Number',
        si_unit='',
        imperial_unit='',
        to_imperial=1.0,  # Dimensionless
        decimals_si=3,
        decimals_imperial=3
    ),
    'ratio': UnitDef(
        name='Ratio',
        si_unit='',
        imperial_unit='',
        to_imperial=1.0,  # Dimensionless
        decimals_si=2,
        decimals_imperial=2
    ),
}


# =============================================================================
# Unit Registry (Singleton)
# =============================================================================

class UnitRegistry:
    """
    Global unit conversion registry.
    
    Manages SI/Imperial toggle and provides conversion methods.
    
    Usage:
        units = UnitRegistry()
        units.set_system(UnitSystem.IMPERIAL)
        
        # Convert value for display
        display_val = units.convert('pressure', 100)  # 100 bar -> 1450 psi
        label = units.get_unit('pressure')  # 'psi'
    """
    
    _instance: Optional['UnitRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._system = UnitSystem.SI
            cls._instance._callbacks = []
        return cls._instance
    
    @property
    def system(self) -> UnitSystem:
        """Current unit system."""
        return self._system
    
    @system.setter
    def system(self, value: UnitSystem):
        """Set unit system and notify listeners."""
        if self._system != value:
            self._system = value
            self._notify_change()
    
    def set_system(self, system: UnitSystem):
        """Set the active unit system."""
        self.system = system
    
    def toggle_system(self):
        """Toggle between SI and Imperial."""
        if self._system == UnitSystem.SI:
            self.system = UnitSystem.IMPERIAL
        else:
            self.system = UnitSystem.SI
    
    def is_si(self) -> bool:
        """Check if current system is SI."""
        return self._system == UnitSystem.SI
    
    def is_imperial(self) -> bool:
        """Check if current system is Imperial."""
        return self._system == UnitSystem.IMPERIAL
    
    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------
    
    def convert(self, unit_type: str, value: float) -> float:
        """
        Convert value from SI to current display system.
        
        Args:
            unit_type: Key from UNIT_DEFINITIONS
            value: Value in SI units (base)
            
        Returns:
            Converted value in current system
        """
        if unit_type not in UNIT_DEFINITIONS:
            return value
        
        if self._system == UnitSystem.SI:
            return value
        
        unit_def = UNIT_DEFINITIONS[unit_type]
        return value * unit_def.to_imperial
    
    def convert_back(self, unit_type: str, value: float) -> float:
        """
        Convert value from current display system back to SI.
        
        Args:
            unit_type: Key from UNIT_DEFINITIONS
            value: Value in current display system
            
        Returns:
            Value in SI units
        """
        if unit_type not in UNIT_DEFINITIONS:
            return value
        
        if self._system == UnitSystem.SI:
            return value
        
        unit_def = UNIT_DEFINITIONS[unit_type]
        return value / unit_def.to_imperial
    
    def get_unit(self, unit_type: str) -> str:
        """Get current unit string for display."""
        if unit_type not in UNIT_DEFINITIONS:
            return ''
        
        unit_def = UNIT_DEFINITIONS[unit_type]
        if self._system == UnitSystem.SI:
            return unit_def.si_unit
        return unit_def.imperial_unit
    
    def get_decimals(self, unit_type: str) -> int:
        """Get decimal places for formatting."""
        if unit_type not in UNIT_DEFINITIONS:
            return 2
        
        unit_def = UNIT_DEFINITIONS[unit_type]
        if self._system == UnitSystem.SI:
            return unit_def.decimals_si
        return unit_def.decimals_imperial
    
    def format_value(self, unit_type: str, value: float, 
                     include_unit: bool = True) -> str:
        """
        Format value with conversion and unit label.
        
        Args:
            unit_type: Type of unit
            value: Value in SI (base) units
            include_unit: Whether to append unit string
            
        Returns:
            Formatted string like "1450.0 psi"
        """
        converted = self.convert(unit_type, value)
        decimals = self.get_decimals(unit_type)
        unit = self.get_unit(unit_type)
        
        if include_unit and unit:
            return f"{converted:.{decimals}f} {unit}"
        return f"{converted:.{decimals}f}"
    
    # -------------------------------------------------------------------------
    # Change Notification
    # -------------------------------------------------------------------------
    
    def register_callback(self, callback: Callable[[], None]):
        """Register callback to be called when unit system changes."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[], None]):
        """Unregister callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_change(self):
        """Notify all registered callbacks of system change."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Unit callback error: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_units() -> UnitRegistry:
    """Get the global UnitRegistry instance."""
    return UnitRegistry()


def convert(unit_type: str, value: float) -> float:
    """Convert value from SI to current display system."""
    return get_units().convert(unit_type, value)


def convert_back(unit_type: str, value: float) -> float:
    """Convert value from current display system to SI."""
    return get_units().convert_back(unit_type, value)


def format_value(unit_type: str, value: float, include_unit: bool = True) -> str:
    """Format value with unit."""
    return get_units().format_value(unit_type, value, include_unit)


# =============================================================================
# Specific Conversion Helpers
# =============================================================================

def bar_to_display(bar: float) -> float:
    """Convert pressure from bar to current display unit."""
    return get_units().convert('pressure', bar)


def display_to_bar(value: float) -> float:
    """Convert pressure from display unit to bar."""
    return get_units().convert_back('pressure', value)


def kn_to_display(kn: float) -> float:
    """Convert thrust from kN to current display unit."""
    return get_units().convert('thrust', kn)


def display_to_kn(value: float) -> float:
    """Convert thrust from display unit to kN."""
    return get_units().convert_back('thrust', value)


def kelvin_to_display(k: float) -> float:
    """Convert temperature from K to current display unit."""
    return get_units().convert('temperature', k)


def display_to_kelvin(value: float) -> float:
    """Convert temperature from display unit to K."""
    return get_units().convert_back('temperature', value)


def mps_to_display(mps: float) -> float:
    """Convert velocity from m/s to current display unit."""
    return get_units().convert('velocity', mps)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Unit Conversion System...")
    print("=" * 50)
    
    units = UnitRegistry()
    
    # Test SI
    print("\nSI Mode:")
    print(f"  100 bar -> {format_value('pressure', 100)}")
    print(f"  10 kN -> {format_value('thrust', 10)}")
    print(f"  3500 K -> {format_value('temperature', 3500)}")
    
    # Switch to Imperial
    units.set_system(UnitSystem.IMPERIAL)
    print("\nImperial Mode:")
    print(f"  100 bar -> {format_value('pressure', 100)}")
    print(f"  10 kN -> {format_value('thrust', 10)}")
    print(f"  3500 K -> {format_value('temperature', 3500)}")
    
    # Round-trip test
    print("\nRound-trip Test (100 bar):")
    val = 100  # bar
    converted = convert('pressure', val)
    back = convert_back('pressure', converted)
    print(f"  {val} bar -> {converted:.1f} psi -> {back:.1f} bar")
    
    print("\n✓ Unit conversion test complete!")
