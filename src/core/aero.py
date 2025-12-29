"""
Barrowman Aerodynamics Module.

Implements the Barrowman equations for rocket stability analysis
including CN_alpha calculations, CP locations, and drag estimation.

Compressibility corrections:
    - Subsonic (M < 0.8): Prandtl-Glauert correction
    - Transonic (0.8 < M < 1.2): Linear interpolation (singularity avoidance)
    - Supersonic (M > 1.2): Ackeret linear theory

References:
    - Barrowman, J.S. "The Practical Calculation of the Aerodynamic
      Characteristics of Slender Finned Vehicles", 1967.
    - Anderson, J.D. "Fundamentals of Aerodynamics", 6th ed., Ch. 9-11.
    - Prandtl, L. & Glauert, H. "Effect of Compressibility" (1929).
"""

import numpy as np
from numba import jit
from dataclasses import dataclass
from typing import Optional

from src.core.rocket import Rocket, NoseCone, BodyTube, FinSet, NoseShape


# =============================================================================
# Compressibility Corrections
# =============================================================================

@jit(nopython=True, cache=True)
def compressibility_correction(mach: float) -> float:
    """
    Calculate compressibility correction factor for CN_alpha.
    
    This factor multiplies the incompressible CN_alpha to account
    for Mach number effects on normal force coefficient.
    
    Physics:
        - Subsonic (M < 0.8): Prandtl-Glauert rule
          CN_alpha(M) = CN_alpha_0 / sqrt(1 - M²)
          
        - Transonic (0.8 < M < 1.2): Linear interpolation
          Avoids singularity at M = 1.0
          
        - Supersonic (M > 1.2): Ackeret linear theory
          CN_alpha(M) = 4 / sqrt(M² - 1) for thin airfoils
          Normalized to match at M = 1.2
    
    Args:
        mach: Flight Mach number
        
    Returns:
        Correction factor to multiply incompressible CN_alpha
        
    Reference:
        Anderson, J.D. "Fundamentals of Aerodynamics", Eq. 9.36, 11.22
    """
    if mach < 0.3:
        # Low speed - negligible compressibility effects
        return 1.0
    
    elif mach < 0.8:
        # Subsonic: Prandtl-Glauert correction
        # Factor = 1 / sqrt(1 - M²)
        beta = np.sqrt(1.0 - mach * mach)
        return 1.0 / beta
    
    elif mach < 1.2:
        # Transonic regime: Linear interpolation to avoid singularity
        # At M=0.8: factor ≈ 1.667 (Prandtl-Glauert)
        # At M=1.2: factor ≈ 1.51 (Ackeret normalized)
        factor_08 = 1.0 / np.sqrt(1.0 - 0.8 * 0.8)  # ≈ 1.667
        factor_12 = 4.0 / np.sqrt(1.2 * 1.2 - 1.0) / 2.0  # ≈ 3.01 / 2 ≈ 1.51
        
        # Linear interpolation
        t = (mach - 0.8) / 0.4
        return factor_08 * (1.0 - t) + factor_12 * t
    
    else:
        # Supersonic: Ackeret linear theory
        # Factor = 4 / sqrt(M² - 1) for 2D, adjusted for 3D body
        # Normalized to provide smooth transition
        beta = np.sqrt(mach * mach - 1.0)
        ackeret_factor = 4.0 / beta
        
        # Normalize to ~1.5 at M=1.2 for continuity
        normalization = 0.5
        return ackeret_factor * normalization


@dataclass
class AeroResult:
    """Results from aerodynamic analysis."""
    # Normal force coefficients
    cn_alpha_nose: float
    cn_alpha_body: float
    cn_alpha_fins: float
    cn_alpha_total: float
    
    # Center of pressure locations (from nose tip)
    cp_nose: float
    cp_body: float
    cp_fins: float
    cp_total: float
    
    # Stability
    cg: float  # Center of gravity
    stability_margin: float  # Calibers
    is_stable: bool
    
    # Drag
    cd_friction: float
    cd_base: float
    cd_total: float


def calculate_cn_alpha_nose(nose: NoseCone) -> float:
    """
    Calculate normal force coefficient slope for nose cone.
    
    For a nose cone, CN_alpha = 2 (theoretical, all shapes).
    
    Args:
        nose: NoseCone component
        
    Returns:
        CN_alpha (per radian)
    """
    return 2.0


def calculate_cp_nose(nose: NoseCone) -> float:
    """
    Calculate center of pressure location for nose cone.
    
    CP location depends on nose shape:
    - Conical: 2/3 L from nose tip
    - Ogive: 0.466 L from nose tip
    - Parabolic: 0.5 L from nose tip
    
    Args:
        nose: NoseCone component
        
    Returns:
        CP distance from nose tip (m)
    """
    L = nose.length
    
    if nose.shape == NoseShape.CONICAL:
        return 2.0 / 3.0 * L
    elif nose.shape == NoseShape.OGIVE:
        return 0.466 * L
    elif nose.shape == NoseShape.PARABOLIC:
        return 0.5 * L
    elif nose.shape == NoseShape.ELLIPTICAL:
        return 0.333 * L
    else:
        return 0.466 * L  # Default to ogive


def calculate_cn_alpha_body(body: BodyTube) -> float:
    """
    Calculate normal force coefficient slope for body tube.
    
    For cylindrical body: CN_alpha ≈ 0 (no contribution at low alpha).
    
    Args:
        body: BodyTube component
        
    Returns:
        CN_alpha (per radian)
    """
    # Body tube contributes negligibly at low angles
    return 0.0


def calculate_cn_alpha_fins(fins: FinSet, body_diameter: float) -> float:
    """
    Calculate normal force coefficient slope for fin set.
    
    Barrowman equation for fins:
    CN_alpha = K_fb × (4N × (S/d)²) / (1 + sqrt(1 + (2Lf/(Cr+Ct))²))
    
    Where:
        K_fb = fin-body interference factor
        N = number of fins
        S = semi-span
        d = body diameter
        Lf = mid-chord sweep length
        
    Args:
        fins: FinSet component
        body_diameter: Reference body diameter (m)
        
    Returns:
        CN_alpha (per radian)
    """
    fin = fins.fin
    N = fins.count
    S = fin.span
    Cr = fin.root_chord
    Ct = fin.tip_chord
    d = body_diameter
    
    # Mid-chord sweep length
    Lf = np.tan(np.radians(fin.sweep_angle)) * S + (Ct - Cr) / 2
    
    # Fin-body interference factor (Barrowman approximation)
    r = d / 2
    K_fb = 1 + r / (r + S)
    
    # Barrowman equation
    denominator = 1 + np.sqrt(1 + (2 * Lf / (Cr + Ct))**2)
    cn_alpha = K_fb * (4 * N * (S / d)**2) / denominator
    
    return cn_alpha


def calculate_cp_fins(fins: FinSet, body_length_to_fins: float, body_diameter: float) -> float:
    """
    Calculate center of pressure for fin set.
    
    Barrowman equation for fin CP (measured from fin root LE):
    X_f = (Lf × (Cr + 2Ct)) / (3 × (Cr + Ct)) + (Cr + Ct - Cr×Ct/(Cr + Ct)) / 6
    
    Args:
        fins: FinSet component  
        body_length_to_fins: Distance from nose to fin leading edge (m)
        body_diameter: Reference body diameter (m)
        
    Returns:
        CP distance from nose tip (m)
    """
    fin = fins.fin
    Cr = fin.root_chord
    Ct = fin.tip_chord
    S = fin.span
    
    # Mid-chord sweep length
    Lf = np.tan(np.radians(fin.sweep_angle)) * S
    
    # CP from fin root leading edge (Barrowman)
    x_fin_local = (Lf * (Cr + 2*Ct) / (3 * (Cr + Ct)) + 
                   (Cr + Ct - Cr*Ct/(Cr + Ct)) / 6)
    
    # Convert to distance from nose
    x_cp_fins = body_length_to_fins + x_fin_local
    
    return x_cp_fins


def calculate_total_cp(rocket: Rocket) -> tuple:
    """
    Calculate total center of pressure for complete rocket.
    
    CP is the weighted average by CN_alpha:
    X_cp_total = Σ(CN_alpha_i × X_cp_i) / Σ(CN_alpha_i)
    
    Args:
        rocket: Rocket object
        
    Returns:
        Tuple of (x_cp_total, cn_alpha_total, component_data)
    """
    # Calculate CN_alpha for each component
    cn_nose = calculate_cn_alpha_nose(rocket.nose)
    cn_body = calculate_cn_alpha_body(rocket.body)
    cn_fins = calculate_cn_alpha_fins(rocket.fins, rocket.body.diameter)
    
    # Calculate CP locations
    cp_nose = calculate_cp_nose(rocket.nose)
    cp_body = rocket.nose.length + rocket.body.length / 2  # Mid-body (not used)
    cp_fins = calculate_cp_fins(
        rocket.fins, 
        rocket.fins.position,
        rocket.body.diameter
    )
    
    # Weighted average (body contribution is zero)
    cn_total = cn_nose + cn_body + cn_fins
    
    if cn_total > 0:
        cp_total = (cn_nose * cp_nose + cn_fins * cp_fins) / cn_total
    else:
        cp_total = rocket.total_length / 2
    
    return cp_total, cn_total, {
        'cn_nose': cn_nose,
        'cn_body': cn_body,
        'cn_fins': cn_fins,
        'cp_nose': cp_nose,
        'cp_fins': cp_fins
    }


def calculate_stability_margin(rocket: Rocket, time: float = 0.0) -> float:
    """
    Calculate stability margin in calibers.
    
    Margin = (X_cp - X_cg) / D_ref
    
    Positive margin = stable (CP behind CG)
    Margin > 1.0 caliber = recommended minimum
    
    Args:
        rocket: Rocket object
        time: Time for dynamic CG calculation (s)
        
    Returns:
        Stability margin in calibers
    """
    cp_total, _, _ = calculate_total_cp(rocket)
    cg = rocket.get_cg_at_time(time)
    d_ref = rocket.reference_diameter
    
    margin = (cp_total - cg) / d_ref
    return margin


@jit(nopython=True, cache=True)
def _cd_friction_turb(Re: float, roughness: float = 0.0) -> float:
    """
    Turbulent friction drag coefficient.
    
    Schlichting's formula for turbulent flat plate.
    """
    if Re <= 0:
        return 0.0
    
    cf = 0.455 / (np.log10(Re) ** 2.58)
    return cf


@jit(nopython=True, cache=True)
def _cd_wave_drag(mach: float) -> float:
    """
    Transonic wave drag coefficient spike.
    
    Peaks at Mach 1.0, decreases in supersonic regime.
    """
    if mach < 0.8:
        return 0.0
    elif mach < 1.0:
        # Rapid increase approaching Mach 1
        return 0.3 * ((mach - 0.8) / 0.2) ** 2
    elif mach < 1.2:
        # Peak around Mach 1, decreasing
        return 0.3 * (1.0 - (mach - 1.0) / 0.2)
    else:
        # Supersonic decay
        return 0.2 / (mach ** 2)


def calculate_drag_coefficient(
    rocket: Rocket,
    mach: float,
    reynolds: float = 1e7
) -> float:
    """
    Calculate total drag coefficient.
    
    Cd = Cd_friction + Cd_pressure + Cd_base + Cd_wave
    
    Args:
        rocket: Rocket object
        mach: Flight Mach number
        reynolds: Reynolds number
        
    Returns:
        Total drag coefficient (referenced to body cross-section)
    """
    # Friction drag (skin friction)
    cf = _cd_friction_turb(reynolds)
    
    # Wetted area ratio
    A_wet = (rocket.nose.volume * 0.5 / rocket.nose.length +  # Nose approximation
             rocket.body.wetted_area +
             rocket.fins.total_area * 2)  # Both sides of fins
    A_ref = rocket.reference_area
    
    cd_friction = cf * A_wet / A_ref * 1.4  # 1.4 includes form factor
    
    # Base drag (blunt base)
    cd_base = 0.12
    
    # Wave drag (transonic spike)
    cd_wave = _cd_wave_drag(mach)
    
    # Fin interference and miscellaneous
    cd_misc = 0.05
    
    cd_total = cd_friction + cd_base + cd_wave + cd_misc
    
    return cd_total


def analyze_rocket(rocket: Rocket, time: float = 0.0, mach: float = 0.3) -> AeroResult:
    """
    Perform complete aerodynamic analysis.
    
    Args:
        rocket: Rocket to analyze
        time: Time for dynamic CG (s)
        mach: Flight Mach number
        
    Returns:
        AeroResult with all aerodynamic data
    """
    # CN_alpha values
    cn_nose = calculate_cn_alpha_nose(rocket.nose)
    cn_body = calculate_cn_alpha_body(rocket.body)
    cn_fins = calculate_cn_alpha_fins(rocket.fins, rocket.body.diameter)
    cn_total = cn_nose + cn_body + cn_fins
    
    # CP locations
    cp_nose = calculate_cp_nose(rocket.nose)
    cp_body = rocket.nose.length + rocket.body.length / 2
    cp_fins = calculate_cp_fins(rocket.fins, rocket.fins.position, rocket.body.diameter)
    
    # Total CP
    if cn_total > 0:
        cp_total = (cn_nose * cp_nose + cn_fins * cp_fins) / cn_total
    else:
        cp_total = rocket.total_length / 2
    
    # CG and stability
    cg = rocket.get_cg_at_time(time)
    margin = (cp_total - cg) / rocket.reference_diameter
    is_stable = margin >= 1.0
    
    # Drag
    cf = _cd_friction_turb(1e7)
    cd_base = 0.12
    cd_total = calculate_drag_coefficient(rocket, mach)
    
    return AeroResult(
        cn_alpha_nose=cn_nose,
        cn_alpha_body=cn_body,
        cn_alpha_fins=cn_fins,
        cn_alpha_total=cn_total,
        cp_nose=cp_nose,
        cp_body=cp_body,
        cp_fins=cp_fins,
        cp_total=cp_total,
        cg=cg,
        stability_margin=margin,
        is_stable=is_stable,
        cd_friction=cf,
        cd_base=cd_base,
        cd_total=cd_total
    )
