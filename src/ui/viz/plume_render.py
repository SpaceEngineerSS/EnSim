"""
Plume Visualization Module.

Renders exhaust plume using PyVista with:
- Shock diamond visualization for over-expanded nozzles
- Mach-to-color mapping
- Volumetric semi-transparent mesh

References:
    - Sutton, G.P. "Rocket Propulsion Elements", Ch. 4
    - Anderson, J.D. "Modern Compressible Flow", Ch. 5
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Try to import PyVista
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

# Color maps for Mach number visualization
# White -> Yellow -> Orange -> Red (high to low Mach)
MACH_COLORMAP = 'plasma'  # Built-in colormap


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlumeConditions:
    """Exhaust plume operating conditions."""
    P_exit: float  # Exit pressure (Pa)
    P_ambient: float  # Ambient pressure (Pa)
    M_exit: float  # Exit Mach number
    T_exit: float  # Exit temperature (K)
    V_exit: float  # Exit velocity (m/s)
    gamma: float  # Ratio of specific heats

    # Nozzle geometry
    exit_diameter: float  # m

    @property
    def pressure_ratio(self) -> float:
        """Exit to ambient pressure ratio."""
        if self.P_ambient <= 0:
            return float('inf')
        return self.P_exit / self.P_ambient

    @property
    def is_overexpanded(self) -> bool:
        """Check if nozzle is over-expanded."""
        return self.P_exit < self.P_ambient

    @property
    def is_underexpanded(self) -> bool:
        """Check if nozzle is under-expanded."""
        if self.P_ambient <= 0:
            return True  # Vacuum is always under-expanded
        return self.P_exit > self.P_ambient

    @property
    def is_adapted(self) -> bool:
        """Check if nozzle is ideally adapted."""
        if self.P_ambient <= 0:
            return False
        return abs(self.P_exit - self.P_ambient) / self.P_ambient < 0.05


@dataclass
class ShockDiamondResult:
    """Result from shock diamond calculation."""
    wavelength: float  # Shock cell length (m)
    n_cells: int  # Number of visible cells
    intensity_decay: float  # Decay factor per cell
    positions: np.ndarray  # Axial positions of shock centers
    radii: np.ndarray  # Radius at each position


# =============================================================================
# Shock Diamond Physics
# =============================================================================

class ShockDiamondCalculator:
    """
    Calculate shock diamond pattern in exhaust plume.

    Shock diamonds form when the exit pressure doesn't match ambient:
    - Over-expanded (Pe < Pa): Oblique shocks form
    - Under-expanded (Pe > Pa): Expansion fans form

    The shock cell wavelength is approximately:
    L ≈ 0.8 * D_exit * sqrt(|M_exit² - 1|)
    """

    def calculate(
        self,
        conditions: PlumeConditions,
        n_cells: int = 5,
    ) -> ShockDiamondResult:
        """
        Calculate shock diamond pattern.

        Args:
            conditions: Plume operating conditions
            n_cells: Maximum number of cells to calculate

        Returns:
            ShockDiamondResult with positions and radii
        """
        M = conditions.M_exit
        D_exit = conditions.exit_diameter

        # Shock cell wavelength
        # Prandtl-Meyer factor for supersonic expansion
        if M > 1.0:
            wavelength = 0.8 * D_exit * np.sqrt(abs(M * M - 1.0))
        else:
            wavelength = D_exit  # Subsonic fallback

        # Minimum wavelength
        wavelength = max(wavelength, D_exit * 0.5)

        # Intensity decay (empirical)
        intensity_decay = 0.7

        # Calculate positions
        positions = np.zeros(n_cells)
        radii = np.zeros(n_cells)

        for i in range(n_cells):
            # Axial position of shock center
            positions[i] = (i + 0.5) * wavelength

            # Radius varies with position (plume expansion)
            # For under-expanded: plume expands
            # For over-expanded: plume contracts then expands
            if conditions.is_underexpanded:
                # Expansion angle (simplified)
                # Clamp pressure_ratio to prevent inf in log10
                pr_clamped = min(conditions.pressure_ratio, 1000.0)
                expansion_angle = np.radians(5 + 3 * np.log10(max(0.1, pr_clamped)))
                # Clamp angle to prevent extreme tan values
                expansion_angle = min(expansion_angle, np.radians(45))
                radii[i] = D_exit / 2 + positions[i] * np.tan(expansion_angle)
            else:
                # Over-expanded: initial contraction
                phase = ((i % 2) * 2 - 1)  # -1, 1, -1, 1 ...
                contraction = 0.1 * (conditions.P_ambient / conditions.P_exit - 1)
                radii[i] = D_exit / 2 * (1 + phase * contraction * (intensity_decay ** i))

        return ShockDiamondResult(
            wavelength=wavelength,
            n_cells=n_cells,
            intensity_decay=intensity_decay,
            positions=positions,
            radii=radii
        )


# =============================================================================
# Plume Renderer
# =============================================================================

class PlumeRenderer:
    """
    Render exhaust plume using PyVista.

    Creates a volumetric mesh representing the exhaust plume
    with Mach-based coloring and shock diamonds.
    """

    def __init__(self):
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for plume rendering")

        self.shock_calc = ShockDiamondCalculator()

    def generate_plume_mesh(
        self,
        conditions: PlumeConditions,
        plume_length: float = None,
        n_axial: int = 50,
        n_radial: int = 20,
        n_circumferential: int = 36,
    ) -> "pv.PolyData":
        """
        Generate plume volumetric mesh.

        Args:
            conditions: Plume operating conditions
            plume_length: Total plume length (m), auto if None
            n_axial: Number of axial stations
            n_radial: Number of radial layers
            n_circumferential: Points around circumference

        Returns:
            PyVista PolyData with Mach field
        """
        D_exit = conditions.exit_diameter
        R_exit = D_exit / 2

        # Auto plume length
        if plume_length is None:
            plume_length = D_exit * 10  # Typical visible length

        # Calculate shock diamonds
        shock_result = self.shock_calc.calculate(conditions)

        # Generate axial positions
        x_positions = np.linspace(0, plume_length, n_axial)

        # Calculate local radius at each axial position
        radii = self._calculate_plume_radius(
            x_positions, conditions, shock_result, R_exit
        )

        # Calculate local Mach at each position
        mach_values = self._calculate_mach_distribution(
            x_positions, conditions, shock_result
        )

        # Generate 3D mesh
        points = []
        mach_data = []

        theta = np.linspace(0, 2 * np.pi, n_circumferential, endpoint=False)

        for _i_x, (x, R, M_local) in enumerate(zip(x_positions, radii, mach_values, strict=False)):
            for _i_r, r_frac in enumerate(np.linspace(0.1, 1.0, n_radial)):
                r = R * r_frac

                for t in theta:
                    y = r * np.cos(t)
                    z = r * np.sin(t)
                    points.append([x, y, z])

                    # Mach varies radially (core faster)
                    mach_local = M_local * (1.0 - 0.3 * r_frac * r_frac)
                    mach_data.append(mach_local)

        points = np.array(points)
        mach_data = np.array(mach_data)

        # Create point cloud
        cloud = pv.PolyData(points)
        cloud['Mach'] = mach_data
        cloud['Intensity'] = self._calculate_intensity(x_positions, mach_values, n_radial, n_circumferential)

        return cloud

    def generate_plume_surface(
        self,
        conditions: PlumeConditions,
        plume_length: float = None,
        n_axial: int = 100,
        n_circumferential: int = 48,
    ) -> "pv.PolyData":
        """
        Generate plume surface mesh (faster rendering).

        Args:
            conditions: Plume operating conditions
            plume_length: Total plume length (m)
            n_axial: Number of axial stations
            n_circumferential: Points around circumference

        Returns:
            PyVista surface mesh
        """
        D_exit = conditions.exit_diameter
        R_exit = D_exit / 2

        if plume_length is None:
            plume_length = D_exit * 8

        shock_result = self.shock_calc.calculate(conditions)

        # Axial positions
        x = np.linspace(0, plume_length, n_axial)

        # Radii along plume
        radii = self._calculate_plume_radius(x, conditions, shock_result, R_exit)

        # Mach values
        mach = self._calculate_mach_distribution(x, conditions, shock_result)

        # Create surface of revolution
        theta = np.linspace(0, 2 * np.pi, n_circumferential)

        X, Theta = np.meshgrid(x, theta)
        R_mesh = np.tile(radii, (n_circumferential, 1))
        M_mesh = np.tile(mach, (n_circumferential, 1))

        Y = R_mesh * np.cos(Theta)
        Z = R_mesh * np.sin(Theta)

        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        grid['Mach'] = M_mesh.ravel(order='F')

        # Convert to surface
        surface = grid.extract_surface()

        return surface

    def generate_shock_disks(
        self,
        conditions: PlumeConditions,
        n_disks: int = 3,
        disk_thickness: float = 0.02,
        n_circumferential: int = 36,
    ) -> Optional["pv.PolyData"]:
        """
        Generate Mach disk visualization for over-expanded nozzle.

        Returns None if not over-expanded.
        """
        if not conditions.is_overexpanded:
            return None

        shock_result = self.shock_calc.calculate(conditions, n_cells=n_disks)

        disks = []
        for i, (pos, radius) in enumerate(zip(shock_result.positions, shock_result.radii, strict=False)):
            # Create disk
            disk = pv.Disc(
                center=(pos, 0, 0),
                inner=0,
                outer=radius * 0.8,
                normal=(1, 0, 0),
                r_res=1,
                c_res=n_circumferential
            )

            # Intensity decreases with distance
            intensity = shock_result.intensity_decay ** i
            disk['Intensity'] = np.full(disk.n_points, intensity)
            disk['Mach'] = np.full(disk.n_points, conditions.M_exit * (1 - 0.1 * i))

            disks.append(disk)

        if disks:
            return pv.MultiBlock(disks).combine()
        return None

    def _calculate_plume_radius(
        self,
        x: np.ndarray,
        conditions: PlumeConditions,
        shock_result: ShockDiamondResult,
        R_exit: float
    ) -> np.ndarray:
        """Calculate plume radius at each axial position."""
        radii = np.zeros_like(x)

        for i, xi in enumerate(x):
            # Base expansion
            if conditions.is_underexpanded or conditions.P_ambient <= 0:
                # Plume expands
                expansion_half_angle = np.radians(8)  # Typical
                radii[i] = R_exit + xi * np.tan(expansion_half_angle)
            elif conditions.is_overexpanded:
                # Oscillating radius due to shock cells
                cell_idx = xi / shock_result.wavelength
                phase = np.sin(2 * np.pi * cell_idx)
                contraction = 0.15 * (shock_result.intensity_decay ** int(cell_idx))
                radii[i] = R_exit * (1 + phase * contraction)
            else:
                # Adapted - parallel flow
                radii[i] = R_exit

        return radii

    def _calculate_mach_distribution(
        self,
        x: np.ndarray,
        conditions: PlumeConditions,
        shock_result: ShockDiamondResult
    ) -> np.ndarray:
        """Calculate Mach number at each axial position."""
        M_exit = conditions.M_exit
        mach = np.zeros_like(x)

        for i, xi in enumerate(x):
            # Mach decays with distance due to mixing
            decay_length = shock_result.wavelength * 5
            mixing_decay = np.exp(-xi / decay_length)

            # Shock oscillations (for non-adapted flow)
            if not conditions.is_adapted:
                cell_idx = xi / shock_result.wavelength
                oscillation = 0.1 * np.cos(2 * np.pi * cell_idx) * (shock_result.intensity_decay ** int(cell_idx))
            else:
                oscillation = 0

            mach[i] = M_exit * mixing_decay * (1 + oscillation)
            mach[i] = max(0.1, mach[i])  # Minimum Mach

        return mach

    def _calculate_intensity(
        self,
        x: np.ndarray,
        mach: np.ndarray,
        n_radial: int,
        n_circumferential: int
    ) -> np.ndarray:
        """Calculate intensity for opacity mapping."""
        n_total = len(x) * n_radial * n_circumferential
        intensity = np.zeros(n_total)

        idx = 0
        for _i, M in enumerate(mach):
            for _ in range(n_radial * n_circumferential):
                # Intensity based on Mach (brighter for higher Mach)
                intensity[idx] = min(1.0, M / 3.0)
                idx += 1

        return intensity


# =============================================================================
# Utility Functions
# =============================================================================

def create_plume_colormap():
    """
    Create custom colormap for plume visualization.

    White (high Mach) -> Yellow -> Orange -> Red (low Mach) -> Transparent
    """
    if not PYVISTA_AVAILABLE:
        return None

    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.0, 'darkred'),
        (0.3, 'red'),
        (0.5, 'orange'),
        (0.7, 'yellow'),
        (0.9, 'white'),
        (1.0, 'lightblue'),
    ]

    return LinearSegmentedColormap.from_list('plume',
                                             [c[1] for c in colors],
                                             N=256)


def calculate_plume_bounds(conditions: PlumeConditions) -> dict:
    """
    Calculate approximate plume extent for camera setup.

    Returns:
        Dictionary with 'length', 'max_radius'
    """
    D = conditions.exit_diameter
    M = conditions.M_exit

    # Plume length scales with Mach number and pressure ratio
    if conditions.P_ambient <= 0:
        # Vacuum - very long plume
        length = D * 20
    else:
        length = D * (5 + 3 * abs(np.log10(max(0.1, conditions.pressure_ratio))))

    # Max radius
    max_radius = D * (1 + 0.2 * M) if conditions.is_underexpanded else D * 1.2

    return {
        'length': length,
        'max_radius': max_radius
    }


# =============================================================================
# Integration with existing 3D widget
# =============================================================================

def add_plume_to_scene(
    plotter,
    conditions: PlumeConditions,
    nozzle_exit_x: float = 0.0,
    opacity: float = 0.6,
    show_shock_disks: bool = True,
) -> None:
    """
    Add plume visualization to an existing PyVista plotter.

    Args:
        plotter: PyVista Plotter or QtInteractor
        conditions: Plume operating conditions
        nozzle_exit_x: X position of nozzle exit
        opacity: Plume opacity (0-1)
        show_shock_disks: Whether to show Mach disks
    """
    if not PYVISTA_AVAILABLE:
        return

    renderer = PlumeRenderer()

    # Generate plume surface
    plume = renderer.generate_plume_surface(conditions)

    # Translate to nozzle exit
    plume.translate([nozzle_exit_x, 0, 0], inplace=True)

    # Add to scene
    plotter.add_mesh(
        plume,
        scalars='Mach',
        cmap=MACH_COLORMAP,
        opacity=opacity,
        show_scalar_bar=False,
        name='exhaust_plume'
    )

    # Add shock disks if applicable
    if show_shock_disks and conditions.is_overexpanded:
        disks = renderer.generate_shock_disks(conditions)
        if disks is not None:
            disks.translate([nozzle_exit_x, 0, 0], inplace=True)
            plotter.add_mesh(
                disks,
                color='white',
                opacity=0.8,
                name='shock_disks'
            )


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Plume Renderer...")
    print("=" * 50)

    if not PYVISTA_AVAILABLE:
        print("PyVista not available - skipping visual test")
    else:
        # Create test conditions
        conditions = PlumeConditions(
            P_exit=50000,  # 0.5 bar
            P_ambient=101325,  # 1 atm (over-expanded)
            M_exit=2.5,
            T_exit=1500,
            V_exit=2500,
            gamma=1.2,
            exit_diameter=0.1
        )

        print(f"Over-expanded: {conditions.is_overexpanded}")
        print(f"Pressure ratio: {conditions.pressure_ratio:.2f}")

        # Test shock diamond calculation
        calc = ShockDiamondCalculator()
        shock = calc.calculate(conditions)
        print(f"\nShock wavelength: {shock.wavelength*1000:.1f} mm")
        print(f"Shock positions: {shock.positions}")

        # Test mesh generation
        renderer = PlumeRenderer()
        plume = renderer.generate_plume_surface(conditions)
        print(f"\nPlume mesh: {plume.n_points} points")

        disks = renderer.generate_shock_disks(conditions)
        if disks:
            print(f"Shock disks: {disks.n_points} points")

    print("\n✓ Plume Renderer test complete!")
