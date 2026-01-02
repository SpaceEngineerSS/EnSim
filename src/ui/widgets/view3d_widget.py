"""3D Nozzle visualization using PyVista."""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

# Try to import PyVista - it may not be installed
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from src.core.constants import GAS_CONSTANT
from src.core.propulsion import get_nozzle_profile

# Try to import plume renderer
try:
    from src.ui.viz.plume_render import PlumeConditions, PlumeRenderer
    PLUME_AVAILABLE = True
except ImportError:
    PLUME_AVAILABLE = False


class NozzleView3D(QWidget):
    """
    3D visualization of nozzle geometry with temperature/Mach coloring.

    Uses PyVista for interactive 3D rendering if available,
    otherwise shows a placeholder.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plotter = None
        self._mesh = None
        self._plume_mesh = None
        self._show_plume = True
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the 3D view."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if PYVISTA_AVAILABLE:
            # Create PyVista Qt interactor
            self._plotter = QtInteractor(self)
            self._plotter.set_background('#0a0e14')
            self._plotter.add_axes(color='#2a3a4a')
            layout.addWidget(self._plotter.interactor)

            # Add initial placeholder text
            self._add_placeholder_text()
        else:
            # Fallback if PyVista not installed
            placeholder = QFrame()
            placeholder.setStyleSheet("""
                QFrame {
                    background-color: #0a0e14;
                    border: 1px solid #1e2830;
                    border-radius: 8px;
                }
            """)
            placeholder_layout = QVBoxLayout(placeholder)

            label = QLabel("3D Visualization requires PyVista")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #888888; font-size: 14pt;")
            placeholder_layout.addWidget(label)

            install_label = QLabel("Install with: pip install pyvista pyvistaqt")
            install_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            install_label.setStyleSheet("color: #555555; font-size: 10pt;")
            placeholder_layout.addWidget(install_label)

            layout.addWidget(placeholder)

    def _add_placeholder_text(self):
        """Add placeholder text to empty plotter."""
        if self._plotter:
            self._plotter.add_text(
                "Run simulation to see 3D nozzle",
                position='upper_left',
                font_size=12,
                color='#555555',
                name='placeholder'
            )

    def update_view(
        self,
        exit_area_ratio: float,
        gamma: float,
        T_chamber: float,
        P_chamber: float,
        mean_mw: float,
        color_by: str = 'temperature',
    ):
        """
        Update the 3D nozzle visualization.

        Args:
            exit_area_ratio: Nozzle expansion ratio
            gamma: Ratio of specific heats
            T_chamber: Chamber temperature (K)
            P_chamber: Chamber pressure (Pa)
            mean_mw: Mean molecular weight (g/mol)
            color_by: 'temperature' or 'mach'
        """
        if not PYVISTA_AVAILABLE or not self._plotter:
            return

        # Clear previous mesh
        self._plotter.clear()
        self._plotter.add_axes(color='#2a3a4a')

        # Use Rao Bell nozzle contour (V2.0)
        from src.core.geometry import generate_bell_contour

        throat_radius = 0.05  # 5cm
        contour = generate_bell_contour(
            throat_radius=throat_radius,
            expansion_ratio=exit_area_ratio,
            theta_n=35.0,  # Entry angle
            theta_e=8.0,   # Exit angle
            percent_bell=80.0,
            n_points=80
        )

        # Get flow properties
        R_specific = GAS_CONSTANT / (mean_mw / 1000.0)
        profile = get_nozzle_profile(
            gamma=gamma,
            T_chamber=T_chamber,
            P_chamber=P_chamber,
            exit_area_ratio=exit_area_ratio,
            R_specific=R_specific,
            n_points=len(contour.x)
        )

        # Create surface of revolution using Bell contour
        mesh = self._create_nozzle_mesh(contour.x, contour.r, profile, color_by)

        if mesh is not None:
            self._mesh = mesh

            # Choose colormap and field
            if color_by == 'temperature':
                scalars = 'temperature'
                cmap = 'hot'
                clabel = 'Temperature (K)'
            else:
                scalars = 'mach'
                cmap = 'viridis'
                clabel = 'Mach Number'

            self._plotter.add_mesh(
                mesh,
                scalars=scalars,
                cmap=cmap,
                show_edges=False,
                smooth_shading=True,
                scalar_bar_args={
                    'title': clabel,
                    'color': '#cccccc',
                    'title_font_size': 12,
                    'label_font_size': 10,
                }
            )

            # Add info text
            self._plotter.add_text(
                f"ε = {exit_area_ratio:.1f}  |  Tc = {T_chamber:.0f} K",
                position='upper_left',
                font_size=10,
                color='#888888',
                name='info'
            )

            # Add exhaust plume visualization (Phase 6)
            if self._show_plume and PLUME_AVAILABLE:
                try:
                    self._add_plume(
                        gamma=gamma,
                        T_chamber=T_chamber,
                        P_chamber=P_chamber,
                        mean_mw=mean_mw,
                        exit_area_ratio=exit_area_ratio,
                        exit_x=contour.x[-1],
                        exit_diameter=contour.r[-1] * 2
                    )
                except Exception as e:
                    print(f"Plume rendering skipped: {e}")

            # Reset camera
            self._plotter.reset_camera()
            self._plotter.view_isometric()

    def _add_plume(
        self,
        gamma: float,
        T_chamber: float,
        P_chamber: float,
        mean_mw: float,
        exit_area_ratio: float,
        exit_x: float,
        exit_diameter: float,
        P_ambient: float = 0.0
    ):
        """
        Add exhaust plume visualization.

        Phase 6: Hyper-Visualization feature.
        """
        if not PLUME_AVAILABLE or not self._plotter:
            return

        # Calculate exit conditions
        from src.core.propulsion import (
            calculate_exit_conditions,
            solve_mach_from_area_ratio_supersonic,
        )

        M_exit = solve_mach_from_area_ratio_supersonic(exit_area_ratio, gamma)
        T_exit, P_exit, _ = calculate_exit_conditions(gamma, T_chamber, P_chamber, M_exit)

        # Calculate exit velocity
        R_specific = 8314.46 / mean_mw
        V_exit = M_exit * np.sqrt(gamma * R_specific * T_exit)

        # Create plume conditions
        conditions = PlumeConditions(
            P_exit=P_exit,
            P_ambient=P_ambient,
            M_exit=M_exit,
            T_exit=T_exit,
            V_exit=V_exit,
            gamma=gamma,
            exit_diameter=exit_diameter
        )

        # Render plume
        renderer = PlumeRenderer()
        plume = renderer.generate_plume_surface(conditions, plume_length=exit_diameter * 6)

        # Translate to nozzle exit position
        plume.translate([exit_x, 0, 0], inplace=True)

        # Add to scene with transparency
        self._plotter.add_mesh(
            plume,
            scalars='Mach',
            cmap='plasma',
            opacity=0.5,
            show_scalar_bar=False,
            name='exhaust_plume'
        )

        self._plume_mesh = plume

    def toggle_plume(self, show: bool = True):
        """Toggle plume visibility."""
        self._show_plume = show

    def _create_nozzle_mesh(
        self,
        x: np.ndarray,
        r: np.ndarray,
        profile: dict,
        color_by: str
    ):
        """
        Create a 3D surface mesh by revolving the nozzle contour.

        Args:
            x: Axial coordinates
            r: Radius at each x
            profile: Flow property profile
            color_by: Field to use for coloring
        """
        try:
            n_theta = 36  # Angular resolution
            n_x = len(x)

            # Create theta array
            theta = np.linspace(0, 2 * np.pi, n_theta)

            # Create meshgrid
            X = np.zeros((n_x, n_theta))
            Y = np.zeros((n_x, n_theta))
            Z = np.zeros((n_x, n_theta))

            for i in range(n_x):
                X[i, :] = x[i]
                Y[i, :] = r[i] * np.cos(theta)
                Z[i, :] = r[i] * np.sin(theta)

            # Create structured grid
            mesh = pv.StructuredGrid(X, Y, Z)

            # Interpolate flow properties to mesh
            # Map profile data (which may have different resolution) to geometry
            if len(profile['temperature']) != n_x:
                # Interpolate if sizes don't match
                from scipy.interpolate import interp1d

                x_profile = np.linspace(x[0], x[-1], len(profile['temperature']))

                temp_interp = interp1d(x_profile, profile['temperature'],
                                       kind='linear', fill_value='extrapolate')
                mach_interp = interp1d(x_profile, profile['mach'],
                                       kind='linear', fill_value='extrapolate')

                temp_values = temp_interp(x)
                mach_values = mach_interp(x)
            else:
                temp_values = profile['temperature']
                mach_values = profile['mach']

            # Replicate values around circumference
            temp_field = np.tile(temp_values.reshape(-1, 1), (1, n_theta))
            mach_field = np.tile(mach_values.reshape(-1, 1), (1, n_theta))

            mesh['temperature'] = temp_field.flatten()
            mesh['mach'] = mach_field.flatten()

            return mesh

        except Exception as e:
            print(f"Error creating mesh: {e}")
            return None

    def export_stl(self, filepath: str, binary: bool = True) -> bool:
        """
        Export the current nozzle mesh to STL format.

        Args:
            filepath: Path to save the STL file
            binary: If True, save as binary STL (smaller file).
                   If False, save as ASCII STL.

        Returns:
            True if export successful, False otherwise.
        """
        if not PYVISTA_AVAILABLE:
            print("PyVista not available for STL export")
            return False

        if self._mesh is None:
            print("No mesh to export. Run simulation first.")
            return False

        try:
            # Ensure filepath has .stl extension
            if not filepath.lower().endswith('.stl'):
                filepath = filepath + '.stl'

            # PyVista StructuredGrid needs to be converted to surface
            # Extract the surface as a PolyData
            surface = self._mesh.extract_surface()

            # Save as STL
            surface.save(filepath, binary=binary)

            print(f"Exported mesh to {filepath}")
            return True

        except Exception as e:
            print(f"Error exporting STL: {e}")
            return False

    def export_ply(self, filepath: str) -> bool:
        """
        Export the current nozzle mesh to PLY format (with colors).

        Args:
            filepath: Path to save the PLY file

        Returns:
            True if export successful, False otherwise.
        """
        if not PYVISTA_AVAILABLE or self._mesh is None:
            return False

        try:
            if not filepath.lower().endswith('.ply'):
                filepath = filepath + '.ply'

            surface = self._mesh.extract_surface()
            surface.save(filepath)

            print(f"Exported mesh to {filepath}")
            return True

        except Exception as e:
            print(f"Error exporting PLY: {e}")
            return False

    def export_obj(self, filepath: str) -> bool:
        """
        Export the current nozzle mesh to OBJ format.

        Args:
            filepath: Path to save the OBJ file

        Returns:
            True if export successful, False otherwise.
        """
        if not PYVISTA_AVAILABLE or self._mesh is None:
            return False

        try:
            if not filepath.lower().endswith('.obj'):
                filepath = filepath + '.obj'

            surface = self._mesh.extract_surface()
            surface.save(filepath)

            print(f"Exported mesh to {filepath}")
            return True

        except Exception as e:
            print(f"Error exporting OBJ: {e}")
            return False

    def has_mesh(self) -> bool:
        """Check if a mesh is available for export."""
        return self._mesh is not None

    def update_attitude(self, q: np.ndarray):
        """
        Update the 3D model orientation based on quaternion.

        Args:
            q: Quaternion [w, x, y, z] representing attitude
        """
        if not PYVISTA_AVAILABLE or not self._plotter or self._mesh is None:
            return

        try:
            # Convert quaternion to rotation matrix
            # q = [w, x, y, z] format
            w, x, y, z = q[0], q[1], q[2], q[3]

            # Normalize quaternion
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            if norm < 1e-10:
                return
            w, x, y, z = w/norm, x/norm, y/norm, z/norm

            # Convert to rotation matrix (3x3)
            rot_matrix = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ])

            # Create 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix

            # Apply transformation to mesh copy
            transformed_mesh = self._mesh.copy()
            transformed_mesh.transform(transform, inplace=True)

            # Update display (clear and re-add)
            self._plotter.clear()
            self._plotter.add_axes(color='#888888')
            self._plotter.add_mesh(
                transformed_mesh,
                color='#00d4ff',
                show_edges=False,
                smooth_shading=True
            )

        except Exception as e:
            print(f"Attitude update error: {e}")

    def clear(self):
        """Clear the 3D view."""
        if self._plotter:
            self._plotter.clear()
            self._plotter.add_axes(color='#888888')
            self._add_placeholder_text()

    def closeEvent(self, event):
        """Clean up on close."""
        if self._plotter:
            self._plotter.close()
        super().closeEvent(event)

