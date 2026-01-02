"""
3D Vehicle Rendering Module.

Generates PyVista meshes for rocket components
for interactive 3D visualization.
"""


import numpy as np

# Try to import PyVista
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from src.core.rocket import NoseShape, Rocket


def generate_nose_mesh(
    shape: NoseShape,
    length: float,
    diameter: float,
    resolution: int = 32
):
    """
    Generate nose cone mesh.

    Args:
        shape: Nose cone shape type
        length: Nose length (m)
        diameter: Base diameter (m)
        resolution: Angular resolution

    Returns:
        PyVista mesh or None
    """
    if not PYVISTA_AVAILABLE:
        return None

    R = diameter / 2
    n_axial = 20

    # Generate profile based on shape
    x = np.linspace(0, length, n_axial)

    if shape == NoseShape.CONICAL:
        # Linear profile
        r = R * x / length
    elif shape == NoseShape.OGIVE:
        # Tangent ogive
        rho = (R**2 + length**2) / (2 * R)  # Ogive radius
        r = np.sqrt(rho**2 - (length - x)**2) - (rho - R)
        r = np.clip(r, 0, R)
    elif shape == NoseShape.ELLIPTICAL:
        # Ellipse profile
        r = R * np.sqrt(1 - (1 - x/length)**2)
    elif shape == NoseShape.PARABOLIC:
        # Parabolic profile
        r = R * np.sqrt(x / length)
    else:
        # Default to ogive
        r = R * x / length

    # Generate surface of revolution
    theta = np.linspace(0, 2*np.pi, resolution)

    # Create points
    points = []
    for i in range(n_axial):
        for j in range(resolution):
            px = x[i]
            py = r[i] * np.cos(theta[j])
            pz = r[i] * np.sin(theta[j])
            points.append([px, py, pz])

    points = np.array(points)

    # Create faces
    faces = []
    for i in range(n_axial - 1):
        for j in range(resolution - 1):
            p1 = i * resolution + j
            p2 = i * resolution + j + 1
            p3 = (i + 1) * resolution + j + 1
            p4 = (i + 1) * resolution + j
            faces.append([4, p1, p2, p3, p4])

    faces = np.hstack(faces)

    mesh = pv.PolyData(points, faces)
    return mesh


def generate_body_mesh(
    length: float,
    diameter: float,
    offset: float = 0.0,
    resolution: int = 32
):
    """
    Generate body tube mesh (cylinder).

    Args:
        length: Tube length (m)
        diameter: Outer diameter (m)
        offset: Axial offset from origin (m)
        resolution: Angular resolution

    Returns:
        PyVista mesh or None
    """
    if not PYVISTA_AVAILABLE:
        return None

    # Use PyVista cylinder
    cylinder = pv.Cylinder(
        center=(offset + length/2, 0, 0),
        direction=(1, 0, 0),
        radius=diameter/2,
        height=length,
        resolution=resolution
    )

    return cylinder


def generate_fin_mesh(
    root_chord: float,
    tip_chord: float,
    span: float,
    sweep_angle: float,
    thickness: float,
    body_radius: float,
    axial_position: float,
    fin_index: int,
    total_fins: int
):
    """
    Generate single fin mesh.

    Args:
        root_chord, tip_chord, span, sweep_angle: Fin geometry
        thickness: Fin thickness (m)
        body_radius: Body tube radius (m)
        axial_position: Fin root LE position from nose (m)
        fin_index: Which fin (0, 1, 2, ...)
        total_fins: Total number of fins

    Returns:
        PyVista mesh or None
    """
    if not PYVISTA_AVAILABLE:
        return None

    # Fin profile (2D, in XY plane)
    sweep_offset = np.tan(np.radians(sweep_angle)) * span

    # Points: LE root, LE tip, TE tip, TE root
    profile = np.array([
        [axial_position, body_radius, 0],
        [axial_position + sweep_offset, body_radius + span, 0],
        [axial_position + sweep_offset + tip_chord, body_radius + span, 0],
        [axial_position + root_chord, body_radius, 0]
    ])

    # Create extruded polygon
    # Offset for thickness
    top_profile = profile.copy()
    top_profile[:, 2] = thickness / 2
    bottom_profile = profile.copy()
    bottom_profile[:, 2] = -thickness / 2

    # Combine
    points = np.vstack([bottom_profile, top_profile])

    # Faces: bottom, top, and sides
    faces = []
    # Bottom face
    faces.extend([4, 0, 1, 2, 3])
    # Top face
    faces.extend([4, 4, 7, 6, 5])
    # Side faces
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([4, i, j, j+4, i+4])

    mesh = pv.PolyData(points, faces)

    # Rotate around X axis for fin position
    angle = 360 / total_fins * fin_index
    mesh = mesh.rotate_x(angle, point=(axial_position, 0, 0))

    return mesh


def generate_rocket_mesh(rocket: Rocket):
    """
    Generate complete rocket mesh from all components.

    Args:
        rocket: Rocket configuration

    Returns:
        Dict with component meshes and combined mesh
    """
    if not PYVISTA_AVAILABLE:
        return None

    meshes = {}

    # Nose cone
    nose_mesh = generate_nose_mesh(
        shape=rocket.nose.shape,
        length=rocket.nose.length,
        diameter=rocket.nose.diameter
    )
    if nose_mesh:
        meshes['nose'] = nose_mesh

    # Body tube
    body_mesh = generate_body_mesh(
        length=rocket.body.length,
        diameter=rocket.body.diameter,
        offset=rocket.nose.length
    )
    if body_mesh:
        meshes['body'] = body_mesh

    # Fins
    fin_meshes = []
    for i in range(rocket.fins.count):
        fin_mesh = generate_fin_mesh(
            root_chord=rocket.fins.fin.root_chord,
            tip_chord=rocket.fins.fin.tip_chord,
            span=rocket.fins.fin.span,
            sweep_angle=rocket.fins.fin.sweep_angle,
            thickness=rocket.fins.fin.thickness,
            body_radius=rocket.body.diameter / 2,
            axial_position=rocket.fins.position,
            fin_index=i,
            total_fins=rocket.fins.count
        )
        if fin_mesh:
            fin_meshes.append(fin_mesh)

    if fin_meshes:
        meshes['fins'] = fin_meshes

    # Combine all meshes
    combined = None
    for key, mesh in meshes.items():
        if key == 'fins':
            for fm in mesh:
                combined = fm if combined is None else combined.merge(fm)
        else:
            combined = mesh if combined is None else combined.merge(mesh)

    meshes['combined'] = combined

    return meshes


def create_rocket_plotter(rocket: Rocket, background: str = '#1e1e1e'):
    """
    Create a PyVista plotter with rocket visualization.

    Args:
        rocket: Rocket configuration
        background: Background color

    Returns:
        PyVista Plotter or None
    """
    if not PYVISTA_AVAILABLE:
        return None

    meshes = generate_rocket_mesh(rocket)
    if not meshes:
        return None

    plotter = pv.Plotter()
    plotter.set_background(background)

    # Add components with colors
    if 'nose' in meshes:
        plotter.add_mesh(meshes['nose'], color='#ffffff',
                        smooth_shading=True, label='Nose')

    if 'body' in meshes:
        plotter.add_mesh(meshes['body'], color='#cccccc',
                        smooth_shading=True, label='Body')

    if 'fins' in meshes:
        for fm in meshes['fins']:
            plotter.add_mesh(fm, color='#ff4444',
                           smooth_shading=True)

    plotter.add_axes(color='#888888')
    plotter.view_isometric()

    return plotter
