"""Nozzle in 3D: a mesh exporter that needs only NumPy, and an optional viewer.

The viewer uses PyVista/VTK, which is heavy and needs working OpenGL, so it
is created only when the user asks for it and opens in its own window.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from .. import theme
from .controls import Choice


def revolve(x: np.ndarray, r: np.ndarray, segments: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Surface of revolution about the x axis: (vertices, triangle indices)."""
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    vertices = np.column_stack(
        [
            np.repeat(x, segments),
            np.outer(r, np.cos(theta)).ravel(),
            np.outer(r, np.sin(theta)).ravel(),
        ]
    )
    faces = []
    n = len(x)
    for i in range(n - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + j
            d = (i + 1) * segments + (j + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    return vertices, np.array(faces, dtype=np.int64)


def revolve_to_stl(x: np.ndarray, r: np.ndarray, path: str | Path, segments: int = 96) -> None:
    """Write the wall surface as a binary STL (metres)."""
    vertices, faces = revolve(np.asarray(x), np.asarray(r), segments)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(
        normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0
    )
    with Path(path).open("wb") as handle:
        handle.write(b"EnSim nozzle wall".ljust(80, b" "))
        handle.write(struct.pack("<I", len(faces)))
        records = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        records["n"] = normals
        records["v"] = triangles
        handle.write(records.tobytes())


def viewer_available() -> tuple[bool, str]:
    if os.environ.get("ENSIM_DISABLE_3D", "").lower() in {"1", "true", "yes"}:
        return False, "3D rendering is disabled (ENSIM_DISABLE_3D)."
    if os.environ.get("QT_QPA_PLATFORM", "").lower() in {"offscreen", "minimal"}:
        return False, "3D rendering needs a display."
    try:
        import pyvista  # noqa: F401
        import pyvistaqt  # noqa: F401
    except ImportError:
        return False, 'The 3D viewer needs PyVista: pip install "ensim[gui]"'
    return True, ""


class Nozzle3DDialog(QDialog):
    def __init__(self, parent: QWidget, run) -> None:
        super().__init__(parent)
        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.setWindowTitle("Nozzle, 3D")
        self.resize(900, 620)
        self._run = run
        p = theme.current()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setContentsMargins(14, 8, 14, 8)
        bar.addWidget(QLabel("Colour by"))
        self.field = Choice(
            [("Mach number", "mach"), ("Temperature", "temperature"), ("Pressure", "pressure")]
        )
        bar.addWidget(self.field)
        bar.addStretch()
        layout.addLayout(bar)
        self.plotter = QtInteractor(self)
        self.plotter.set_background(p.bg)
        layout.addWidget(self.plotter.interactor, 1)
        vertices, faces = revolve(run.contour.x, run.contour.r, 72)
        cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
        self.mesh = pv.PolyData(vertices, cells)
        self.field.currentIndexChanged.connect(self._paint)
        self._paint()

    def _paint(self) -> None:
        key = self.field.value()
        values = np.repeat(self._run.profile[key], 72)
        self.mesh.point_data[key] = values
        self.plotter.clear()
        self.plotter.add_mesh(
            self.mesh,
            scalars=key,
            cmap="inferno" if key == "temperature" else "cividis",
            smooth_shading=True,
            scalar_bar_args={"title": key, "color": theme.current().muted},
        )
        self.plotter.view_isometric()
        self.plotter.reset_camera()

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)


def show_nozzle_3d(parent: QWidget, run) -> None:
    ok, reason = viewer_available()
    if not ok:
        QMessageBox.information(parent, "3D view", reason)
        return
    try:
        dialog = Nozzle3DDialog(parent, run)
    except Exception as error:  # OpenGL problems surface here on old drivers
        QMessageBox.warning(parent, "3D view", f"The 3D view could not start: {error}")
        return
    dialog.show()
