"""Mesh loading + rendering + surface point-picking. No ROS dependency at
all -- runnable and testable standalone:

    python3 -m fus_targeting_gui.mesh_view /path/to/mesh.stl
"""

import sys

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget


class MeshView(QWidget):
    """Emits point_picked(point_xyz: tuple, normal_xyz: tuple) -- both in
    the mesh's own local (pre-registration) coordinate frame -- every time
    the user clicks a point on the mesh surface."""

    point_picked = Signal(tuple, tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mesh = None
        self._picked_actor = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.interactor = QtInteractor(self)
        layout.addWidget(self.interactor)

    def load_mesh(self, path: str, scale: float = 1.0):
        mesh = pv.read(path)
        if scale != 1.0:
            mesh = mesh.scale([scale, scale, scale], inplace=False)
        if mesh.point_normals is None or len(mesh.point_normals) == 0:
            mesh = mesh.compute_normals(
                cell_normals=False, point_normals=True, auto_orient_normals=True
            )
        self._mesh = mesh

        self.interactor.clear()
        self.interactor.add_mesh(
            mesh, color=(0.9, 0.85, 0.8), smooth_shading=True, name="skull_mesh"
        )
        self.interactor.add_axes()
        self.interactor.reset_camera()

        self.interactor.enable_surface_point_picking(
            callback=self._on_pick,
            show_message="Click a point on the mesh to select a target.",
            color="red",
            point_size=14,
            show_point=True,
            pickable_window=False,
        )

    def _on_pick(self, picked_point):
        if self._mesh is None or picked_point is None:
            return
        point = np.asarray(picked_point, dtype=float)

        # Nearest-vertex normal rather than trusting a specific pyvista
        # picker callback signature to hand back a normal directly --
        # robust across pyvista versions, close enough for a surface pick
        # (mesh is fine enough that adjacent-vertex normal differences are
        # negligible for targeting purposes).
        closest_idx = self._mesh.find_closest_point(point)
        normal = np.asarray(self._mesh.point_normals[closest_idx], dtype=float)
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm

        self.point_picked.emit(tuple(point.tolist()), tuple(normal.tolist()))


def _standalone_main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m fus_targeting_gui.mesh_view <mesh_path> [scale]")
        sys.exit(1)
    path = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    app = QApplication(sys.argv)
    view = MeshView()
    view.point_picked.connect(
        lambda point, normal: print(f"picked point={point}  normal={normal}")
    )
    view.load_mesh(path, scale=scale)
    view.resize(1000, 800)
    view.setWindowTitle(f"MeshView standalone -- {path}")
    view.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _standalone_main()
