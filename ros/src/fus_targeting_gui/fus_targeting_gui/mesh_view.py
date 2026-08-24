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

from .geometry_utils import look_at_basis, tilt_direction

_PREVIEW_ACTOR_NAMES = ("preview_shaft", "preview_x", "preview_y", "preview_z")
_AREA_BOUNDARY_ACTOR_NAME = "area_boundary_preview"
_PREDEFINED_POINTS_ACTOR_NAME = "predefined_points"


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
            left_clicking=True,  # a plain left-click picks; PyVista's default
            # requires pressing 'P' while hovering instead, which doesn't
            # match this GUI's own on-screen instruction text.
        )

    def _on_pick(self, picked_point):
        if self._mesh is None or picked_point is None:
            return
        point = np.asarray(picked_point, dtype=float)
        normal = self.nearest_surface_normal(point)
        self.point_picked.emit(tuple(point.tolist()), normal)

    def nearest_surface_normal(self, point_local):
        """Nearest-vertex normal lookup, rather than trusting a specific
        pyvista picker callback signature to hand back a normal directly --
        robust across pyvista versions, close enough for a surface pick or
        a predefined point without its own normal (mesh is fine enough
        that adjacent-vertex normal differences are negligible here).
        Returns (0, 0, 1) if no mesh is loaded."""
        if self._mesh is None:
            return (0.0, 0.0, 1.0)
        point = np.asarray(point_local, dtype=float)
        closest_idx = self._mesh.find_closest_point(point)
        normal = np.asarray(self._mesh.point_normals[closest_idx], dtype=float)
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        return tuple(normal.tolist())

    def raycast_onto_surface(self, point_local, direction_local, max_distance_m=0.5):
        """Casts a ray through `point_local` along +-`direction_local` to
        find where it actually intersects the mesh surface -- used to snap
        a flat-plane search-area grid point (see search_area.py) onto the
        true curved surface. Returns (surface_point, surface_normal), or
        None if the ray misses the mesh entirely within max_distance_m in
        either direction (e.g. the plane point falls outside the mesh's
        silhouette from this direction -- routine near an irregular
        boundary, not an error)."""
        if self._mesh is None:
            return None
        point = np.asarray(point_local, dtype=float)
        direction = np.asarray(direction_local, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None
        direction = direction / norm

        start = point - direction * max_distance_m
        end = point + direction * max_distance_m
        intersections, _cell_ids = self._mesh.ray_trace(start, end, first_point=False)
        if len(intersections) == 0:
            return None

        # A CT-derived skull mesh commonly has both an outer and an inner
        # (endocranial) surface, so a ray can hit twice -- take whichever
        # intersection is closest to the original plane point, which is
        # reliably the near/outer surface since the boundary points that
        # define the plane were themselves picked on that outer surface.
        dists = np.linalg.norm(intersections - point, axis=1)
        surface_point = intersections[int(np.argmin(dists))]
        surface_normal = self.nearest_surface_normal(surface_point)
        return tuple(surface_point.tolist()), surface_normal

    def show_predefined_points(self, points_local):
        """Draws a small cyan marker at each of a list of (x, y, z) points
        (mesh-local frame) -- pure visual context for the "Predefined
        Points" list in main_window.py; selection itself happens via that
        list, not by clicking these markers (small markers are unreliable
        to click precisely)."""
        if not points_local:
            return
        cloud = pv.PolyData(np.asarray(points_local, dtype=float))
        self.interactor.add_mesh(
            cloud, color="cyan", point_size=12, render_points_as_spheres=True,
            name=_PREDEFINED_POINTS_ACTOR_NAME,
        )

    def update_area_boundary_preview(self, boundary_points_local):
        """Draws the in-progress search-area boundary as a closed polyline
        connecting the points picked so far, in order."""
        try:
            self.interactor.remove_actor(_AREA_BOUNDARY_ACTOR_NAME)
        except Exception:
            pass
        if len(boundary_points_local) < 2:
            return
        pts = np.asarray(boundary_points_local, dtype=float)
        closed = np.vstack([pts, pts[0]])  # close the loop for the preview only
        line = pv.MultipleLines(points=closed)
        self.interactor.add_mesh(
            line, color="orange", line_width=3, name=_AREA_BOUNDARY_ACTOR_NAME
        )

    def clear_area_boundary_preview(self):
        try:
            self.interactor.remove_actor(_AREA_BOUNDARY_ACTOR_NAME)
        except Exception:
            pass

    def clear_target_preview(self):
        for name in _PREVIEW_ACTOR_NAMES:
            try:
                self.interactor.remove_actor(name)
            except Exception:
                pass

    def update_target_preview(
        self, point_local, normal_local, tilt_deg, azimuth_deg, roll_deg, standoff_mm
    ):
        """Live preview of the probe alignment at a target: a shaft from
        the commanded probe position back to the picked surface point, plus
        an RGB (x/y/z) frame at the probe position showing the commanded
        end-effector orientation, including roll. All computed directly in
        the mesh's own local (already-scaled) frame -- a visual aid for
        dialing in tilt/azimuth/roll/standoff, not the pose actually
        planned/executed (that's computed separately in registration.py
        against the robot's base frame; see its own note on why the two
        can differ slightly in azimuth/roll reference, harmlessly)."""
        if self._mesh is None:
            return
        point = np.asarray(point_local, dtype=float)
        normal = np.asarray(normal_local, dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return
        normal = normal / norm

        approach_dir = tilt_direction(-normal, tilt_deg, azimuth_deg)
        standoff_m = standoff_mm / 1000.0
        probe_position = point - approach_dir * standoff_m

        if np.linalg.norm(point - probe_position) > 1e-6:
            shaft = pv.Line(probe_position, point)
            self.interactor.add_mesh(
                shaft, color="yellow", line_width=4, name="preview_shaft"
            )
        else:
            try:
                self.interactor.remove_actor("preview_shaft")
            except Exception:
                pass

        x_axis, y_axis, z_axis = look_at_basis(approach_dir, roll_deg=roll_deg)
        axis_len = max(standoff_m * 0.5, 0.008)
        for axis_vec, color, name in (
            (x_axis, "red", "preview_x"),
            (y_axis, "green", "preview_y"),
            (z_axis, "blue", "preview_z"),
        ):
            arrow = pv.Arrow(start=probe_position, direction=axis_vec, scale=axis_len)
            self.interactor.add_mesh(arrow, color=color, name=name)


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
