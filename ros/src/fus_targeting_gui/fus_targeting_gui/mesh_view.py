"""Mesh loading, rendering, and surface point-picking. This module has no
ROS dependency and is runnable and testable standalone:

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
_AREA_BOUNDARY_MARKERS_ACTOR_NAME = "area_boundary_markers"
_PREDEFINED_POINTS_ACTOR_NAME = "predefined_points"
_TARGET_MARKERS_ACTOR_NAME = "target_markers"
_TARGET_LABELS_ACTOR_NAME = "target_labels"


class MeshView(QWidget):
    """Emits point_picked(point_xyz: tuple, normal_xyz: tuple) every time
    the user clicks a point on the mesh surface. Both values are in the
    mesh's own local, pre-registration coordinate frame."""

    point_picked = Signal(tuple, tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mesh = None  # the currently displayed/pickable mesh (possibly clipped)
        self._original_mesh = None  # the full, unclipped mesh as loaded from disk
        self._clip_fractions = {"top": 0.0, "bottom": 0.0, "right": 0.0, "left": 0.0}
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
        self._original_mesh = mesh
        self._clip_fractions = {"top": 0.0, "bottom": 0.0, "right": 0.0, "left": 0.0}
        self._mesh = mesh

        self.interactor.clear()
        # QtInteractor (unlike a plain pv.Plotter(), which auto-configures
        # 5 lights) starts with zero lights, and clear() wipes any that
        # were added. Confirmed directly (renderer.lights was empty). This
        # caused the mesh to render as a flat, gradient-less silhouette
        # despite correct material params below. enable_lightkit() adds
        # the same 5-light rig a plain Plotter() gets by default; safe and
        # idempotent to call on every load_mesh().
        self.interactor.enable_lightkit()
        self._add_or_replace_skull_actor(mesh)
        self.interactor.add_axes()
        self.interactor.reset_camera()
        # reset_camera() alone frames the bounds face-on, which reads as
        # flat with no perspective foreshortening to hint at depth. Swing
        # to an oblique 3/4 view by default so curvature is visible
        # without the user having to discover they need to rotate first.
        self.interactor.camera.azimuth = 35
        self.interactor.camera.elevation = 20

        # load_mesh() can be called more than once (startup, any "Load
        # Mesh..." click, or a clip-slider rebuild). pyvista refuses to
        # enable_surface_point_picking a second time without disabling the
        # previous one first ("Picking is already enabled..."). Safe to
        # call even the first time, when nothing was enabled yet.
        self._enable_picking()

    def _add_or_replace_skull_actor(self, mesh):
        """(Re)adds the main skull-mesh actor with its material params.
        Shared by load_mesh() and _rebuild_clipped_mesh() so both apply
        identical shading. Reusing name="skull_mesh" replaces the actor in
        place if one already exists, avoiding a full clear()."""
        self.interactor.add_mesh(
            mesh, color=(0.9, 0.85, 0.8), smooth_shading=True, name="skull_mesh",
            # Explicit material params rather than pyvista's flatter
            # defaults: a specular highlight reads as "curved 3D surface"
            # instead of "flat painted silhouette" to the eye, especially
            # under a single default light.
            ambient=0.15, diffuse=0.8, specular=0.5, specular_power=20,
        )

    def _enable_picking(self):
        self.interactor.disable_picking()
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

    def get_collision_mesh_data(self, max_triangles=2000):
        """Decimated (vertices, triangles) of the full, unclipped original
        mesh, for publishing as a MoveIt collision object (see
        MoveItBridge.set_skull_collision_object()). Deliberately always
        uses self._original_mesh, not the possibly-clipped self._mesh:
        clipping is a picking and visualization convenience (see
        set_clip_fraction()'s own docstring), not a claim that the
        physical object has less material there. The planner should stay
        aware of the whole skull regardless of what's currently clipped
        away for viewing. Decimated because FCL collision checking against
        the full ~200k-triangle mesh would be too slow for interactive
        planning; ~2000 triangles is enough to represent gross shape for
        avoidance purposes (verified directly: bounds stay accurate to
        <1mm after decimating this mesh to 2000 triangles).

        Returns:
            (vertices, triangles) as plain numpy arrays: vertices (N, 3)
            float, triangles (M, 3) int indices into vertices. Both in the
            mesh's own local (already-scaled) frame; the caller transforms
            into base_frame (see
            Registration.transform_points_to_base_frame()) before building
            the CollisionObject message, keeping this method, and this
            whole file, free of any ROS dependency.
        """
        if self._original_mesh is None:
            return None, None
        mesh = self._original_mesh.triangulate()
        if mesh.n_cells > max_triangles:
            target_reduction = 1.0 - (max_triangles / mesh.n_cells)
            mesh = mesh.decimate(target_reduction)
        vertices = np.asarray(mesh.points, dtype=float)
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]  # pyvista's flat [3, i, j, k, 3, i, j, k, ...]
        return vertices, np.asarray(faces, dtype=int)

    def get_mesh_centroid(self):
        """Vertex centroid of the full, unclipped original mesh, in the
        mesh's own local (already-scaled) frame. Used by search_area.py's
        volume-probing grid as the inner reference point a 3D probing
        column extends toward, so it is always available from the same
        mesh get_collision_mesh_data() already reads.

        Returns:
            (3,) numpy array, or None if no mesh is loaded.
        """
        if self._original_mesh is None:
            return None
        return np.asarray(self._original_mesh.points, dtype=float).mean(axis=0)

    def set_clip_fraction(self, axis: str, fraction: float):
        """Progressively clips away part of the mesh so obstructing
        geometry can be moved out of the way for picking. axis is "top"
        (cuts down from the +Z/superior end), "bottom" (cuts up from the
        -Z/inferior end), "right" or "left" (cuts inward from the mesh's
        own +X/-X sides; these are the mesh's own local coordinate
        frame's sides, not independently verified against true anatomical
        left/right). fraction is 0.0 (no cut) to 1.0 (capped at 0.95
        internally so the mesh can never fully vanish). Multiple axes
        combine (e.g. cut from both top and right at once).
        Rebuilds the displayed mesh and re-registers picking against it,
        but deliberately does not touch the camera: adjusting a slider
        should not reset the user's current view."""
        if self._original_mesh is None:
            return
        self._clip_fractions[axis] = max(0.0, min(fraction, 0.95))
        self._rebuild_clipped_mesh()

    def reset_clipping(self):
        """Zeroes every clip axis and rebuilds once, cheaper than calling
        set_clip_fraction() per axis, which would rebuild redundantly."""
        if self._original_mesh is None:
            return
        self._clip_fractions = {axis: 0.0 for axis in self._clip_fractions}
        self._rebuild_clipped_mesh()

    def _rebuild_clipped_mesh(self):
        mesh = self._original_mesh
        xmin, xmax, _ymin, _ymax, zmin, zmax = mesh.bounds

        top_frac = self._clip_fractions["top"]
        if top_frac > 0.0:
            origin_z = zmax - top_frac * (zmax - zmin)
            # invert=True keeps the side the normal points away from.
            # normal=+Z keeps the lower/inferior part, i.e. cuts more of
            # the top away as top_frac increases (verified directly
            # against this mesh before implementing this).
            mesh = mesh.clip(normal="z", origin=(0, 0, origin_z), invert=True)

        bottom_frac = self._clip_fractions["bottom"]
        if bottom_frac > 0.0:
            origin_z = zmin + bottom_frac * (zmax - zmin)
            # invert=False keeps the side the normal points toward. Same
            # normal=+Z as "top" above, opposite invert, so this keeps the
            # upper/superior part instead, cutting the bottom away.
            mesh = mesh.clip(normal="z", origin=(0, 0, origin_z), invert=False)

        right_frac = self._clip_fractions["right"]
        if right_frac > 0.0:
            origin_x = xmax - right_frac * (xmax - xmin)
            mesh = mesh.clip(normal="x", origin=(origin_x, 0, 0), invert=True)

        left_frac = self._clip_fractions["left"]
        if left_frac > 0.0:
            origin_x = xmin + left_frac * (xmax - xmin)
            mesh = mesh.clip(normal="x", origin=(origin_x, 0, 0), invert=False)

        if mesh.n_points == 0:
            # All active clips together removed everything. Fall back to
            # the unclipped mesh rather than displaying and picking
            # against nothing silently.
            mesh = self._original_mesh
        else:
            mesh = mesh.compute_normals(
                cell_normals=False, point_normals=True, auto_orient_normals=True
            )

        self._mesh = mesh
        self._add_or_replace_skull_actor(mesh)
        self._enable_picking()

    def _on_pick(self, picked_point):
        if self._mesh is None or picked_point is None:
            return
        point = np.asarray(picked_point, dtype=float)
        normal = self.nearest_surface_normal(point)
        self.point_picked.emit(tuple(point.tolist()), normal)

    def nearest_surface_normal(self, point_local):
        """Nearest-vertex normal lookup, rather than trusting a specific
        pyvista picker callback signature to hand back a normal directly.
        Robust across pyvista versions, close enough for a surface pick or
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
        find where it intersects the mesh surface. Used to snap a
        flat-plane search-area grid point (see search_area.py) onto the
        true curved surface.

        Returns:
            (surface_point, surface_normal), or None if the ray misses the
            mesh entirely within max_distance_m in either direction (e.g.
            the plane point falls outside the mesh's silhouette from this
            direction, which is routine near an irregular boundary, not
            an error).
        """
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
        # (endocranial) surface, so a ray can hit twice. Take whichever
        # intersection is closest to the original plane point, which is
        # reliably the near/outer surface since the boundary points that
        # define the plane were themselves picked on that outer surface.
        dists = np.linalg.norm(intersections - point, axis=1)
        surface_point = intersections[int(np.argmin(dists))]
        surface_normal = self.nearest_surface_normal(surface_point)
        return tuple(surface_point.tolist()), surface_normal

    def show_predefined_points(self, points_local):
        """Draws a small cyan marker at each of a list of (x, y, z) points
        (mesh-local frame), as visual context for the "Predefined Points"
        list in main_window.py. Selection happens via that list, not by
        clicking these markers, since small markers are unreliable to
        click precisely."""
        if not points_local:
            return
        cloud = pv.PolyData(np.asarray(points_local, dtype=float))
        self.interactor.add_mesh(
            cloud, color="cyan", point_size=12, render_points_as_spheres=True,
            name=_PREDEFINED_POINTS_ACTOR_NAME,
        )

    def update_target_markers(self, points_local, labels):
        """Persistent small numbered markers for every target currently in
        the Picked Targets list. Distinct from update_target_preview(),
        which only shows the live alignment-parameter preview for the
        single currently-selected target. Without this, a previously-added
        target has no visible trace on the mesh once you pick a new one or
        select something else, which is confusing ("where did I even pick
        points?") on a mesh with no other visual landmarks. Call with the
        full current set each time, not incrementally: cheap to redraw,
        and avoids tracking per-target actor names."""
        try:
            self.interactor.remove_actor(_TARGET_MARKERS_ACTOR_NAME)
            self.interactor.remove_actor(_TARGET_LABELS_ACTOR_NAME)
        except Exception:
            pass
        if not points_local:
            return
        points = np.asarray(points_local, dtype=float)
        cloud = pv.PolyData(points)
        self.interactor.add_mesh(
            cloud, color="magenta", point_size=10, render_points_as_spheres=True,
            name=_TARGET_MARKERS_ACTOR_NAME,
        )
        self.interactor.add_point_labels(
            points, labels, point_size=0, font_size=14, text_color="magenta",
            shape=None, always_visible=True, name=_TARGET_LABELS_ACTOR_NAME,
        )

    def update_area_boundary_preview(self, boundary_points_local):
        """Draws the in-progress search-area boundary: a sphere marker at
        every point picked so far, visible even after just the first
        click. A thin line alone, sitting directly on the surface it was
        picked from, z-fights against the mesh and can render as barely
        visible specks (confirmed directly by screenshot); spheres do not
        have that problem, same as update_target_markers(). Also draws a
        closed connecting polyline once there are >= 2 points."""
        for name in (_AREA_BOUNDARY_ACTOR_NAME, _AREA_BOUNDARY_MARKERS_ACTOR_NAME):
            try:
                self.interactor.remove_actor(name)
            except Exception:
                pass
        if not boundary_points_local:
            return

        pts = np.asarray(boundary_points_local, dtype=float)
        cloud = pv.PolyData(pts)
        self.interactor.add_mesh(
            cloud, color="orange", point_size=14, render_points_as_spheres=True,
            name=_AREA_BOUNDARY_MARKERS_ACTOR_NAME,
        )

        if len(pts) >= 2:
            closed = np.vstack([pts, pts[0]])  # close the loop for the preview only
            line = pv.MultipleLines(points=closed)
            self.interactor.add_mesh(
                line, color="orange", line_width=5, name=_AREA_BOUNDARY_ACTOR_NAME
            )

    def clear_area_boundary_preview(self):
        for name in (_AREA_BOUNDARY_ACTOR_NAME, _AREA_BOUNDARY_MARKERS_ACTOR_NAME):
            try:
                self.interactor.remove_actor(name)
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
        the mesh's own local (already-scaled) frame, as a visual aid for
        dialing in tilt/azimuth/roll/standoff, not the pose planned and
        executed (that is computed separately in registration.py against
        the robot's base frame; see its own note on why the two can
        differ slightly, and harmlessly, in azimuth/roll reference)."""
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
    view.setWindowTitle(f"MeshView standalone: {path}")
    view.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _standalone_main()
