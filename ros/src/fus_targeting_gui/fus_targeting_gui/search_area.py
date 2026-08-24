"""Search-area grid generation: given a polygon of boundary points picked
on the mesh surface (in picking order) and their surface normals, generates
a regular grid of interior sample points spaced `spacing_mm` apart across
the flat plane that best fits that boundary.

Deliberately pure numpy, no pyvista/ROS dependency -- independently
testable, and kept separate from the actual surface-snapping step (which
needs the real mesh and lives in mesh_view.py's raycast_onto_surface()):
this module only knows about the flat approximating plane, not the true
curved surface.
"""

import numpy as np

from .geometry_utils import orthonormal_basis


def _point_in_polygon_2d(point, polygon):
    """Standard ray-casting point-in-polygon test. `polygon` is an (N, 2)
    array of vertices in order (open -- do not repeat the first vertex).
    Boundary-exact points may go either way; irrelevant for grid sampling,
    where landing exactly on the boundary is a measure-zero edge case."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def generate_scan_grid(boundary_points_3d, boundary_normals_3d, spacing_mm):
    """boundary_points_3d: (x, y, z) tuples in the mesh's local (already-
    scaled, meters) frame, in picking order, forming a closed polygon loop
    -- do not repeat the first point at the end.
    boundary_normals_3d: matching unit-normal tuples at each boundary
    point; averaged to define the scan plane's normal.
    spacing_mm: grid spacing in millimeters.

    Returns (grid_points_3d, plane_normal): grid_points_3d are 3D points
    lying ON THE FLAT APPROXIMATING PLANE, not yet snapped to the true
    mesh surface -- the caller (main_window.py) raycasts each one onto the
    real mesh via mesh_view.raycast_onto_surface() along plane_normal.
    """
    pts = np.asarray(boundary_points_3d, dtype=float)
    normals = np.asarray(boundary_normals_3d, dtype=float)
    if len(pts) < 3:
        raise ValueError("Need at least 3 boundary points to define a search area.")
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be positive.")

    plane_normal = normals.mean(axis=0)
    norm = np.linalg.norm(plane_normal)
    if norm < 1e-9:
        raise ValueError("Boundary normals cancel out -- can't determine a scan plane.")
    plane_normal = plane_normal / norm

    origin = pts.mean(axis=0)
    u_axis, v_axis = orthonormal_basis(plane_normal)

    rel = pts - origin
    poly_uv = np.column_stack([rel @ u_axis, rel @ v_axis])

    spacing_m = spacing_mm / 1000.0
    u_min, v_min = poly_uv.min(axis=0)
    u_max, v_max = poly_uv.max(axis=0)

    n_u = max(int(np.floor((u_max - u_min) / spacing_m)) + 1, 1)
    n_v = max(int(np.floor((v_max - v_min) / spacing_m)) + 1, 1)
    u_values = u_min + np.arange(n_u) * spacing_m
    v_values = v_min + np.arange(n_v) * spacing_m

    grid_points_3d = []
    for u in u_values:
        for v in v_values:
            if _point_in_polygon_2d((u, v), poly_uv):
                point_3d = origin + u * u_axis + v * v_axis
                grid_points_3d.append(tuple(point_3d.tolist()))

    return grid_points_3d, tuple(plane_normal.tolist())
