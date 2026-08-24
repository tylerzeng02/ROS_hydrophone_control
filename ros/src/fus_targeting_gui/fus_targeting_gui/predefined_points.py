"""Loads user-authored target points (e.g. exported from CAD) from a CSV,
for selection via the GUI's "Predefined Points" list -- an alternative to
free-hand surface picking, for points that were already marked out ahead
of time rather than picked interactively.

No ROS/pyvista dependency -- just csv/dataclasses, kept independently
testable like geometry_utils.py.

CSV columns: label,x,y,z[,normal_x,normal_y,normal_z]
x/y/z are in the mesh's own NATIVE/raw units -- the same units the STL
file itself was authored in, e.g. mm from a CAD export, BEFORE
MeshConfig.scale is applied. NOT already-scaled meters. normal_x/y/z are
optional; if a row omits them (or the CSV has no such columns at all),
main_window.py resolves the normal from the nearest mesh vertex once the
mesh is loaded (same technique mesh_view.py's own surface picking uses).
"""

import csv
from dataclasses import dataclass


@dataclass
class PredefinedPoint:
    label: str
    point_local: tuple  # (x, y, z) already scaled to the mesh's local render frame (meters)
    normal_local: tuple | None  # None means "look up from the mesh once it's loaded"


def load_predefined_points_csv(path, scale):
    """`scale` is the same MeshConfig.scale factor mesh_view.py applies to
    the mesh geometry itself -- required so these points land in the exact
    same local frame the mesh (and its picker) actually uses."""
    points = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"]) * scale
            y = float(row["y"]) * scale
            z = float(row["z"]) * scale
            normal = None
            if row.get("normal_x"):
                normal = (
                    float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"]),
                )
            label = (row.get("label") or "").strip()
            points.append(PredefinedPoint(label=label, point_local=(x, y, z), normal_local=normal))
    return points
