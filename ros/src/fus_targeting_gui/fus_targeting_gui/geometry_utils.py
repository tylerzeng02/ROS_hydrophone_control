"""Pure-numpy geometry helpers shared by mesh_view.py (no ROS dependency)
and registration.py. Kept dependency-free specifically so mesh_view.py's
standalone runnability (see its own docstring) isn't compromised by an
import chain that pulls in tf_transformations/geometry_msgs.
"""

import math

import numpy as np


def orthonormal_basis(axis):
    """Two unit vectors (u, v) forming a right-handed basis with `axis`
    (u, v, axis). The choice of u is arbitrary but deterministic -- only
    used to fix a consistent zero-reference for azimuth/roll, not anything
    physically meaningful on its own."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def tilt_direction(base_dir, tilt_deg, azimuth_deg):
    """Rotate `base_dir` (need not be unit-length) away from itself by
    `tilt_deg`, in the tangent-plane direction selected by `azimuth_deg`
    (degrees, 0-360, measured from the basis's own u axis toward v as
    azimuth increases). tilt_deg=0 returns normalize(base_dir) unchanged
    for any azimuth. Used both for the picked-point approach cone (tilt
    off the surface normal) and for mesh_view's matching live preview."""
    base = np.asarray(base_dir, dtype=float)
    base = base / np.linalg.norm(base)
    if abs(tilt_deg) < 1e-9:
        return base
    u, v = orthonormal_basis(base)
    tilt = math.radians(tilt_deg)
    az = math.radians(azimuth_deg)
    tangent = math.cos(az) * u + math.sin(az) * v
    result = math.cos(tilt) * base + math.sin(tilt) * tangent
    return result / np.linalg.norm(result)


def look_at_basis(direction, up_hint=(0.0, 0.0, 1.0), roll_deg=0.0):
    """Right-handed orthonormal basis (x, y, z) with z = normalize(direction).
    x/y are first resolved via `up_hint` (a "look-at" construction, not a
    claim about a physically correct zero-roll), then both rotated together
    about z by `roll_deg`. z is what should be commanded as the end
    effector's approach (+Z) axis; x/y indicate probe roll, for building a
    rotation matrix or for drawing a preview frame."""
    z = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(z)
    if norm < 1e-9:
        raise ValueError("Zero-length direction.")
    z = z / norm

    up = np.asarray(up_hint, dtype=float)
    if abs(np.dot(z, up)) > 0.999:
        # direction is (anti)parallel to the up hint -- pick a different
        # reference so cross() below doesn't degenerate.
        up = np.array([1.0, 0.0, 0.0])

    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    if abs(roll_deg) > 1e-9:
        roll = math.radians(roll_deg)
        x, y = (
            math.cos(roll) * x + math.sin(roll) * y,
            -math.sin(roll) * x + math.cos(roll) * y,
        )

    return x, y, z
