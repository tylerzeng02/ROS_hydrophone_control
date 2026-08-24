"""Rigid-transform (rotation + translation) fit between two matched sets of
3D points -- the standard Kabsch/SVD algorithm. Used by the GUI's
Registration panel to compute a real mesh-local -> base_frame transform
from point pairs (a mesh point you clicked, paired with the robot's live
end-effector position when its probe touches the same physical landmark on
the real skull) -- replacing FixedPoseRegistration's hand-typed guess with
an actual measurement.

Pure numpy, no ROS/pyvista dependency -- independently testable.
"""

import numpy as np


def fit_rigid_transform(source_points, target_points):
    """Finds the rotation matrix R and translation t minimizing
    sum(||R @ source_i + t - target_i||^2) over all point pairs (Kabsch
    algorithm via SVD, with the standard reflection-correction step so R is
    always a proper rotation, never a reflection). Returns (R, t, rmse) --
    R is a 3x3 orthonormal rotation matrix (det = +1), t is a length-3
    translation, and rmse is the root-mean-square residual distance (same
    units as the input points) after applying the fit -- a measure of how
    well the two point sets actually agree, not just a formality; a large
    rmse means the point pairs are inconsistent (bad touches, wrong mesh
    point selected, etc.), not that the fit itself failed.

    Needs at least 3 point pairs to fully constrain a 3D rotation, and
    those 3+ points must not be collinear (raises ValueError below 3
    pairs; collinear inputs won't raise but will produce a degenerate,
    untrustworthy fit -- check the returned rmse).
    """
    src = np.asarray(source_points, dtype=float)
    tgt = np.asarray(target_points, dtype=float)
    if src.shape != tgt.shape:
        raise ValueError(
            f"source_points and target_points must have the same shape "
            f"(got {src.shape} vs {tgt.shape})."
        )
    if len(src) < 3:
        raise ValueError("Need at least 3 point pairs to fit a rigid transform.")

    src_centroid = src.mean(axis=0)
    tgt_centroid = tgt.mean(axis=0)
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T

    t = tgt_centroid - R @ src_centroid

    transformed = (R @ src.T).T + t
    residuals = np.linalg.norm(transformed - tgt, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return R, t, rmse
