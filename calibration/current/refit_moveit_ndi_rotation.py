"""Re-derives move_between_points.cpp's R_MOVEIT_TO_NDI rotation matrix from
fresh paired NDI-frame/MoveIt-frame poses collected by ndi_measure (after
the 2026-08-10 change that logs MoveGroupInterface's live getCurrentPose()
alongside each NDI capture).

Motivated by check_fixed_marker_drift.py finding ~6.3deg of drift in the
fixed marker's orientation since the batch2 calibration session that the
current hardcoded rotation was fit from.

Uses the Kabsch algorithm on zero-centered point sets (equivalent to
fitting on delta vectors, so translation of either frame's origin doesn't
matter -- consistent with move_between_points.cpp's own reasoning for why
only the rotation, not the full base-frame transform, is needed here).

Fit convention matches move_between_points.cpp's R_MOVEIT_TO_NDI exactly:
    v_ndi = R * v_moveit
so P (Kabsch source) = MoveIt-frame points, Q (Kabsch target) = NDI-frame
points -- the resulting R can be pasted directly into the C++ array, no
transposing needed.

Usage:
    uv run --with numpy python refit_moveit_ndi_rotation.py <input_csv>
"""

import csv
import sys

import numpy as np

# The rotation move_between_points.cpp currently hardcodes (from the
# batch2-only 12-param fit) -- printed alongside the new one for comparison.
CURRENT_R_MOVEIT_TO_NDI = np.array([
    [-0.0352, 0.8862, 0.4619],
    [0.6846, 0.3581, -0.6349],
    [-0.7281, 0.2939, -0.6193],
])


def load_pairs(path):
    moveit_pts, ndi_pts = [], []
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mx, my, mz = float(row["moveit_pose_x_mm"]), float(row["moveit_pose_y_mm"]), float(row["moveit_pose_z_mm"])
            qw, qx, qy, qz = (float(row["moveit_pose_qw"]), float(row["moveit_pose_qx"]),
                               float(row["moveit_pose_qy"]), float(row["moveit_pose_qz"]))
            # Detect the getCurrentPose()-failed sentinel: exact zero position
            # with identity orientation -- not a real reading.
            if mx == 0.0 and my == 0.0 and mz == 0.0 and qw == 1.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
                skipped += 1
                continue
            nx = float(row["moving_relative_fixed_tx_mm"])
            ny = float(row["moving_relative_fixed_ty_mm"])
            nz = float(row["moving_relative_fixed_tz_mm"])
            moveit_pts.append([mx, my, mz])
            ndi_pts.append([nx, ny, nz])
    print(f"Loaded {len(moveit_pts)} valid pairs ({skipped} skipped: failed getCurrentPose() sentinel)")
    return np.array(moveit_pts), np.array(ndi_pts)


def kabsch(P, Q):
    """Best rotation R minimizing sum|R @ P_i - Q_i|^2, both zero-centered first."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


def rms_delta_error_mm(R, P, Q):
    """RMS error of rotated delta vectors (pairwise, matches how
    move_between_points.cpp actually uses this rotation -- on deltas
    between two live readings, not absolute positions)."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    predicted = (R @ Pc.T).T
    err = np.linalg.norm(predicted - Qc, axis=1)
    return np.sqrt(np.mean(err ** 2)), err


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_csv>", file=sys.stderr)
        sys.exit(1)

    P_moveit, Q_ndi = load_pairs(sys.argv[1])
    if len(P_moveit) < 4:
        print(f"Only {len(P_moveit)} valid pairs -- too few for a robust rotation fit "
              f"(want at least ~5, ideally spread across varied poses).", file=sys.stderr)
        sys.exit(1)

    R_new = kabsch(P_moveit, Q_ndi)

    print("\n=== New R_MOVEIT_TO_NDI (v_ndi = R * v_moveit) ===")
    for row in R_new:
        print("    {" + ", ".join(f"{v:.4f}" for v in row) + "},")

    rms_new, err_new = rms_delta_error_mm(R_new, P_moveit, Q_ndi)
    rms_old, err_old = rms_delta_error_mm(CURRENT_R_MOVEIT_TO_NDI, P_moveit, Q_ndi)

    print(f"\nRMS delta-vector error on this data:")
    print(f"  OLD (batch2-derived) rotation: {rms_old:.3f} mm  (max {err_old.max():.3f} mm)")
    print(f"  NEW (this fit) rotation:       {rms_new:.3f} mm  (max {err_new.max():.3f} mm)")

    # Sanity: how far did the rotation itself move, as a single angle
    # (matches check_fixed_marker_drift.py's angle-between-rotations idea).
    R_diff = CURRENT_R_MOVEIT_TO_NDI.T @ R_new
    angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"\nAngle between OLD and NEW rotation matrices: {angle_deg:.2f} deg")


if __name__ == "__main__":
    main()
