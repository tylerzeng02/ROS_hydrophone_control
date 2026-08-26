"""Refits move_between_points.cpp's R_MOVEIT_TO_NDI rotation (MoveIt frame
-> NDI frame) via Kabsch alignment on paired position readings, and reports
the fit quality against the currently deployed rotation for comparison.
"""

import csv
import sys

import numpy as np

# Currently deployed in move_between_points.cpp, refit 2026-08-13 from
# ndi_moveit_rotation_calibration_data.csv (13 valid pairs).
CURRENT_R_MOVEIT_TO_NDI = np.array([
    [0.0033, 0.8971, 0.4418],
    [0.6142, 0.3469, -0.7088],
    [-0.7891, 0.2737, -0.5499],
])


def load_pairs(path):
    """Loads paired MoveIt/NDI position readings from a capture CSV,
    dropping rows where getCurrentPose() failed (a real, distinguishable
    sentinel: exact zero position with identity orientation never occurs
    from a real reading).

    Args:
        path: Path to the capture CSV.

    Returns:
        Tuple of (moveit_pts, ndi_pts), each (N, 3) arrays in millimeters.
    """
    moveit_pts, ndi_pts = [], []
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mx, my, mz = float(row["moveit_pose_x_mm"]), float(row["moveit_pose_y_mm"]), float(row["moveit_pose_z_mm"])
            qw, qx, qy, qz = (float(row["moveit_pose_qw"]), float(row["moveit_pose_qx"]),
                               float(row["moveit_pose_qy"]), float(row["moveit_pose_qz"]))
            # Exact zero position + identity orientation: getCurrentPose()
            # failed for this row, not a real reading.
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
    """RMS error of rotated delta vectors. move_between_points.cpp applies
    this rotation to deltas between two live readings, not absolute
    positions, so that is what is scored here."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    predicted = (R @ Pc.T).T
    err = np.linalg.norm(predicted - Qc, axis=1)
    return np.sqrt(np.mean(err ** 2)), err


def main():
    """Fits R_MOVEIT_TO_NDI on the given capture CSV and prints it
    alongside an RMS comparison against the currently deployed rotation."""
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

    R_diff = CURRENT_R_MOVEIT_TO_NDI.T @ R_new
    angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"\nAngle between OLD and NEW rotation matrices: {angle_deg:.2f} deg")


if __name__ == "__main__":
    main()
