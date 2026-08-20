"""Validates the real-world accuracy of the calibration corrections actually
DEPLOYED to src/robot_calibration.cpp and
references/cyton_gamma_1500_trac_ik.urdf (as of 2026-08-06) -- distinct from
every other script in this directory, which fits/tests a richer 60-param
research model (offset+scale+tilt+origin+joint-coupling+gravity-deflection+
shoulder_pitch-Fourier). Only the static subset (offset/scale/tilt/origin)
was ever baked into the URDF -- the pose-dependent terms (coupling, gravity,
Fourier) can't live in a static URDF and were never deployed. Testing
against the full 60-param model (as gen_validation_predictions.py does)
answers "how good is our best research model"; this script answers "how
good is what's actually in the URDF right now."

Joint-level parameters are HARDCODED here to their exact deployed values
(copied directly from src/robot_calibration.cpp's jointCalibrations array
and references/cyton_gamma_1500_trac_ik.urdf's <axis>/<origin>), not fit.
The only thing fit is the base-frame/tool-frame transform (12 params)
needed to translate a predicted end-effector pose into the NDI tracker's
measurement frame (moving-marker-relative-to-fixed-marker) -- nothing in
the deployed system relates base_link to that frame (deliberately: that
transform is anchored to the calibration rig, not real deployment). This
mirrors exactly how calibrate_kinematics.py treats
base_xyz/base_rpy (unregularized, unknown rig geometry) and tool_xyz/
tool_rpy (weakly regularized around nominal, since the marker mount is
close to its designed position) -- just isolated here since joint offsets
are not being fit.

IMPORTANT: angles are recomputed from each CSV row's actual_tick_* columns
via this script's own ticks_to_deployed_radians(), NOT read from the CSV's
actual_rad_* columns. Those columns are written by whatever version of
ticksToRadians() was compiled into ndi_capture_and_validate.cpp at
capture time -- the 374-pose deployed_model_training_dataset_374pose.csv (this
dataset) was captured BEFORE the 2026-08-06 deployment (pre-correction angles), while
the new batch2 data was captured with the current, already-corrected
binary (post-correction angles). Recomputing from raw ticks with a single,
explicit deployed formula sidesteps that inconsistency entirely.

Usage:
    python deployed_model_predictions.py --fit-csv a.csv,b.csv --points ticks.csv --out predictions.csv

Requires: numpy, scipy
"""

import argparse
import csv
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

# Deployed joint-level calibration (src/robot_calibration.cpp jointCalibrations,
# as of 2026-08-06). direction is +1 for every joint on this arm.
# radians = direction * scale * (tick - zeroTick) / TICKS_PER_RADIAN
# -- exactly matches ticksToRadians() in src/robot_calibration.cpp.
DEPLOYED_ZERO_TICKS = [2048, 2047, 2060, 2102, 2078, 2042, 2048]
DEPLOYED_DIRECTION = [1, 1, 1, 1, 1, 1, 1]
DEPLOYED_SCALE = [0.988203, 1.001931, 0.964711, 1.014467, 1.0, 1.006796, 1.002933]

# Deployed geometry (references/cyton_gamma_1500_trac_ik.urdf, as of
# 2026-08-06). Axis tilt applied to all 7 joints except elbow_yaw (index 4,
# left at its nominal axis -- poorly identified fit, moot since the joint is
# now permanently locked). Origin correction applied only to shoulder_yaw
# (x,z), elbow_pitch (x,y,z), wrist_pitch (x,y,z); every other joint's
# origin is unchanged from nominal.
DEPLOYED_AXES_RAW = np.array([
    [0.013792, 0.014877, 0.999794],
    [0.998997, -0.043573, -0.010357],
    [-0.027678, -0.999437, 0.018973],
    [0.999677, -0.024005, 0.008304],
    [0.0, -1.0, 0.0],
    [0.999044, 0.042957, 0.008105],
    [-0.006451, -0.018572, 0.999807],
])
DEPLOYED_AXES = DEPLOYED_AXES_RAW / np.linalg.norm(DEPLOYED_AXES_RAW, axis=1, keepdims=True)

DEPLOYED_ORIGINS_M = np.array([
    [0.0, 0.0, 0.05315],
    [0.0205, 0.0, 0.12435],
    [-0.02478414, -0.0205, 0.1308452],
    [0.01656849, 0.02722018, 0.11356304],
    [-0.0171, -0.018, 0.09746],
    [0.02765348, 0.01273746, 0.07244612],
    [-0.026255, 0.0, 0.051425],
])

# virtual_endeffector_joint -- unchanged from nominal in the deployed URDF
# (the offline tool-frame correction was fit but never baked into the URDF
# geometry itself; it's only ever used, here and elsewhere, as a free
# nuisance parameter to interpret the NDI marker-mount offset).
DEPLOYED_TOOL_ORIGIN_M = ck.TOOL_ORIGIN_NOMINAL_M.copy()


def ticks_to_deployed_radians(ticks):
    ticks = np.array(ticks, dtype=float)
    zero = np.array(DEPLOYED_ZERO_TICKS, dtype=float)
    direction = np.array(DEPLOYED_DIRECTION, dtype=float)
    scale = np.array(DEPLOYED_SCALE, dtype=float)
    return direction * scale * (ticks - zero) / ck.TICKS_PER_RADIAN


def forward_kinematics_deployed(joint_angles_rad):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        R = Rotation.from_rotvec(DEPLOYED_AXES[i] * joint_angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, DEPLOYED_ORIGINS_M[i])
    return T


# Frame-fit parameters (12): tool_xyz/rpy (3+3, weakly regularized around
# nominal) + base_xyz/rpy (3+3, unregularized -- no small-value prior, same
# reasoning as calibrate_kinematics.py).

def unpack(x):
    return dict(tool_xyz=x[0:3], tool_rpy=x[3:6], base_xyz=x[6:9], base_rpy=x[9:12])


def pack(tool_xyz, tool_rpy, base_xyz, base_rpy):
    return np.concatenate([tool_xyz, tool_rpy, base_xyz, base_rpy])


def predict_relative_pose_deployed(angles_rad, p):
    T_fk = forward_kinematics_deployed(angles_rad)
    T_tool = ck.build_tool_transform(p["tool_xyz"], p["tool_rpy"])
    T_base = ck.build_base_transform(p["base_xyz"], p["base_rpy"])
    return T_base @ T_fk @ T_tool


def residual(x, angles, pos_mm, quat_xyzw):
    p = unpack(x)
    n = len(angles)
    pos_res = np.zeros((n, 3))
    orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict_relative_pose_deployed(angles[i], p)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_err = T_pred[:3, :3].T @ Rotation.from_quat(quat_xyzw[i]).as_matrix()
        orient_res[i] = (
            Rotation.from_matrix(R_err).as_rotvec()
            * ck.ORIENTATION_SCALE_MM * ck.ORIENTATION_WEIGHT
        )
    # Regularize tool_xyz toward zero only (a mild small-value prior on the
    # marker-mount translation). tool_rpy/base_xyz/base_rpy are left
    # unregularized, matching every "current best" script in this project --
    # the core calibrate_kinematics.py's tight ±10mm/±10deg/±180deg
    # bounds+regularization on these caused bound-pinning (confirmed
    # directly: even the known-good 374-pose dataset blows up to ~70mm RMS
    # under those tight bounds), so this script follows the wider, proven
    # convention instead.
    reg = p["tool_xyz"] * (ck.TOOL_REG_WEIGHT / TOOL_XYZ_BOUND_M)
    return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])


TOOL_XYZ_BOUND_M = 0.10  # matches gen_validation_predictions.py's widened tool bound


def bounds():
    lo = np.concatenate([
        -TOOL_XYZ_BOUND_M * np.ones(3), [-np.inf] * 3,
        [-np.inf] * 3, [-np.inf] * 3,
    ])
    up = -lo
    return lo, up


def x0(angles, pos_mm, quat_xyzw):
    # Closed-form base guess (assuming nominal tool) so the optimizer starts
    # near a sane answer rather than at identity.
    T_fk = forward_kinematics_deployed(angles[0])
    T_tool_nominal = ck.build_tool_transform(np.zeros(3), np.zeros(3))
    T_partial = T_fk @ T_tool_nominal
    T_measured = np.eye(4)
    T_measured[:3, :3] = Rotation.from_quat(quat_xyzw[0]).as_matrix()
    T_measured[:3, 3] = pos_mm[0] / 1000.0
    T_base_guess = T_measured @ np.linalg.inv(T_partial)
    base_xyz_guess = T_base_guess[:3, 3]
    base_rpy_guess = Rotation.from_matrix(T_base_guess[:3, :3]).as_euler("xyz")
    return pack(np.zeros(3), np.zeros(3), base_xyz_guess, base_rpy_guess)


def fit(angles, pos_mm, quat_xyzw):
    lo, up = bounds()
    guess = x0(angles, pos_mm, quat_xyzw)
    lo = np.minimum(lo, guess - 1e-6)
    up = np.maximum(up, guess + 1e-6)
    return least_squares(
        residual, guess, args=(angles, pos_mm, quat_xyzw),
        method="trf", x_scale="jac", bounds=(lo, up), max_nfev=5000,
    )


def load_poses_deployed_angles(csv_path):
    """Like calibrate_kinematics.load_poses_from_csv, but recomputes angles
    from actual_tick_* via ticks_to_deployed_radians() instead of trusting
    the CSV's actual_rad_* columns (see module docstring for why)."""
    angles, pos_mm, quat_xyzw = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticks = [float(row[f"actual_tick_{i}"]) for i in range(ck.N_JOINTS)]
            angles.append(ticks_to_deployed_radians(ticks))
            pos_mm.append(np.array([
                float(row["moving_relative_fixed_tx_mm"]),
                float(row["moving_relative_fixed_ty_mm"]),
                float(row["moving_relative_fixed_tz_mm"]),
            ]))
            q0 = float(row["moving_relative_fixed_q0"])
            qx = float(row["moving_relative_fixed_qx"])
            qy = float(row["moving_relative_fixed_qy"])
            qz = float(row["moving_relative_fixed_qz"])
            quat_xyzw.append(np.array([qx, qy, qz, q0]))
    return np.array(angles), np.array(pos_mm), np.array(quat_xyzw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-csv", required=True, help="Comma-separated CSVs to fit base/tool frame on.")
    parser.add_argument("--points", required=True, help="Input CSV of tick_0..tick_6 test points.")
    parser.add_argument("--out", required=True, help="Output CSV with predicted_x/y/z_mm added.")
    args = parser.parse_args()

    fit_paths = [p.strip() for p in args.fit_csv.split(",") if p.strip()]
    angles_list, pos_list, quat_list = [], [], []
    for path in fit_paths:
        a, p, q = load_poses_deployed_angles(path)
        angles_list.append(a)
        pos_list.append(p)
        quat_list.append(q)
        print(f"Loaded {len(a)} poses from {path}")
    angles_all = np.concatenate(angles_list)
    pos_mm_all = np.concatenate(pos_list)
    quat_xyzw_all = np.concatenate(quat_list)
    print(f"Fitting base/tool frame (12 params) on {len(angles_all)} poses, "
          f"joint-level parameters fixed at their deployed values...")

    result = fit(angles_all, pos_mm_all, quat_xyzw_all)
    p = unpack(result.x)
    print("\n=== Fitted frame parameters ===")
    print(f"  tool_xyz (mm)  = {p['tool_xyz'] * 1000.0}")
    print(f"  tool_rpy (deg) = {np.degrees(p['tool_rpy'])}")
    print(f"  base_xyz (mm)  = {p['base_xyz'] * 1000.0}")
    print(f"  base_rpy (deg) = {np.degrees(p['base_rpy'])}")

    e = np.zeros(len(angles_all))
    for i in range(len(angles_all)):
        T_pred = predict_relative_pose_deployed(angles_all[i], p)
        e[i] = np.linalg.norm(pos_mm_all[i] - T_pred[:3, 3] * 1000.0)
    print(f"\nDeployed-model in-sample RMS on fit dataset: {np.sqrt(np.mean(e ** 2)):.2f}mm "
          f"(sanity check only -- not a held-out number)")

    ticks_rows = []
    with open(args.points, newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        try:
            [int(v) for v in first[:7]]
            ticks_rows.append(first)
        except ValueError:
            pass  # header row, skip
        for row in reader:
            if row:
                ticks_rows.append(row)

    if not ticks_rows:
        print(f"No test points found in {args.points}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tick_0", "tick_1", "tick_2", "tick_3", "tick_4", "tick_5", "tick_6",
            "predicted_x_mm", "predicted_y_mm", "predicted_z_mm",
        ])
        for row in ticks_rows:
            ticks = [int(v) for v in row[:7]]
            angles = ticks_to_deployed_radians(ticks)
            T_pred = predict_relative_pose_deployed(angles, p)
            pred_mm = T_pred[:3, 3] * 1000.0
            writer.writerow(ticks + [f"{pred_mm[0]:.4f}", f"{pred_mm[1]:.4f}", f"{pred_mm[2]:.4f}"])
            print(f"  ticks={ticks} -> predicted (x,y,z)mm = "
                  f"({pred_mm[0]:.2f}, {pred_mm[1]:.2f}, {pred_mm[2]:.2f})")

    print(f"\nWrote {len(ticks_rows)} predicted test points (deployed model) to {args.out}")
    print(f"Run on the arm with: ./ndi_capture_and_validate --validate {args.out}")


if __name__ == "__main__":
    main()
