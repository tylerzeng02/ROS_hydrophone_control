import csv
import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

QUICK_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
FULL_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"

# Fit on the 15-pose quick test (same as before)
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(QUICK_CSV)


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)
params = ck.unpack_params(result.x)

train_rms, _ = ck.rms_position_error_mm(result.x, angles, pos_mm)
print(f"Fitted on 15-pose quick test. Training RMS: {train_rms:.2f} mm\n")

# Pull pose_id 199 directly out of the original 200-pose CSV
target_pose_angles = None
target_pos_mm = None
target_quat_xyzw = None

with open(FULL_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["pose_id"]) == 199:
            target_pose_angles = np.array(
                [float(row[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)]
            )
            target_pos_mm = np.array([
                float(row["moving_relative_fixed_tx_mm"]),
                float(row["moving_relative_fixed_ty_mm"]),
                float(row["moving_relative_fixed_tz_mm"]),
            ])
            q0 = float(row["moving_relative_fixed_q0"])
            qx = float(row["moving_relative_fixed_qx"])
            qy = float(row["moving_relative_fixed_qy"])
            qz = float(row["moving_relative_fixed_qz"])
            target_quat_xyzw = np.array([qx, qy, qz, q0])
            break

if target_pose_angles is None:
    raise RuntimeError("pose_id 199 not found in five_pose_ndi_capture.csv")

print("Pose 199 actual joint angles (deg):", np.degrees(target_pose_angles))

# Check whether pose 199 is interpolation or extrapolation relative to the
# 15-pose training set's per-joint coverage
print("\nPer-joint: is pose 199 inside the 15-pose training range?")
for j in range(ck.N_JOINTS):
    lo = np.degrees(angles[:, j].min())
    hi = np.degrees(angles[:, j].max())
    val = np.degrees(target_pose_angles[j])
    inside = lo <= val <= hi
    print(f"  {ck.JOINT_NAMES[j]:24s} train=[{lo:7.1f}, {hi:7.1f}]  pose199={val:7.1f}  {'INSIDE' if inside else 'EXTRAPOLATED'}")

T_pred = ck.predict_relative_pose(target_pose_angles, params)
pred_pos_mm = T_pred[:3, 3] * 1000.0
pos_error = np.linalg.norm(target_pos_mm - pred_pos_mm)

from scipy.spatial.transform import Rotation
R_pred = T_pred[:3, :3]
R_meas = Rotation.from_quat(target_quat_xyzw).as_matrix()
orient_error_deg = np.degrees(Rotation.from_matrix(R_pred.T @ R_meas).magnitude())

print(f"\nPredicted position (mm): {pred_pos_mm}")
print(f"Measured position (mm):  {target_pos_mm}")
print(f"Position error: {pos_error:.2f} mm")
print(f"Orientation error: {orient_error_deg:.2f} deg")
