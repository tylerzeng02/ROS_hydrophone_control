import csv
import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
n = len(rows)

angles = np.array([[float(r[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)] for r in rows])
pos_mm = np.array([[float(r["moving_relative_fixed_tx_mm"]),
                     float(r["moving_relative_fixed_ty_mm"]),
                     float(r["moving_relative_fixed_tz_mm"])] for r in rows])
quat_xyzw = np.array([[float(r["moving_relative_fixed_qx"]), float(r["moving_relative_fixed_qy"]),
                        float(r["moving_relative_fixed_qz"]), float(r["moving_relative_fixed_q0"])] for r in rows])

moving_visible = np.array([int(r["moving_camera_visible_markers"]) for r in rows])
fixed_visible = np.array([int(r["fixed_camera_visible_markers"]) for r in rows])

print(f"n poses: {n}")
print(f"moving_camera_visible_markers: unique values = {sorted(set(moving_visible.tolist()))}")
print(f"fixed_camera_visible_markers:  unique values = {sorted(set(fixed_visible.tolist()))}\n")


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)
rms, errs = ck.rms_position_error_mm(result.x, angles, pos_mm)
print(f"Baseline fit RMS: {rms:.2f} mm\n")

corr_moving = np.corrcoef(moving_visible, errs)[0, 1]
corr_fixed = np.corrcoef(fixed_visible, errs)[0, 1]
print(f"Correlation of error with moving-marker visible count: {corr_moving:.3f}")
print(f"Correlation of error with fixed-marker visible count:  {corr_fixed:.3f}\n")

print("Mean error by fixed_camera_visible_markers count:")
for v in sorted(set(fixed_visible.tolist())):
    mask = fixed_visible == v
    print(f"  {v} markers visible: mean error = {errs[mask].mean():.2f} mm, n={mask.sum()}")

print("\nMean error by moving_camera_visible_markers count:")
for v in sorted(set(moving_visible.tolist())):
    mask = moving_visible == v
    print(f"  {v} markers visible: mean error = {errs[mask].mean():.2f} mm, n={mask.sum()}")
