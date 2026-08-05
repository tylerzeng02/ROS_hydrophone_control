import csv
import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

pose_ids = []
angles_list = []
pos_list = []
quat_list = []
moving_err_list = []
fixed_err_list = []
rel_err_list = []
moving_samples = []
fixed_samples = []

with open("C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pose_ids.append(int(row["pose_id"]))
        angles_list.append([float(row[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)])
        pos_list.append([
            float(row["moving_relative_fixed_tx_mm"]),
            float(row["moving_relative_fixed_ty_mm"]),
            float(row["moving_relative_fixed_tz_mm"]),
        ])
        q0 = float(row["moving_relative_fixed_q0"])
        qx = float(row["moving_relative_fixed_qx"])
        qy = float(row["moving_relative_fixed_qy"])
        qz = float(row["moving_relative_fixed_qz"])
        quat_list.append([qx, qy, qz, q0])
        moving_err_list.append(float(row["moving_camera_error"]))
        fixed_err_list.append(float(row["fixed_camera_error"]))
        rel_err_list.append(float(row["moving_relative_fixed_error"]))
        moving_samples.append(int(row["moving_accepted_samples"]))
        fixed_samples.append(int(row["fixed_accepted_samples"]))

angles = np.array(angles_list)
pos_mm = np.array(pos_list)
quat_xyzw = np.array(quat_list)
moving_err = np.array(moving_err_list)
fixed_err = np.array(fixed_err_list)
rel_err = np.array(rel_err_list)
moving_samples = np.array(moving_samples)
fixed_samples = np.array(fixed_samples)

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])
result = least_squares(
    ck.residual_function, x0, method="lm",
    args=(angles, pos_mm, quat_xyzw), verbose=0, max_nfev=50000
)
rms_pos, pos_errs = ck.rms_position_error_mm(result.x, angles, pos_mm)

print("Correlation of fit position error with NDI's own quality metrics:")
print("  vs moving_camera_error:", np.corrcoef(moving_err, pos_errs)[0, 1])
print("  vs fixed_camera_error: ", np.corrcoef(fixed_err, pos_errs)[0, 1])
print("  vs moving_relative_fixed_error:", np.corrcoef(rel_err, pos_errs)[0, 1])
print("  vs moving_accepted_samples:", np.corrcoef(moving_samples, pos_errs)[0, 1])
print("  vs fixed_accepted_samples:", np.corrcoef(fixed_samples, pos_errs)[0, 1])

print("\nNDI quality metric ranges:")
print("  moving_camera_error: min=%.3f max=%.3f mean=%.3f" % (moving_err.min(), moving_err.max(), moving_err.mean()))
print("  fixed_camera_error: min=%.3f max=%.3f mean=%.3f" % (fixed_err.min(), fixed_err.max(), fixed_err.mean()))
print("  moving_accepted_samples: min=%d max=%d" % (moving_samples.min(), moving_samples.max()))
print("  fixed_accepted_samples: min=%d max=%d" % (fixed_samples.min(), fixed_samples.max()))
