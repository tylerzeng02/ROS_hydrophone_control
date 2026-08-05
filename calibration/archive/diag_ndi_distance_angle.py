import csv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
n = len(rows)

angles = np.array([[float(r[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)] for r in rows])
pos_mm = np.array([[float(r["moving_relative_fixed_tx_mm"]),
                     float(r["moving_relative_fixed_ty_mm"]),
                     float(r["moving_relative_fixed_tz_mm"])] for r in rows])
quat_xyzw = np.array([[float(r["moving_relative_fixed_qx"]),
                        float(r["moving_relative_fixed_qy"]),
                        float(r["moving_relative_fixed_qz"]),
                        float(r["moving_relative_fixed_q0"])] for r in rows])

# Raw camera-frame (tracker-relative) data -- distance from tracker to each
# marker, and the moving marker's orientation relative to the tracker.
moving_cam_pos = np.array([[float(r["moving_camera_tx_mm"]),
                             float(r["moving_camera_ty_mm"]),
                             float(r["moving_camera_tz_mm"])] for r in rows])
fixed_cam_pos = np.array([[float(r["fixed_camera_tx_mm"]),
                            float(r["fixed_camera_ty_mm"]),
                            float(r["fixed_camera_tz_mm"])] for r in rows])
moving_cam_quat = np.array([[float(r["moving_camera_qx"]), float(r["moving_camera_qy"]),
                              float(r["moving_camera_qz"]), float(r["moving_camera_q0"])] for r in rows])

moving_dist = np.linalg.norm(moving_cam_pos, axis=1)
fixed_dist = np.linalg.norm(fixed_cam_pos, axis=1)

# Viewing-angle proxy: angle between the moving marker's local Z axis
# (rotated into camera frame) and the direction from the marker back to the
# tracker (assumed near the camera-frame origin) -- smaller angle = marker
# facing more directly at the tracker.
facing_angle_deg = np.zeros(n)
for i in range(n):
    R = Rotation.from_quat(moving_cam_quat[i]).as_matrix()
    marker_z_in_camera = R[:, 2]
    to_tracker = -moving_cam_pos[i] / np.linalg.norm(moving_cam_pos[i])
    cos_angle = np.clip(np.dot(marker_z_in_camera, to_tracker), -1.0, 1.0)
    facing_angle_deg[i] = np.degrees(np.arccos(cos_angle))

# Fit the baseline 19-param model to get residual position errors
def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)
rms, errs = ck.rms_position_error_mm(result.x, angles, pos_mm)
print(f"Baseline fit RMS: {rms:.2f} mm\n")

print(f"Moving marker distance from tracker: {moving_dist.min():.0f} - {moving_dist.max():.0f} mm")
print(f"Fixed marker distance from tracker:   {fixed_dist.min():.0f} - {fixed_dist.max():.0f} mm")
print(f"Moving marker facing angle: {facing_angle_deg.min():.1f} - {facing_angle_deg.max():.1f} deg\n")

corr_moving_dist = np.corrcoef(moving_dist, errs)[0, 1]
corr_fixed_dist = np.corrcoef(fixed_dist, errs)[0, 1]
corr_facing = np.corrcoef(facing_angle_deg, errs)[0, 1]

print(f"Correlation of error with moving-marker distance from tracker: {corr_moving_dist:.3f}")
print(f"Correlation of error with fixed-marker distance from tracker:  {corr_fixed_dist:.3f}")
print(f"Correlation of error with moving-marker facing angle:          {corr_facing:.3f}\n")

order = np.argsort(moving_dist)
q = n // 4
print("Moving-marker-distance quartile -> mean error (mm):")
for i in range(4):
    s, e = i * q, (i + 1) * q if i < 3 else n
    idx = order[s:e]
    print(f"  Q{i+1} (dist {moving_dist[idx].min():.0f}-{moving_dist[idx].max():.0f}mm): "
          f"mean error = {errs[idx].mean():.2f} mm, n={len(idx)}")

order2 = np.argsort(facing_angle_deg)
print("\nFacing-angle quartile -> mean error (mm):")
for i in range(4):
    s, e = i * q, (i + 1) * q if i < 3 else n
    idx = order2[s:e]
    print(f"  Q{i+1} (angle {facing_angle_deg[idx].min():.1f}-{facing_angle_deg[idx].max():.1f}deg): "
          f"mean error = {errs[idx].mean():.2f} mm, n={len(idx)}")
