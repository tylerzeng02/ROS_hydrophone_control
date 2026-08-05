import numpy as np
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

base_xyz, base_rpy = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
params = ck.CalibParams(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz, base_rpy)

pred_mm = np.zeros_like(pos_mm)
for i in range(len(angles)):
    T = ck.predict_relative_pose(angles[i], params)
    pred_mm[i] = T[:3, 3] * 1000.0

# displacement relative to pose 0
meas_disp = pos_mm - pos_mm[0]
pred_disp = pred_mm - pred_mm[0]

meas_mag = np.linalg.norm(meas_disp, axis=1)
pred_mag = np.linalg.norm(pred_disp, axis=1)

corr = np.corrcoef(meas_mag, pred_mag)[0, 1]
print("Correlation between measured and predicted displacement magnitude (rel. to pose0):", corr)

# per-axis correlation too
for axis, name in enumerate(["x", "y", "z"]):
    c = np.corrcoef(meas_disp[:, axis], pred_disp[:, axis])[0, 1]
    print(f"Per-axis correlation ({name}):", c)

# print a handful of rows for manual inspection
print("\nSample rows (meas_disp vs pred_disp, mm):")
for i in [1, 5, 10, 50, 100, 150]:
    print(f"pose {i}: meas={meas_disp[i]}, pred={pred_disp[i]}, |meas|={meas_mag[i]:.1f}, |pred|={pred_mag[i]:.1f}")

# also check: how much does EACH joint individually move across dataset (range in degrees)
print("\nPer-joint angle range across dataset (deg):")
for j in range(7):
    vals = np.degrees(angles[:, j])
    print(f"  joint {j}: min={vals.min():.1f} max={vals.max():.1f} range={vals.max()-vals.min():.1f}")
