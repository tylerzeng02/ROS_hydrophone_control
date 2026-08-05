import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

def residual(x):
    joint_offsets = x[0:7]
    tool_xyz = x[7:10]
    tool_rpy = x[10:13]
    base_xyz = x[13:16]
    base_rpy = x[16:19]
    params = ck.CalibParams(joint_offsets, tool_xyz, tool_rpy, base_xyz, base_rpy)
    res = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        T = ck.predict_relative_pose(angles[i], params)
        res[i] = pos_mm[i] - T[:3, 3] * 1000.0
    return res.ravel()

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])
result = least_squares(residual, x0, method="lm", verbose=0, max_nfev=50000)

rms_pos, pos_errs = ck.rms_position_error_mm(result.x, angles, pos_mm)
rms_orient, orient_errs = ck.rms_orientation_error_deg(result.x, angles, quat_xyzw)

print("Position RMS (mm):", rms_pos)
print("Orientation RMS (deg):", rms_orient)
print("Orientation error stats: min=%.2f max=%.2f median=%.2f" %
      (orient_errs.min(), orient_errs.max(), np.median(orient_errs)))

# correlation between position error and orientation error per pose
corr = np.corrcoef(pos_errs, orient_errs)[0, 1]
print("Correlation between per-pose position error and orientation error:", corr)

print("\nWorst 5 poses by ORIENTATION error:")
worst = np.argsort(orient_errs)[-5:]
for i in worst:
    print(f"  pose {i}: pos_err={pos_errs[i]:.1f}mm orient_err={orient_errs[i]:.2f}deg")

print("\nBest 5 poses by ORIENTATION error:")
best = np.argsort(orient_errs)[:5]
for i in best:
    print(f"  pose {i}: pos_err={pos_errs[i]:.1f}mm orient_err={orient_errs[i]:.2f}deg")
