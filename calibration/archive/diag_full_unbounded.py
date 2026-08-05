import numpy as np
from scipy.optimize import least_squares
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
final_res = residual(result.x).reshape(-1, 3)
errs = np.linalg.norm(final_res, axis=1)
rms = np.sqrt(np.mean(errs ** 2))
print("Full fit, NO bounds anywhere:")
print("  joint_offsets (deg):", np.degrees(result.x[0:7]))
print("  tool_xyz (mm):", result.x[7:10] * 1000.0)
print("  tool_rpy (deg):", np.degrees(result.x[10:13]))
print("  base_xyz (mm):", result.x[13:16] * 1000.0)
print("  base_rpy (deg):", np.degrees(result.x[16:19]))
print("  RMS position error (mm):", rms)
print("  max error (mm):", errs.max())
print("  median error (mm):", np.median(errs))
