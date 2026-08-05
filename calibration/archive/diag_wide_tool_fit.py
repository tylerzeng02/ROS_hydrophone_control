import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

def residual(x):
    tool_xyz = x[0:3]
    tool_rpy = x[3:6]
    base_xyz = x[6:9]
    base_rpy = x[9:12]
    params = ck.CalibParams(np.zeros(7), tool_xyz, tool_rpy, base_xyz, base_rpy)
    res = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        T = ck.predict_relative_pose(angles[i], params)
        res[i] = pos_mm[i] - T[:3, 3] * 1000.0
    return res.ravel()

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])

result = least_squares(residual, x0, method="lm", verbose=0, max_nfev=20000)
final_res = residual(result.x).reshape(-1, 3)
rms = np.sqrt(np.mean(np.linalg.norm(final_res, axis=1) ** 2))
print("Tool + base fit, NO bounds, joint offsets forced to zero:")
print("  tool_xyz (mm):", result.x[0:3] * 1000.0)
print("  tool_rpy (deg):", np.degrees(result.x[3:6]))
print("  base_xyz (mm):", result.x[6:9] * 1000.0)
print("  base_rpy (deg):", np.degrees(result.x[9:12]))
print("  RMS position error (mm):", rms)
print("  max error (mm):", np.linalg.norm(final_res, axis=1).max())
