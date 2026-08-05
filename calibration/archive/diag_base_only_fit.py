import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

def residual(base_params):
    base_xyz = base_params[:3]
    base_rpy = base_params[3:]
    params = ck.CalibParams(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz, base_rpy)
    res = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        T = ck.predict_relative_pose(angles[i], params)
        res[i] = pos_mm[i] - T[:3, 3] * 1000.0
    return res.ravel()

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([base_xyz0, base_rpy0])

result = least_squares(residual, x0, method="lm", verbose=0)
final_res = residual(result.x).reshape(-1, 3)
rms = np.sqrt(np.mean(np.linalg.norm(final_res, axis=1) ** 2))
print("Base-only fit (joint offsets & tool forced to zero):")
print("  base_xyz (mm):", result.x[:3] * 1000.0)
print("  base_rpy (deg):", np.degrees(result.x[3:]))
print("  RMS position error (mm):", rms)
print("  max error (mm):", np.linalg.norm(final_res, axis=1).max())
