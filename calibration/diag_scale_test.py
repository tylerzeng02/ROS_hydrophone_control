import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

original_origins = ck.JOINT_ORIGINS_M.copy()
original_tool = ck.TOOL_ORIGIN_NOMINAL_M.copy()

def residual(x):
    scale = x[0]
    joint_offsets = x[1:8]
    tool_xyz = x[8:11]
    tool_rpy = x[11:14]
    base_xyz = x[14:17]
    base_rpy = x[17:20]

    ck.JOINT_ORIGINS_M = original_origins * scale
    ck.TOOL_ORIGIN_NOMINAL_M = original_tool * scale

    params = ck.CalibParams(joint_offsets, tool_xyz, tool_rpy, base_xyz, base_rpy)
    res_pos = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        T = ck.predict_relative_pose(angles[i], params)
        res_pos[i] = pos_mm[i] - T[:3, 3] * 1000.0
    return res_pos.ravel()

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([[1.0], np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])

result = least_squares(residual, x0, method="lm", verbose=0, max_nfev=50000)

ck.JOINT_ORIGINS_M = original_origins.copy()
ck.TOOL_ORIGIN_NOMINAL_M = original_tool.copy()

final_res = residual(result.x).reshape(-1, 3)
errs = np.linalg.norm(final_res, axis=1)
rms = np.sqrt(np.mean(errs ** 2))

print("Fit with a single global link-length scale factor added:")
print("  scale factor:", result.x[0])
print("  joint_offsets (deg):", np.degrees(result.x[1:8]))
print("  RMS position error (mm):", rms)
print("  median/max error (mm):", np.median(errs), errs.max())
