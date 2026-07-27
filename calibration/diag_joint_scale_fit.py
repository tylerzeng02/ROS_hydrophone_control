import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

# Full model + a per-joint multiplicative scale factor applied to the
# measured angle before adding the offset: corrected = angle*scale + offset.
# Tests whether some joint's real ticks-per-radian differs from the assumed
# 4096/2pi (e.g. an unmodeled gear/belt reduction), which a constant offset
# alone cannot correct.

def unpack(x):
    joint_offsets = x[0:7]
    joint_scales = x[7:14]
    tool_xyz = x[14:17]
    tool_rpy = x[17:20]
    base_xyz = x[20:23]
    base_rpy = x[23:26]
    return joint_offsets, joint_scales, tool_xyz, tool_rpy, base_xyz, base_rpy


def predict(angles_row, joint_offsets, joint_scales, tool_xyz, tool_rpy, base_xyz, base_rpy):
    corrected = angles_row * joint_scales + joint_offsets
    T_fk = ck.forward_kinematics(corrected)
    T_tool = ck.build_tool_transform(tool_xyz, tool_rpy)
    T_base = ck.build_base_transform(base_xyz, base_rpy)
    return T_base @ T_fk @ T_tool


def residual(x):
    joint_offsets, joint_scales, tool_xyz, tool_rpy, base_xyz, base_rpy = unpack(x)
    n = len(angles)
    pos_res = np.zeros((n, 3))
    orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict(angles[i], joint_offsets, joint_scales, tool_xyz, tool_rpy, base_xyz, base_rpy)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3
    reg = np.concatenate([
        joint_offsets * (8.0 / np.radians(8.0)),
        (joint_scales - 1.0) * 50.0,   # regularize scale toward 1.0 (nominal)
        tool_xyz * (1.0 / 0.01),
        tool_rpy * (1.0 / np.radians(10.0)),
    ])
    return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(7), np.ones(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])

result = least_squares(residual, x0, method="lm", verbose=0, max_nfev=100000)
joint_offsets, joint_scales, tool_xyz, tool_rpy, base_xyz, base_rpy = unpack(result.x)

errs = []
for i in range(len(angles)):
    T_pred = predict(angles[i], joint_offsets, joint_scales, tool_xyz, tool_rpy, base_xyz, base_rpy)
    errs.append(np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0))
errs = np.array(errs)

print("Per-joint scale + offset fit:")
print("  joint_offsets (deg):", np.degrees(joint_offsets))
print("  joint_scales (should be ~1.0 if no gear/belt error):", joint_scales)
print("  tool_xyz (mm):", tool_xyz * 1000.0)
print("  tool_rpy (deg):", np.degrees(tool_rpy))
print("  RMS position error (mm):", np.sqrt(np.mean(errs**2)))
print("  max/median error (mm):", errs.max(), np.median(errs))
