import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

original_origins = ck.JOINT_ORIGINS_M.copy()

def forward_kinematics_with_origin_deltas(joint_angles_rad, origin_deltas):
    T = np.eye(4)
    origins = original_origins + origin_deltas
    for i in range(ck.N_JOINTS):
        joint_rotation = Rotation.from_rotvec(ck.JOINT_AXES[i] * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, origins[i])
        T = T @ T_joint
    return T

def predict_with_origin_deltas(measured_angles_rad, params, origin_deltas):
    corrected_angles = measured_angles_rad + params.joint_offsets
    T_fk = forward_kinematics_with_origin_deltas(corrected_angles, origin_deltas)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool

def residual(x):
    joint_offsets = x[0:7]
    tool_xyz = x[7:10]
    tool_rpy = x[10:13]
    base_xyz = x[13:16]
    base_rpy = x[16:19]
    origin_deltas = x[19:19+21].reshape(7, 3)

    params = ck.CalibParams(joint_offsets, tool_xyz, tool_rpy, base_xyz, base_rpy)

    res_pos = np.zeros((len(angles), 3))
    res_orient = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        T = predict_with_origin_deltas(angles[i], params, origin_deltas)
        res_pos[i] = pos_mm[i] - T[:3, 3] * 1000.0
        R_pred = T[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        res_orient[i] = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec() * 100.0 * 0.3

    # light regularization pulling origin deltas toward zero (they should be
    # small manufacturing-tolerance-scale corrections, not free translations)
    reg = origin_deltas.ravel() * (1.0 / 0.005)  # ~5mm scale
    return np.concatenate([res_pos.ravel(), res_orient.ravel(), reg])

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0, np.zeros(21)])

result = least_squares(residual, x0, method="lm", verbose=0, max_nfev=100000)

joint_offsets = result.x[0:7]
tool_xyz = result.x[7:10]
tool_rpy = result.x[10:13]
base_xyz = result.x[13:16]
base_rpy = result.x[16:19]
origin_deltas = result.x[19:40].reshape(7, 3)
params = ck.CalibParams(joint_offsets, tool_xyz, tool_rpy, base_xyz, base_rpy)

pos_errs = np.zeros(len(angles))
orient_errs = np.zeros(len(angles))
for i in range(len(angles)):
    T = predict_with_origin_deltas(angles[i], params, origin_deltas)
    pos_errs[i] = np.linalg.norm(pos_mm[i] - T[:3, 3] * 1000.0)
    R_pred = T[:3, :3]
    R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
    orient_errs[i] = np.degrees(Rotation.from_matrix(R_pred.T @ R_meas).magnitude())

print("Fit with per-joint origin deltas (regularized ~5mm scale) + orientation:")
print("  joint_offsets (deg):", np.degrees(joint_offsets))
print("  tool_xyz (mm):", tool_xyz * 1000.0)
print("  origin_deltas (mm) per joint:")
for i in range(7):
    print(f"    joint {i}: {origin_deltas[i]*1000.0}")
print("  Position RMS (mm):", np.sqrt(np.mean(pos_errs**2)))
print("  Orientation RMS (deg):", np.sqrt(np.mean(orient_errs**2)))
print("  Position median/max (mm):", np.median(pos_errs), pos_errs.max())
