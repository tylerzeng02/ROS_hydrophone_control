import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)

perp_vectors = []
for axis in ck.JOINT_AXES:
    axis = axis / np.linalg.norm(axis)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    perp_vectors.append((u, v))


def tilted_axes(tilt_params):
    axes = np.zeros((ck.N_JOINTS, 3))
    for i in range(ck.N_JOINTS):
        axis = ck.JOINT_AXES[i]
        u, v = perp_vectors[i]
        perturbed = axis + tilt_params[i, 0] * u + tilt_params[i, 1] * v
        axes[i] = perturbed / np.linalg.norm(perturbed)
    return axes


def forward_kinematics_combined(joint_angles_rad, axes, origin_deltas_m):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        joint_rotation = Rotation.from_rotvec(axes[i] * joint_angles_rad[i]).as_matrix()
        origin = ck.JOINT_ORIGINS_M[i] + origin_deltas_m[i]
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


N_TILT = ck.N_JOINTS * 2       # 14
N_ORIGIN = ck.N_JOINTS * 3     # 21
TILT_REG_WEIGHT = 15.0
TILT_SCALE_RAD = np.radians(5.0)
ORIGIN_REG_WEIGHT = 20.0       # mm-equivalent penalty per meter of origin delta
ORIGIN_SCALE_M = 0.005         # regularize toward sub-5mm origin corrections


def residual_combined(x):
    base_params = x[:19]
    tilt_flat = x[19:19 + N_TILT]
    origin_flat = x[19 + N_TILT:]

    tilt_params = tilt_flat.reshape(ck.N_JOINTS, 2)
    origin_deltas = origin_flat.reshape(ck.N_JOINTS, 3)
    axes = tilted_axes(tilt_params)

    params = ck.unpack_params(base_params)
    n_local = len(angles)
    pos_res = np.zeros((n_local, 3))
    orient_res = np.zeros((n_local, 3))

    for i in range(n_local):
        corrected_angles = angles[i] + params.joint_offsets
        T_fk = forward_kinematics_combined(corrected_angles, axes, origin_deltas)
        T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
        T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
        T_pred = T_base @ T_fk @ T_tool

        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3

    reg = np.concatenate([
        params.joint_offsets * (8.0 / np.radians(8.0)),
        params.tool_xyz * (1.0 / 0.01),
        params.tool_rpy * (1.0 / np.radians(10.0)),
        tilt_flat * (TILT_REG_WEIGHT / TILT_SCALE_RAD),
        origin_flat * (ORIGIN_REG_WEIGHT / ORIGIN_SCALE_M),
    ])

    return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0_base = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
x0 = np.concatenate([x0_base, np.zeros(N_TILT), np.zeros(N_ORIGIN)])

result = least_squares(residual_combined, x0, method="lm", max_nfev=300000)

n_pos = 3 * n
pos_res_final = result.fun[:n_pos].reshape(n, 3)
errs = np.linalg.norm(pos_res_final, axis=1)
rms = np.sqrt(np.mean(errs ** 2))

print(f"Combined fit (19 base + 14 axis-tilt + 21 origin-delta = 54 params):")
print(f"  Position RMS: {rms:.2f} mm (median {np.median(errs):.2f}, max {errs.max():.2f})")
print()
print("For reference:")
print("  Baseline (19 params):              15.43 mm")
print("  + axis tilt only (33 params):       12.29 mm")
print(f"  + axis tilt + origin (54 params):   {rms:.2f} mm")
