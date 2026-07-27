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


def fk_combined(joint_angles_rad, axes, origin_deltas_m):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        joint_rotation = Rotation.from_rotvec(axes[i] * joint_angles_rad[i]).as_matrix()
        origin = ck.JOINT_ORIGINS_M[i] + origin_deltas_m[i]
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


N_TILT = ck.N_JOINTS * 2
N_ORIGIN = ck.N_JOINTS * 3


def unpack_combined(x):
    base_params = x[:19]
    tilt_flat = x[19:19 + N_TILT]
    origin_flat = x[19 + N_TILT:]
    return base_params, tilt_flat.reshape(ck.N_JOINTS, 2), origin_flat.reshape(ck.N_JOINTS, 3)


def predict_combined(angles_row, base_params, tilt_params, origin_deltas):
    params = ck.unpack_params(base_params)
    axes = tilted_axes(tilt_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, axes, origin_deltas)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def residual_no_reg(x):
    base_params, tilt_params, origin_deltas = unpack_combined(x)
    pos_res = np.zeros((n, 3))
    orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict_combined(angles[i], base_params, tilt_params, origin_deltas)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3
    return np.concatenate([pos_res.ravel(), orient_res.ravel()])


def rms_of(x):
    base_params, tilt_params, origin_deltas = unpack_combined(x)
    errs = np.zeros(n)
    for i in range(n):
        T_pred = predict_combined(angles[i], base_params, tilt_params, origin_deltas)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0_base = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
x0 = np.concatenate([x0_base, np.zeros(N_TILT), np.zeros(N_ORIGIN)])

result_noreg = least_squares(residual_no_reg, x0, method="lm", max_nfev=300000)
rms_noreg, _ = rms_of(result_noreg.x)

print(f"54-param model WITHOUT any regularization: RMS = {rms_noreg:.2f} mm")
print(f"(for reference, WITH regularization it was 10.40 mm)\n")

# Identifiability check: SVD of the Jacobian at the unregularized solution
# (position+orientation rows only, no synthetic regularization rows to bias it)
J = result_noreg.jac
singular_values = np.linalg.svd(J, compute_uv=False)
condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 1e-12 else float("inf")

print("Jacobian singular values (unregularized combined model, 54 params):")
print(np.array2string(singular_values, precision=2, suppress_small=True))
print(f"\nCondition number: {condition_number:.1f}")
print("\n(Look for MULTIPLE very-small singular values beyond the 2 we already know")
print("about (joint0<->base_rpy, joint6<->tool_rpy) -- extra near-zero values would")
print("mean additional redundant/degenerate parameter combinations among the new")
print("axis-tilt/origin-delta parameters.)")
