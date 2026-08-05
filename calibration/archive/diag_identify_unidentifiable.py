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

PARAM_NAMES = (
    [f"joint_offset[{i}]({ck.JOINT_NAMES[i]})" for i in range(7)] +
    ["tool_x", "tool_y", "tool_z", "tool_roll", "tool_pitch", "tool_yaw"] +
    ["base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw"] +
    [f"tilt[{ck.JOINT_NAMES[i]}]_{c}" for i in range(7) for c in ("u", "v")] +
    [f"origin[{ck.JOINT_NAMES[i]}]_{c}" for i in range(7) for c in ("x", "y", "z")]
)
assert len(PARAM_NAMES) == 54


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


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0_base = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
x0 = np.concatenate([x0_base, np.zeros(N_TILT), np.zeros(N_ORIGIN)])

result = least_squares(residual_no_reg, x0, method="lm", max_nfev=300000)

rms_check, _ = (lambda: (
    np.sqrt(np.mean(np.array([
        np.linalg.norm(pos_mm[i] - predict_combined(angles[i], *unpack_combined(result.x))[:3, 3] * 1000.0)
        for i in range(n)
    ]) ** 2)), None
))()
print(f"RMS of this run: {rms_check:.3f} mm")

U, S, Vt = np.linalg.svd(result.jac)
print("Raw singular values:")
print(np.array2string(S, precision=4, suppress_small=False))
print()
V = Vt.T  # columns are right singular vectors

# Null-space columns: singular values below a tiny threshold
null_mask = S < 1.0  # huge gap between real values (>400) and null ones (<0.001)
n_null = null_mask.sum()
print(f"Number of (near-)zero singular values: {n_null} out of {len(S)}\n")

null_vectors = V[:, -n_null:] if n_null > 0 else V[:, len(S):]
# For each ORIGINAL parameter, how much of its basis direction projects onto
# the unidentifiable null space (sum of squared components across all null vectors)
projection = np.sum(null_vectors ** 2, axis=1)

order = np.argsort(-projection)
print(f"{'parameter':40s} {'null-space projection':>22s}")
for idx in order:
    marker = "  <-- LARGELY UNIDENTIFIABLE" if projection[idx] > 0.3 else ""
    print(f"{PARAM_NAMES[idx]:40s} {projection[idx]:22.3f}{marker}")
