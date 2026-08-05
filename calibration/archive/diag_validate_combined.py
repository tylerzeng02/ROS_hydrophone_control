import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

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
TILT_REG_WEIGHT = 15.0
TILT_SCALE_RAD = np.radians(5.0)
ORIGIN_REG_WEIGHT = 20.0
ORIGIN_SCALE_M = 0.005


def predict_combined(angles_row, base_params, tilt_params, origin_deltas):
    params = ck.unpack_params(base_params)
    axes = tilted_axes(tilt_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, axes, origin_deltas)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def make_residual(angles, pos_mm, quat_xyzw):
    def residual_combined(x):
        base_params = x[:19]
        tilt_flat = x[19:19 + N_TILT]
        origin_flat = x[19 + N_TILT:]
        tilt_params = tilt_flat.reshape(ck.N_JOINTS, 2)
        origin_deltas = origin_flat.reshape(ck.N_JOINTS, 3)

        n_local = len(angles)
        pos_res = np.zeros((n_local, 3))
        orient_res = np.zeros((n_local, 3))
        for i in range(n_local):
            T_pred = predict_combined(angles[i], base_params, tilt_params, origin_deltas)
            pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
            R_pred = T_pred[:3, :3]
            R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
            R_err = R_pred.T @ R_meas
            orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3

        params = ck.unpack_params(base_params)
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)),
            params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)),
            tilt_flat * (TILT_REG_WEIGHT / TILT_SCALE_RAD),
            origin_flat * (ORIGIN_REG_WEIGHT / ORIGIN_SCALE_M),
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual_combined


def rms_combined(angles, pos_mm, base_params, tilt_params, origin_deltas):
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict_combined(angles[i], base_params, tilt_params, origin_deltas)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


def simple_residual(x, angles, pos_mm, quat_xyzw):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


rng = np.random.default_rng(0)
perm = rng.permutation(n_all)
n_test = int(n_all * 0.2)
test_idx = perm[:n_test]
train_idx = perm[n_test:]

angles_tr, pos_tr, quat_tr = angles_all[train_idx], pos_mm_all[train_idx], quat_xyzw_all[train_idx]
angles_te, pos_te, quat_te = angles_all[test_idx], pos_mm_all[test_idx], quat_xyzw_all[test_idx]

print(f"Train: {len(train_idx)}  Test (held out): {len(test_idx)}\n")

# --- Simple 19-param model ---
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_tr[0], pos_tr[0], quat_tr[0])
x0_simple = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result_simple = least_squares(lambda x: simple_residual(x, angles_tr, pos_tr, quat_tr),
                               x0_simple, method="lm", max_nfev=100000)
train_rms_simple, _ = ck.rms_position_error_mm(result_simple.x, angles_tr, pos_tr)
test_rms_simple, _ = ck.rms_position_error_mm(result_simple.x, angles_te, pos_te)

print("19-param model:")
print(f"  train RMS: {train_rms_simple:.2f} mm   held-out RMS: {test_rms_simple:.2f} mm   gap: {test_rms_simple-train_rms_simple:+.2f} mm\n")

# --- Combined 54-param model ---
residual_fn = make_residual(angles_tr, pos_tr, quat_tr)
x0_combined = np.concatenate([x0_simple, np.zeros(N_TILT), np.zeros(N_ORIGIN)])
result_combined = least_squares(residual_fn, x0_combined, method="lm", max_nfev=300000)

base_params = result_combined.x[:19]
tilt_params = result_combined.x[19:19 + N_TILT].reshape(ck.N_JOINTS, 2)
origin_deltas = result_combined.x[19 + N_TILT:].reshape(ck.N_JOINTS, 3)

train_rms_combined, _ = rms_combined(angles_tr, pos_tr, base_params, tilt_params, origin_deltas)
test_rms_combined, _ = rms_combined(angles_te, pos_te, base_params, tilt_params, origin_deltas)

print("54-param model (axis tilt + origin deltas):")
print(f"  train RMS: {train_rms_combined:.2f} mm   held-out RMS: {test_rms_combined:.2f} mm   gap: {test_rms_combined-train_rms_combined:+.2f} mm\n")

print("Summary:")
print(f"  19-param:  train={train_rms_simple:.2f}  test={test_rms_simple:.2f}")
print(f"  54-param:  train={train_rms_combined:.2f}  test={test_rms_combined:.2f}")
if test_rms_combined < test_rms_simple:
    print("  -> 54-param model generalizes BETTER on held-out data: improvement looks real.")
else:
    print("  -> 54-param model does NOT generalize better (or is worse) on held-out data: likely overfitting.")
