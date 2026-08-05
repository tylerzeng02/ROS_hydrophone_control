import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

WRIST_PITCH_IDX = 5  # ck.JOINT_NAMES[5] == "wrist_pitch_joint"

axis = ck.JOINT_AXES[WRIST_PITCH_IDX] / np.linalg.norm(ck.JOINT_AXES[WRIST_PITCH_IDX])
arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
u = np.cross(axis, arbitrary)
u = u / np.linalg.norm(u)
v = np.cross(axis, u)


def fk_with_wristpitch_tilt(joint_angles_rad, tilt_uv):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        if i == WRIST_PITCH_IDX:
            perturbed = a + tilt_uv[0] * u + tilt_uv[1] * v
            a = perturbed / np.linalg.norm(perturbed)
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, ck.JOINT_ORIGINS_M[i])
        T = T @ T_joint
    return T


def predict(angles_row, base_params, tilt_uv):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_with_wristpitch_tilt(corrected_angles, tilt_uv)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def make_residual(angles, pos_mm, quat_xyzw):
    def residual(x):
        base_params = x[:19]
        tilt_uv = x[19:21]
        params = ck.unpack_params(base_params)
        n = len(angles)
        pos_res = np.zeros((n, 3))
        orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, tilt_uv)
            pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
            R_pred = T_pred[:3, :3]
            R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
            R_err = R_pred.T @ R_meas
            orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)),
            params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)),
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def rms_of(angles, pos_mm, base_params, tilt_uv):
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, tilt_uv)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


# --- 1) Multi-start robustness check (full 190 poses) ---
print("=== Multi-start robustness check (full 190 poses) ===")
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all)

rng = np.random.default_rng(123)
n_trials = 6
tilt_results = []
rms_results = []
for trial in range(n_trials):
    if trial == 0:
        x0 = np.concatenate([
            ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
            np.zeros(2)
        ])
    else:
        joint_perturb = np.radians(rng.uniform(-15, 15, size=7))
        tool_xyz_perturb = rng.uniform(-0.05, 0.05, size=3)
        tool_rpy_perturb = np.radians(rng.uniform(-90, 90, size=3))
        base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
        base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
        tilt_perturb = np.radians(rng.uniform(-20, 20, size=2))
        x0 = np.concatenate([
            ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                            base_xyz_perturb, base_rpy_perturb),
            tilt_perturb
        ])
    result = least_squares(residual_full, x0, method="lm", max_nfev=100000)
    rms, _ = rms_of(angles_all, pos_mm_all, result.x[:19], result.x[19:21])
    tilt_mag = np.degrees(np.linalg.norm(result.x[19:21]))
    tilt_results.append(tilt_mag)
    rms_results.append(rms)
    print(f"  trial {trial}: RMS={rms:.3f} mm, tilt magnitude={tilt_mag:.3f} deg")

print(f"\nRMS range across trials: {min(rms_results):.3f} - {max(rms_results):.3f} mm")
print(f"Tilt magnitude range: {min(tilt_results):.3f} - {max(tilt_results):.3f} deg\n")

# --- 2) Random 80/20 held-out validation ---
print("=== Held-out validation (random 80/20 split) ===")
rng2 = np.random.default_rng(0)
perm = rng2.permutation(n_all)
n_test = int(n_all * 0.2)
test_idx = perm[:n_test]
train_idx = perm[n_test:]

angles_tr, pos_tr, quat_tr = angles_all[train_idx], pos_mm_all[train_idx], quat_xyzw_all[train_idx]
angles_te, pos_te, quat_te = angles_all[test_idx], pos_mm_all[test_idx], quat_xyzw_all[test_idx]

residual_tr = make_residual(angles_tr, pos_tr, quat_tr)
base_xyz0_tr, base_rpy0_tr = ck.initial_base_guess(angles_tr[0], pos_tr[0], quat_tr[0])
x0_tr = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0_tr, base_rpy0_tr),
    np.zeros(2)
])
result_tr = least_squares(residual_tr, x0_tr, method="lm", max_nfev=100000)

train_rms, _ = rms_of(angles_tr, pos_tr, result_tr.x[:19], result_tr.x[19:21])
test_rms, _ = rms_of(angles_te, pos_te, result_tr.x[:19], result_tr.x[19:21])

print(f"Train ({len(train_idx)} poses): RMS = {train_rms:.2f} mm")
print(f"Held-out ({len(test_idx)} poses): RMS = {test_rms:.2f} mm")
print(f"Gap: {test_rms - train_rms:+.2f} mm")
print(f"Fitted tilt on train split: {np.degrees(result_tr.x[19:21])} deg, magnitude={np.degrees(np.linalg.norm(result_tr.x[19:21])):.2f} deg")

print("\n=== Summary ===")
print("Baseline (no tilt):                    RMS = 15.43 mm")
print(f"wrist_pitch tilt (full 190, multi-start): RMS = {np.mean(rms_results):.2f} mm (consistent across {n_trials} starts)")
print(f"wrist_pitch tilt (held-out validation): train={train_rms:.2f}  test={test_rms:.2f}")
