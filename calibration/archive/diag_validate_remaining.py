import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

WRIST_PITCH_IDX = 5
ELBOW_PITCH_IDX = 3
SHOULDER_YAW_IDX = 2

axis0 = ck.JOINT_AXES[WRIST_PITCH_IDX] / np.linalg.norm(ck.JOINT_AXES[WRIST_PITCH_IDX])
arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
u0 = np.cross(axis0, arbitrary); u0 /= np.linalg.norm(u0)
v0 = np.cross(axis0, u0)
WRIST_PITCH_TILT_UV = np.radians(np.array([-0.63, -4.55]))
ELBOW_PITCH_ORIGIN_DELTA = np.array([-2.08, 4.46, 6.59]) / 1000.0
SHOULDER_YAW_ORIGIN_DELTA = np.array([-1.18, 0.04, 4.19]) / 1000.0


def fk_combined(joint_angles_rad, target_joint, origin_delta_target):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        if i == WRIST_PITCH_IDX:
            perturbed = a + WRIST_PITCH_TILT_UV[0] * u0 + WRIST_PITCH_TILT_UV[1] * v0
            a = perturbed / np.linalg.norm(perturbed)
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == ELBOW_PITCH_IDX:
            origin = origin + ELBOW_PITCH_ORIGIN_DELTA
        if i == SHOULDER_YAW_IDX:
            origin = origin + SHOULDER_YAW_ORIGIN_DELTA
        if i == target_joint:
            origin = origin + origin_delta_target
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


def predict(angles_row, base_params, target_joint, origin_delta_target):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, target_joint, origin_delta_target)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def make_residual(angles, pos_mm, quat_xyzw, target_joint):
    def residual(x):
        base_params = x[:19]
        origin_delta = x[19:22]
        params = ck.unpack_params(base_params)
        n = len(angles)
        pos_res = np.zeros((n, 3))
        orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, target_joint, origin_delta)
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


def rms_of(angles, pos_mm, base_params, target_joint, origin_delta):
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, target_joint, origin_delta)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


def validate_joint(target_joint, joint_name):
    print(f"\n{'='*60}\nValidating {joint_name} origin delta\n{'='*60}")

    base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
    residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all, target_joint)

    rng = np.random.default_rng(999 + target_joint)
    n_trials = 5
    rms_results = []
    last_result = None
    for trial in range(n_trials):
        if trial == 0:
            x0 = np.concatenate([
                ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
                np.zeros(3)
            ])
        else:
            joint_perturb = np.radians(rng.uniform(-15, 15, size=7))
            tool_xyz_perturb = rng.uniform(-0.05, 0.05, size=3)
            tool_rpy_perturb = np.radians(rng.uniform(-90, 90, size=3))
            base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
            base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
            origin_perturb = rng.uniform(-0.02, 0.02, size=3)
            x0 = np.concatenate([
                ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                                base_xyz_perturb, base_rpy_perturb),
                origin_perturb
            ])
        result = least_squares(residual_full, x0, method="lm", max_nfev=100000)
        rms, _ = rms_of(angles_all, pos_mm_all, result.x[:19], target_joint, result.x[19:22])
        rms_results.append(rms)
        last_result = result
        print(f"  trial {trial}: RMS={rms:.3f} mm, origin_delta={np.round(result.x[19:22]*1000,2)} mm")

    S = np.linalg.svd(last_result.jac, compute_uv=False)
    min_sv, cond = S[-1], (S[0] / S[-1] if S[-1] > 1e-12 else float("inf"))
    print(f"\nRMS range: {min(rms_results):.3f} - {max(rms_results):.3f} mm")
    print(f"Min singular value: {min_sv:.4f}   Condition number: {cond:.1f}")

    rng2 = np.random.default_rng(0)
    perm = rng2.permutation(n_all)
    n_test = int(n_all * 0.2)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    angles_tr, pos_tr, quat_tr = angles_all[train_idx], pos_mm_all[train_idx], quat_xyzw_all[train_idx]
    angles_te, pos_te = angles_all[test_idx], pos_mm_all[test_idx]

    residual_tr = make_residual(angles_tr, pos_tr, quat_tr, target_joint)
    base_xyz0_tr, base_rpy0_tr = ck.initial_base_guess(angles_tr[0], pos_tr[0], quat_tr[0])
    x0_tr = np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0_tr, base_rpy0_tr),
        np.zeros(3)
    ])
    result_tr = least_squares(residual_tr, x0_tr, method="lm", max_nfev=100000)
    train_rms, _ = rms_of(angles_tr, pos_tr, result_tr.x[:19], target_joint, result_tr.x[19:22])
    test_rms, _ = rms_of(angles_te, pos_te, result_tr.x[:19], target_joint, result_tr.x[19:22])
    print(f"Held-out: train={train_rms:.2f}mm  test={test_rms:.2f}mm  gap={test_rms-train_rms:+.2f}mm")


validate_joint(4, "elbow_yaw")
validate_joint(6, "wrist_roll")

print(f"\nFor reference, model before either of these: RMS = 11.99 mm")
