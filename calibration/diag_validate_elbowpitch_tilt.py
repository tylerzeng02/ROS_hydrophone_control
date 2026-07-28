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

perp_vectors = {}
for j in [WRIST_PITCH_IDX, ELBOW_PITCH_IDX]:
    axis = ck.JOINT_AXES[j] / np.linalg.norm(ck.JOINT_AXES[j])
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    perp_vectors[j] = (u, v)


def fk_combined(joint_angles_rad, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX:
            u, v = perp_vectors[i]
            perturbed = a + tilt_wp[0] * u + tilt_wp[1] * v
            a = perturbed / np.linalg.norm(perturbed)
            origin = origin + o_wp
        if i == ELBOW_PITCH_IDX:
            u, v = perp_vectors[i]
            perturbed = a + tilt_ep[0] * u + tilt_ep[1] * v
            a = perturbed / np.linalg.norm(perturbed)
            origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + o_sy
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


def predict(angles_row, base_params, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def unpack(x):
    return x[:19], x[19:21], x[21:23], x[23:26], x[26:29], x[29:32]


def make_residual(angles, pos_mm, quat_xyzw):
    n = len(angles)

    def residual(x):
        base_params, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp = unpack(x)
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp)
            pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
            R_pred = T_pred[:3, :3]
            R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
            orient_res[i] = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)),
            params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)),
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def rms_of(angles, pos_mm, x):
    base_params, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


print("=== Multi-start robustness check (full 190 poses, 32 params, all jointly fit) ===")
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all)

rng = np.random.default_rng(777)
n_trials = 5
rms_results = []
last_result = None
for trial in range(n_trials):
    if trial == 0:
        x0 = np.concatenate([
            ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
            np.zeros(13)
        ])
    else:
        joint_perturb = np.radians(rng.uniform(-15, 15, size=7))
        tool_xyz_perturb = rng.uniform(-0.05, 0.05, size=3)
        tool_rpy_perturb = np.radians(rng.uniform(-90, 90, size=3))
        base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
        base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
        extra_perturb = rng.uniform(-0.02, 0.02, size=13)
        x0 = np.concatenate([
            ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                            base_xyz_perturb, base_rpy_perturb),
            extra_perturb
        ])
    result = least_squares(residual_full, x0, method="lm", max_nfev=200000)
    rms, _ = rms_of(angles_all, pos_mm_all, result.x)
    rms_results.append(rms)
    last_result = result
    ep_tilt_deg = np.degrees(np.linalg.norm(result.x[21:23]))
    print(f"  trial {trial}: RMS={rms:.3f} mm, elbow_pitch_tilt={ep_tilt_deg:.2f} deg")

S = np.linalg.svd(last_result.jac, compute_uv=False)
min_sv, cond = S[-1], (S[0] / S[-1] if S[-1] > 1e-12 else float("inf"))
print(f"\nRMS range: {min(rms_results):.3f} - {max(rms_results):.3f} mm")
print(f"Min singular value: {min_sv:.4f}   Condition number: {cond:.1f}")
print(f"Singular values < 1.0: {(S < 1.0).sum()} out of {len(S)}\n")

base_params, tilt_wp, tilt_ep, o_elbow, o_sy, o_wp = unpack(last_result.x)
print("Fitted values:")
print(f"  wrist_pitch tilt: {np.degrees(np.linalg.norm(tilt_wp)):.2f} deg (was 4.7 deg before)")
print(f"  elbow_pitch tilt (NEW): {np.degrees(np.linalg.norm(tilt_ep)):.2f} deg")
print(f"  elbow_pitch origin: {np.round(o_elbow*1000, 2)} mm (was [-2,4.5,6.6]mm before)")
print(f"  shoulder_yaw origin: {np.round(o_sy*1000, 2)} mm (was [-1.2,0,4.2]mm before)")
print(f"  wrist_pitch origin: {np.round(o_wp*1000, 2)} mm (was [-1.2,1.5,2.4]mm before)")

print("\n=== Held-out validation (random 80/20 split) ===")
rng2 = np.random.default_rng(0)
perm = rng2.permutation(n_all)
n_test = int(n_all * 0.2)
test_idx, train_idx = perm[:n_test], perm[n_test:]
angles_tr, pos_tr, quat_tr = angles_all[train_idx], pos_mm_all[train_idx], quat_xyzw_all[train_idx]
angles_te, pos_te = angles_all[test_idx], pos_mm_all[test_idx]

residual_tr = make_residual(angles_tr, pos_tr, quat_tr)
base_xyz0_tr, base_rpy0_tr = ck.initial_base_guess(angles_tr[0], pos_tr[0], quat_tr[0])
x0_tr = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0_tr, base_rpy0_tr),
    np.zeros(13)
])
result_tr = least_squares(residual_tr, x0_tr, method="lm", max_nfev=200000)
train_rms, _ = rms_of(angles_tr, pos_tr, result_tr.x)
test_rms, _ = rms_of(angles_te, pos_te, result_tr.x)
print(f"Train ({len(train_idx)} poses): RMS = {train_rms:.2f} mm")
print(f"Held-out ({len(test_idx)} poses): RMS = {test_rms:.2f} mm")
print(f"Gap: {test_rms - train_rms:+.2f} mm")

print(f"\nFor reference, current 4-correction validated model: RMS = 11.86 mm")
