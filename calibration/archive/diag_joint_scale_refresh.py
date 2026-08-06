"""Refresh of the stale diag_joint_scale_fit.py: tests a per-joint
multiplicative gear-ratio error (corrected_angle = measured_angle*scale +
offset) distinct from the additive joint_offset already in the model.
Built on top of the current BEST validated model (19 base + 7 axis tilts
+ 3 origin corrections = 41 params, 8.98mm), on the current 298-pose
dataset, with the trf/x_scale='jac'/bounds methodology that fixed the
scaling problems in the previous (origin-param) test.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)
print(f"Dataset: {n_all} poses\n")

ELBOW_PITCH_IDX = 3
SHOULDER_YAW_IDX = 2
WRIST_PITCH_IDX = 5

PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    PERP.append((u, v))


def fk_combined(joint_angles_rad, tilt_all, o_elbow, o_sy_xz, o_wp):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        u, v = PERP[i]
        perturbed = a + tilt_all[i, 0] * u + tilt_all[i, 1] * v
        a = perturbed / np.linalg.norm(perturbed)
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX:
            origin = origin + o_wp
        if i == ELBOW_PITCH_IDX:
            origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(joint_rotation, origin)
    return T


def predict(angles_row, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row * joint_scales + params.joint_offsets
    T_fk = fk_combined(corrected_angles, tilt_all, o_elbow, o_sy_xz, o_wp)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


# layout: 19 base | 7 joint_scales | 14 tilt | 3 elbow_pitch origin | 2 shoulder_yaw origin(x,z) | 3 wrist_pitch origin
N_TILT = 14
N_SCALE = 7
N_EXTRA = N_SCALE + N_TILT + 3 + 2 + 3  # 29


def unpack(x):
    base_params = x[:19]
    joint_scales = x[19:19 + N_SCALE]
    tilt_all = x[19 + N_SCALE:19 + N_SCALE + N_TILT].reshape(7, 2)
    o_elbow = x[19 + N_SCALE + N_TILT:22 + N_SCALE + N_TILT]
    o_sy_xz = x[22 + N_SCALE + N_TILT:24 + N_SCALE + N_TILT]
    o_wp = x[24 + N_SCALE + N_TILT:27 + N_SCALE + N_TILT]
    return base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp


TILT_REG_WEIGHT = 5.0
TILT_SCALE_RAD = np.radians(8.0)
SCALE_REG_WEIGHT = 20.0  # mm-equivalent penalty pulling scale toward 1.0 (nominal)


def make_residual(angles, pos_mm, quat_xyzw):
    n = len(angles)

    def residual(x):
        base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp = unpack(x)
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp)
            pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
            R_pred = T_pred[:3, :3]
            R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
            orient_res[i] = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)),
            params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)),
            tilt_all.ravel() * (TILT_REG_WEIGHT / TILT_SCALE_RAD),
            (joint_scales - 1.0) * SCALE_REG_WEIGHT,
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def rms_of(angles, pos_mm, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


def build_bounds():
    lower = np.concatenate([
        -np.radians(15.0) * np.ones(7),
        -0.10 * np.ones(3),
        [-np.inf] * 3,
        [-np.inf] * 3,
        [-np.inf] * 3,
        0.90 * np.ones(N_SCALE),
        -np.radians(15.0) * np.ones(N_TILT),
        -0.05 * np.ones(3),
        -0.05 * np.ones(2),
        -0.05 * np.ones(3),
    ])
    upper = np.concatenate([
        np.radians(15.0) * np.ones(7),
        0.10 * np.ones(3),
        [np.inf] * 3,
        [np.inf] * 3,
        [np.inf] * 3,
        1.10 * np.ones(N_SCALE),
        np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3),
        0.05 * np.ones(2),
        0.05 * np.ones(3),
    ])
    return lower, upper


print("=== Multi-start robustness check (7 tilts + 3 origins + 7 NEW joint scales) ===")
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all)

rng = np.random.default_rng(31)
n_trials = 6
rms_results = []
last_result = None
for trial in range(n_trials):
    if trial == 0:
        x0 = np.concatenate([
            ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
            np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3)
        ])
    else:
        joint_perturb = np.radians(rng.uniform(-10, 10, size=7))
        tool_xyz_perturb = rng.uniform(-0.03, 0.03, size=3)
        tool_rpy_perturb = np.radians(rng.uniform(-60, 60, size=3))
        base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
        base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
        scale_perturb = 1.0 + rng.uniform(-0.02, 0.02, size=N_SCALE)
        extra_perturb = rng.uniform(-0.02, 0.02, size=N_TILT + 3 + 2 + 3)
        x0 = np.concatenate([
            ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                            base_xyz_perturb, base_rpy_perturb),
            scale_perturb, extra_perturb
        ])
    lower, upper = build_bounds()
    result = least_squares(residual_full, x0, method="trf", x_scale="jac",
                            bounds=(lower, upper), max_nfev=5000)
    rms, _ = rms_of(angles_all, pos_mm_all, result.x)
    rms_results.append(rms)
    last_result = result
    print(f"  trial {trial}: RMS={rms:.3f} mm  (status={result.status}, nfev={result.nfev})")

S = np.linalg.svd(last_result.jac, compute_uv=False)
cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
print(f"\nRMS range: {min(rms_results):.3f} - {max(rms_results):.3f} mm")
print(f"Smallest 5 singular values: {np.round(S[-5:], 4)}")
print(f"Condition number: {cond:.1f}")
print("Condition number healthy -- no degeneracy" if cond < 1e4 else "WARNING: high condition number")

base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp = unpack(last_result.x)
JOINT_NAMES_SHORT = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw",
                     "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"]
print("\nFitted joint scales (should be ~1.0 if no gear/belt ratio error):")
for j in range(7):
    print(f"  {JOINT_NAMES_SHORT[j]:16s} {joint_scales[j]:.5f}  ({(joint_scales[j]-1)*100:+.2f}%)")

print("\nFitted axis tilts WITH scale included (compare against 8.98mm-model's tilts:")
print("  shoulder_roll 1.47, shoulder_pitch 3.26, shoulder_yaw 4.18, elbow_pitch 1.99,")
print("  elbow_yaw 1.70, wrist_pitch 2.94, wrist_roll 4.02 -- from the trf all-7-tilt-only run):")
for j in range(7):
    tilt_deg = np.degrees(np.linalg.norm(tilt_all[j]))
    print(f"  {JOINT_NAMES_SHORT[j]:16s} {tilt_deg:.2f} deg")
print(f"\nelbow_pitch origin (mm), compare vs [-1.23, 13.1, 6.61]: {np.round(o_elbow*1000, 2)}")
print(f"shoulder_yaw origin x,z (mm), compare vs [-3.55, 1.92]: {np.round(o_sy_xz*1000, 2)}")
print(f"wrist_pitch origin (mm), compare vs [-0.65, 3.13, 4.11]: {np.round(o_wp*1000, 2)}")

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
    np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3)
])
lower, upper = build_bounds()
result_tr = least_squares(residual_tr, x0_tr, method="trf", x_scale="jac",
                           bounds=(lower, upper), max_nfev=5000)
print(f"(held-out fit: status={result_tr.status}, nfev={result_tr.nfev})")
train_rms, _ = rms_of(angles_tr, pos_tr, result_tr.x)
test_rms, _ = rms_of(angles_te, pos_te, result_tr.x)
print(f"Train ({len(train_idx)} poses): RMS = {train_rms:.2f} mm")
print(f"Held-out ({len(test_idx)} poses): RMS = {test_rms:.2f} mm")
print(f"Gap: {test_rms - train_rms:+.2f} mm")

print(f"\nFor reference: current best model without joint scales = 8.98 mm")
