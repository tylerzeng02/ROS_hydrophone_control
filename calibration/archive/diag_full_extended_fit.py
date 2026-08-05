"""Extend the validated all-7-tilt model (8.98mm) with the two newly
confirmed-safe origin components: shoulder_pitch-origin-Y and
elbow_yaw-origin-Z (see diag_check_remaining_origin_gaps.py -- both
verified exactly orthogonal to every parameter already in the model,
so adding them should not re-trigger the degeneracy that hit
elbow_pitch-Y/shoulder_yaw-Y).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)
print(f"Dataset: {n_all} poses\n")

SHOULDER_PITCH_IDX = 1
ELBOW_PITCH_IDX = 3
SHOULDER_YAW_IDX = 2
ELBOW_YAW_IDX = 4
WRIST_PITCH_IDX = 5

PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    PERP.append((u, v))


def fk_combined(joint_angles_rad, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z):
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
        if i == SHOULDER_PITCH_IDX:
            origin = origin + np.array([0.0, o_sp_y[0], 0.0])
        if i == ELBOW_YAW_IDX:
            origin = origin + np.array([0.0, 0.0, o_ey_z[0]])
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(joint_rotation, origin)
    return T


def predict(angles_row, base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


# layout: 19 base | 14 tilt | 3 elbow_pitch origin | 2 shoulder_yaw origin(x,z)
#         | 3 wrist_pitch origin | 1 shoulder_pitch origin(y) | 1 elbow_yaw origin(z)
N_TILT = 14
N_EXTRA = N_TILT + 3 + 2 + 3 + 1 + 1  # 24


def unpack(x):
    base_params = x[:19]
    tilt_all = x[19:19 + N_TILT].reshape(7, 2)
    o_elbow = x[19 + N_TILT:22 + N_TILT]
    o_sy_xz = x[22 + N_TILT:24 + N_TILT]
    o_wp = x[24 + N_TILT:27 + N_TILT]
    o_sp_y = x[27 + N_TILT:28 + N_TILT]
    o_ey_z = x[28 + N_TILT:29 + N_TILT]
    return base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z


TILT_REG_WEIGHT = 5.0
TILT_SCALE_RAD = np.radians(8.0)


def make_residual(angles, pos_mm, quat_xyzw):
    n = len(angles)

    def residual(x):
        base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z = unpack(x)
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z)
            pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
            R_pred = T_pred[:3, :3]
            R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
            orient_res[i] = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)),
            params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)),
            tilt_all.ravel() * (TILT_REG_WEIGHT / TILT_SCALE_RAD),
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def rms_of(angles, pos_mm, x):
    base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


def build_bounds():
    """Generous bounds -- at least 2-3x any value seen in any prior fit this
    session (tool_xyz got to ~30mm, tool_rpy to ~93deg, base_rpy to ~157deg,
    origins to ~14mm, tilts to ~6deg) -- so trf/x_scale='jac' gets a
    well-posed, numerically sane problem without clipping a real solution.
    base_xyz/base_rpy get no prior (established elsewhere) -- left unbounded.
    """
    lower = np.concatenate([
        -np.radians(15.0) * np.ones(7),      # joint_offsets
        -0.10 * np.ones(3),                  # tool_xyz (m)
        [-np.inf] * 3,                        # tool_rpy -- no small-value prior in practice
        [-np.inf] * 3,                        # base_xyz -- no prior
        [-np.inf] * 3,                        # base_rpy -- no prior
        -np.radians(15.0) * np.ones(N_TILT),  # tilts
        -0.05 * np.ones(3),                   # o_elbow (m)
        -0.05 * np.ones(2),                   # o_sy_xz (m)
        -0.05 * np.ones(3),                   # o_wp (m)
        -0.05 * np.ones(1),                   # o_sp_y (m)
        -0.05 * np.ones(1),                   # o_ey_z (m)
    ])
    upper = -lower
    return lower, upper


print("=== Multi-start robustness check (all 7 tilts + 5 origin corrections, incl. 2 new) ===")
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all)

rng = np.random.default_rng(999)
n_trials = 6
rms_results = []
last_result = None
for trial in range(n_trials):
    if trial == 0:
        x0 = np.concatenate([
            ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
            np.zeros(N_EXTRA)
        ])
    else:
        joint_perturb = np.radians(rng.uniform(-15, 15, size=7))
        tool_xyz_perturb = rng.uniform(-0.05, 0.05, size=3)
        tool_rpy_perturb = np.radians(rng.uniform(-90, 90, size=3))
        base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
        base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
        extra_perturb = rng.uniform(-0.02, 0.02, size=N_EXTRA)
        x0 = np.concatenate([
            ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                            base_xyz_perturb, base_rpy_perturb),
            extra_perturb
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
if cond > 1e4:
    print("WARNING: high condition number -- likely a new degeneracy. Inspect smallest singular vector.")
    Vt = np.linalg.svd(last_result.jac, compute_uv=True)[2]
    worst_dir = Vt[-1]
    param_names = (["joint_off_%d" % i for i in range(7)] + ["tool_x", "tool_y", "tool_z"] +
                   ["tool_r", "tool_p", "tool_yaw"] + ["base_x", "base_y", "base_z"] +
                   ["base_r", "base_p", "base_yaw"] +
                   [f"tilt{j}_{c}" for j in range(7) for c in range(2)] +
                   ["o_elbow_x", "o_elbow_y", "o_elbow_z", "o_sy_x", "o_sy_z",
                    "o_wp_x", "o_wp_y", "o_wp_z", "o_sp_y", "o_ey_z"])
    order = np.argsort(-np.abs(worst_dir))
    print("Top contributors to the near-null direction:")
    for idx in order[:8]:
        print(f"  {param_names[idx]:12s} {worst_dir[idx]:+.3f}")
else:
    print("Condition number healthy -- no new degeneracy detected.")

base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z = unpack(last_result.x)
JOINT_NAMES_SHORT = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw",
                     "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"]
print("\nFitted axis tilts (deg):")
for j in range(7):
    tilt_deg = np.degrees(np.linalg.norm(tilt_all[j]))
    print(f"  {JOINT_NAMES_SHORT[j]:16s} {tilt_deg:.2f} deg")
print(f"\nelbow_pitch origin (mm): {np.round(o_elbow*1000, 2)}")
print(f"shoulder_yaw origin x,z (mm): {np.round(o_sy_xz*1000, 2)}")
print(f"wrist_pitch origin (mm): {np.round(o_wp*1000, 2)}")
print(f"shoulder_pitch origin Y (mm, NEW): {np.round(o_sp_y*1000, 2)}")
print(f"elbow_yaw origin Z (mm, NEW): {np.round(o_ey_z*1000, 2)}")

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
    np.zeros(N_EXTRA)
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

print(f"\nFor reference: all-7-tilt model without these 2 new params = 8.98 mm")
