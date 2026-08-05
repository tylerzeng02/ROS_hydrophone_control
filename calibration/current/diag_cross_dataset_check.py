"""Sharper backlash-fix check: does the model fitted on OLD (uncompensated)
data predict the NEW (backlash-compensated) data any worse than it
predicts held-out OLD data? If the old fit was partly absorbing a
consistent backlash bias, its fixed parameters should mismatch the new,
bias-reduced measurements -- showing up as elevated cross-dataset error
compared to same-dataset held-out error.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

OLD_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
NEW_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_backlashfix_51pose.csv"

POSE_IDS = [1,7,13,19,25,31,37,43,49,55,61,67,73,79,85,91,97,103,109,115,121,
            127,133,145,151,157,163,169,175,181,187,193,205,211,217,223,229,
            235,241,247,253,259,265,271,277,283,289,295,301]


def load_subset_by_pose_id(csv_path, pose_ids):
    import csv as csv_mod
    angles, pos_mm, quat = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        rows = {int(row["pose_id"]): row for row in reader}
    for pid in pose_ids:
        row = rows[pid]
        a = np.array([float(row[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)])
        p = np.array([float(row["moving_relative_fixed_tx_mm"]),
                      float(row["moving_relative_fixed_ty_mm"]),
                      float(row["moving_relative_fixed_tz_mm"])])
        q0 = float(row["moving_relative_fixed_q0"])
        qx = float(row["moving_relative_fixed_qx"])
        qy = float(row["moving_relative_fixed_qy"])
        qz = float(row["moving_relative_fixed_qz"])
        angles.append(a); pos_mm.append(p); quat.append([qx, qy, qz, q0])
    return np.array(angles), np.array(pos_mm), np.array(quat)


ELBOW_PITCH_IDX, SHOULDER_YAW_IDX, WRIST_PITCH_IDX = 3, 2, 5
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


N_TILT, N_SCALE = 14, 7


def unpack(x):
    base_params = x[:19]
    joint_scales = x[19:19 + N_SCALE]
    tilt_all = x[19 + N_SCALE:19 + N_SCALE + N_TILT].reshape(7, 2)
    o_elbow = x[19 + N_SCALE + N_TILT:22 + N_SCALE + N_TILT]
    o_sy_xz = x[22 + N_SCALE + N_TILT:24 + N_SCALE + N_TILT]
    o_wp = x[24 + N_SCALE + N_TILT:27 + N_SCALE + N_TILT]
    return base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp


TILT_REG_WEIGHT, TILT_SCALE_RAD, SCALE_REG_WEIGHT = 5.0, np.radians(8.0), 20.0


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
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf]*3, [-np.inf]*3, [-np.inf]*3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT),
        -0.05 * np.ones(3), -0.05 * np.ones(2), -0.05 * np.ones(3),
    ])
    upper = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf]*3, [np.inf]*3, [np.inf]*3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
    ])
    return lower, upper


def fit_model(angles, pos_mm, quat):
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat[0])
    residual_full = make_residual(angles, pos_mm, quat)
    x0 = np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3)
    ])
    lower, upper = build_bounds()
    result = least_squares(residual_full, x0, method="trf", x_scale="jac",
                            bounds=(lower, upper), max_nfev=5000)
    return result.x


angles_old, pos_old, quat_old = load_subset_by_pose_id(OLD_CSV, POSE_IDS)
angles_new, pos_new, quat_new = load_subset_by_pose_id(NEW_CSV, POSE_IDS)
n = len(POSE_IDS)

# 80/20 split of the pose INDEX list, applied identically to both datasets
# (same poses held out in both, for a fair same-vs-cross comparison)
rng = np.random.default_rng(0)
perm = rng.permutation(n)
n_test = int(n * 0.2)
test_idx, train_idx = perm[:n_test], perm[n_test:]

print(f"Train: {len(train_idx)} poses, Held-out: {len(test_idx)} poses\n")

# Fit on OLD-train, check same-dataset held-out vs cross-dataset (NEW) held-out
x_old = fit_model(angles_old[train_idx], pos_old[train_idx], quat_old[train_idx])
rms_old_train, _ = rms_of(angles_old[train_idx], pos_old[train_idx], x_old)
rms_old_heldout, _ = rms_of(angles_old[test_idx], pos_old[test_idx], x_old)
rms_old_to_new_heldout, _ = rms_of(angles_new[test_idx], pos_new[test_idx], x_old)

print("=== Model FIT on OLD-train data ===")
print(f"  OLD train RMS:                        {rms_old_train:.2f} mm")
print(f"  OLD held-out RMS (same dataset):       {rms_old_heldout:.2f} mm")
print(f"  NEW held-out RMS (cross-dataset, SAME poses, backlash-compensated): {rms_old_to_new_heldout:.2f} mm")
print(f"  Cross-dataset gap: {rms_old_to_new_heldout - rms_old_heldout:+.2f} mm\n")

# And the mirror: fit on NEW-train, check against OLD held-out
x_new = fit_model(angles_new[train_idx], pos_new[train_idx], quat_new[train_idx])
rms_new_train, _ = rms_of(angles_new[train_idx], pos_new[train_idx], x_new)
rms_new_heldout, _ = rms_of(angles_new[test_idx], pos_new[test_idx], x_new)
rms_new_to_old_heldout, _ = rms_of(angles_old[test_idx], pos_old[test_idx], x_new)

print("=== Model FIT on NEW-train data ===")
print(f"  NEW train RMS:                        {rms_new_train:.2f} mm")
print(f"  NEW held-out RMS (same dataset):       {rms_new_heldout:.2f} mm")
print(f"  OLD held-out RMS (cross-dataset, SAME poses, uncompensated):        {rms_new_to_old_heldout:.2f} mm")
print(f"  Cross-dataset gap: {rms_new_to_old_heldout - rms_new_heldout:+.2f} mm\n")

print("=== Interpretation ===")
print("If backlash bias is real and direction-dependent, a model fit on one")
print("dataset should predict the OTHER dataset noticeably worse than it")
print("predicts more of its OWN dataset -- a real 'cross-dataset gap' in")
print("both directions above. A small/near-zero gap in both directions")
print("would mean the two datasets are consistent with each other, i.e.")
print("backlash isn't creating a meaningful discrepancy between them.")
