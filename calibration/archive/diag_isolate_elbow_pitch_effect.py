"""Cheap, no-new-hardware check: did elbow_pitch's (motor 3) backlash
correction specifically matter for the 51-pose validation set, or is the
near-neutral aggregate RMS masking a real elbow_pitch improvement offset
by noise from OTHER joints also getting corrected?

Method: reconstruct, from the NEW (20-tick) dataset's own actual capture
SEQUENCE (poses visited in QUICK_TEST_POSE_INDICES order: 0,6,12,...,300),
which pose-to-pose transitions involved elbow_pitch moving in the
"corrected" (decreasing target) direction. Fit a single reference model on
the OLD dataset, then compare how much each pose's measured position
differs from that model's prediction in OLD vs NEW data -- split by
whether elbow_pitch was corrected on that specific transition or not.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck
import csv as csv_mod

OLD_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
NEW_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_backlashfix_v2_20tick.csv"

# ---- Load NEW dataset IN CAPTURE ORDER (rows already ordered by pose_id,
# which matches the actual visitation sequence 0,6,12,...,300) ----
new_rows = []
with open(NEW_CSV, newline="") as f:
    reader = csv_mod.DictReader(f)
    for row in reader:
        new_rows.append(row)

# Determine which poses had elbow_pitch (target_tick_3) corrected: target
# decreased vs the PREVIOUS pose in this same capture sequence.
elbow_corrected = {}  # pose_id -> bool
other_corrected_count = {}  # pose_id -> how many OTHER joints were also corrected
prev_targets = None
for row in new_rows:
    pid = int(row["pose_id"])
    targets = [int(row[f"target_tick_{i}"]) for i in range(7)]
    if prev_targets is not None:
        elbow_corrected[pid] = targets[3] < prev_targets[3]
        other_count = sum(1 for i in range(7) if i != 3 and targets[i] < prev_targets[i])
        other_corrected_count[pid] = other_count
    prev_targets = targets

pose_ids_with_history = list(elbow_corrected.keys())  # excludes the very first pose (no prior)
print(f"Poses with a known predecessor in the capture sequence: {len(pose_ids_with_history)}")
print(f"  elbow_pitch corrected on:     {sum(elbow_corrected.values())} poses")
print(f"  elbow_pitch NOT corrected on: {sum(not v for v in elbow_corrected.values())} poses\n")


def load_subset_by_pose_id(csv_path, pose_ids):
    angles, pos_mm, quat = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        rows = {int(row["pose_id"]): row for row in reader}
    valid_ids = []
    for pid in pose_ids:
        if pid not in rows:
            continue
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
        valid_ids.append(pid)
    return np.array(angles), np.array(pos_mm), np.array(quat), valid_ids


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


def per_pose_error(angles, pos_mm, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return errs


# Fit reference model on OLD data (all poses available there)
all_old_ids = [int(r["pose_id"]) for r in csv_mod.DictReader(open(OLD_CSV))]
angles_old_all, pos_old_all, quat_old_all, ids_old_all = load_subset_by_pose_id(OLD_CSV, all_old_ids)
print("Fitting reference model on full OLD dataset...")
x_ref = fit_model(angles_old_all, pos_old_all, quat_old_all)
ref_rms, _ = np.sqrt(np.mean(per_pose_error(angles_old_all, pos_old_all, x_ref) ** 2)), None
print(f"Reference model RMS on OLD data: {ref_rms:.2f} mm\n")

# Now evaluate that SAME reference model against OLD and NEW measurements
# for the specific pose_ids we have history/correction info for.
angles_old_sub, pos_old_sub, quat_old_sub, ids_old_sub = load_subset_by_pose_id(OLD_CSV, pose_ids_with_history)
angles_new_sub, pos_new_sub, quat_new_sub, ids_new_sub = load_subset_by_pose_id(NEW_CSV, pose_ids_with_history)

assert ids_old_sub == ids_new_sub, "pose_id sets must match between old/new for a fair comparison"

err_old = per_pose_error(angles_old_sub, pos_old_sub, x_ref)
err_new = per_pose_error(angles_new_sub, pos_new_sub, x_ref)
delta = err_new - err_old  # negative = NEW is better (lower error) than OLD

elbow_flag = np.array([elbow_corrected[pid] for pid in ids_old_sub])
other_count = np.array([other_corrected_count[pid] for pid in ids_old_sub])

print("=== Per-pose error vs. OLD-fitted reference model ===")
print(f"{'pose_id':>8s} {'elbow_corr':>11s} {'other_corr':>11s} {'err_old':>9s} {'err_new':>9s} {'delta':>8s}")
for i, pid in enumerate(ids_old_sub):
    print(f"{pid:8d} {str(elbow_flag[i]):>11s} {other_count[i]:11d} {err_old[i]:9.2f} {err_new[i]:9.2f} {delta[i]:+8.2f}")

print("\n=== Summary: mean (err_new - err_old), negative = improvement ===")
print(f"  elbow_pitch corrected     (n={elbow_flag.sum()}): mean delta = {delta[elbow_flag].mean():+.2f} mm")
print(f"  elbow_pitch NOT corrected (n={(~elbow_flag).sum()}): mean delta = {delta[~elbow_flag].mean():+.2f} mm")

print("\nControlling for how many OTHER joints were also corrected on the same move:")
for n_other in sorted(set(other_count)):
    mask = other_count == n_other
    if mask.sum() < 2:
        continue
    print(f"  other_corrected={n_other} (n={mask.sum()}): "
          f"elbow corrected mean delta = "
          f"{delta[mask & elbow_flag].mean() if (mask & elbow_flag).any() else float('nan'):+.2f} mm, "
          f"elbow NOT corrected mean delta = "
          f"{delta[mask & ~elbow_flag].mean() if (mask & ~elbow_flag).any() else float('nan'):+.2f} mm")
