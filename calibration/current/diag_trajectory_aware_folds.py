"""Trajectory-boundary-aware fold construction (2026-07-30), replacing the
blunt "contiguous array index" folds used so far. Those folds only worked
because pose_id order happens to track collection/trajectory order, but a
fold boundary could land mid-trajectory, and any one fold's composition is
whatever trajectories happen to fall in that index range (e.g. fold 5
previously turned out to be almost entirely the multi-joint-simultaneous
region -- useful for finding it, but not a deliberately diversified fold).

This script instead:
  1. Detects real trajectory boundaries from the TARGET_TICK columns (a
     large jump in overall 7-joint tick-space distance between consecutive
     commanded poses signals a new trajectory/session started, as opposed
     to the next incremental step of the same sweep).
  2. Assigns whole trajectories round-robin across 8 folds (trajectory 0
     -> fold 0, trajectory 1 -> fold 1, ..., trajectory 8 -> fold 0, ...),
     so each fold gets a MIX of different trajectory types instead of one
     narrow region, while never splitting a single trajectory across
     train/test (preserving the anti-leakage property).
  3. Re-validates the current-best, already-confirmed 3-coupling-term
     model (no re-search for a 4th term -- that was separately ruled out)
     against these new folds, and specifically re-checks the previously
     flagged bad region (pose_id 191-227) using whichever fold's model
     held each of its poses out, since that region may now be split across
     several folds instead of sitting in just one.

Reference numbers from the previous (contiguous-index) blocked CV:
  pooled test RMS: 9.87mm
  pose_id 191-227 region (all in old fold 5): test RMS 17.08mm
Reference from the leaky random 80/20 split: ~8.187mm (pre-coupling model)
/ 6.874mm full-fit (with coupling).
"""
import csv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))
pose_id_all = np.array([int(r["pose_id"]) for r in rows])
target_ticks_all = np.array([
    [float(r[f"target_tick_{i}"]) for i in range(7)] for r in rows
])
assert len(pose_id_all) == n_all

# ---------------------------------------------------------------------------
# 1. Trajectory boundary detection
# ---------------------------------------------------------------------------
diffs = target_ticks_all[1:] - target_ticks_all[:-1]
step_dist = np.linalg.norm(diffs, axis=1)

print("Step-distance percentiles (ticks, L2 over all 7 joints):")
for p in [50, 75, 80, 85, 90, 95]:
    print(f"  p{p}: {np.percentile(step_dist, p):.0f}")

THRESHOLD_PERCENTILE = 85
threshold = np.percentile(step_dist, THRESHOLD_PERCENTILE)
boundary_after = step_dist > threshold  # boundary_after[i] means a new trajectory starts at row i+1

segment_starts = [0] + [i + 1 for i, b in enumerate(boundary_after) if b]
segment_starts = sorted(set(segment_starts))
segment_bounds = list(zip(segment_starts, segment_starts[1:] + [n_all]))
seg_lengths = [hi - lo for lo, hi in segment_bounds]

print(f"\nThreshold (p{THRESHOLD_PERCENTILE}): {threshold:.0f} ticks")
print(f"Detected {len(segment_bounds)} trajectory segments")
print(f"Segment length: min={min(seg_lengths)}, median={np.median(seg_lengths):.0f}, "
      f"max={max(seg_lengths)}, mean={np.mean(seg_lengths):.1f}")

# ---------------------------------------------------------------------------
# 2. Round-robin assign segments to 8 folds
# ---------------------------------------------------------------------------
K = 8
fold_of_row = np.zeros(n_all, dtype=int)
for seg_idx, (lo, hi) in enumerate(segment_bounds):
    fold_of_row[lo:hi] = seg_idx % K

print("\nFold composition (poses per fold, and how many segments contribute):")
for k in range(K):
    n_segs_k = sum(1 for seg_idx in range(len(segment_bounds)) if seg_idx % K == k)
    print(f"  fold {k}: {np.sum(fold_of_row == k)} poses, from {n_segs_k} segments")

bad_region_mask = (pose_id_all >= 191) & (pose_id_all <= 227)
print(f"\nPreviously-flagged bad region (pose_id 191-227, {bad_region_mask.sum()} poses) "
      f"is now spread across folds: {sorted(set(fold_of_row[bad_region_mask].tolist()))}")

# ---------------------------------------------------------------------------
# Model (current best: 38 base params + 3 confirmed coupling terms)
# ---------------------------------------------------------------------------
SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6
COUPLE_TERMS = [
    (SHOULDER_ROLL_IDX, SHOULDER_YAW_IDX, SHOULDER_YAW_IDX),
    (SHOULDER_YAW_IDX, ELBOW_YAW_IDX, ELBOW_YAW_IDX),
    (SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX, ELBOW_PITCH_IDX),
]
N_COUPLE = len(COUPLE_TERMS)

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


N_TILT, N_SCALE = 14, 7
OFF_SCALE = 19
OFF_TILT = OFF_SCALE + N_SCALE
OFF_OELBOW = OFF_TILT + N_TILT
OFF_OSY = OFF_OELBOW + 3
OFF_OWP = OFF_OSY + 2
OFF_FOURIER = OFF_OWP + 3
OFF_COUPLE = OFF_FOURIER + 2
TOTAL_PARAMS = OFF_COUPLE + N_COUPLE


def unpack(x):
    base_params = x[0:19]
    joint_scales = x[OFF_SCALE:OFF_SCALE + N_SCALE]
    tilt_all = x[OFF_TILT:OFF_TILT + N_TILT].reshape(7, 2)
    o_elbow = x[OFF_OELBOW:OFF_OELBOW + 3]
    o_sy_xz = x[OFF_OSY:OFF_OSY + 2]
    o_wp = x[OFF_OWP:OFF_OWP + 3]
    fourier_ab = x[OFF_FOURIER:OFF_FOURIER + 2]
    couple_c = x[OFF_COUPLE:OFF_COUPLE + N_COUPLE]
    return base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c


def predict(angles_row, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c):
    params = ck.unpack_params(base_params)
    corrected_angles = (angles_row * joint_scales + params.joint_offsets).copy()
    raw_sp = angles_row[SHOULDER_PITCH_IDX]
    corrected_angles[SHOULDER_PITCH_IDX] += (
        fourier_ab[0] * np.sin(raw_sp) + fourier_ab[1] * np.cos(raw_sp)
    )
    for k, (i, j, tgt) in enumerate(COUPLE_TERMS):
        corrected_angles[tgt] += couple_c[k] * angles_row[i] * angles_row[j]
    T_fk = fk_combined(corrected_angles, tilt_all, o_elbow, o_sy_xz, o_wp)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


TILT_REG_WEIGHT, TILT_SCALE_RAD, SCALE_REG_WEIGHT = 5.0, np.radians(8.0), 20.0
FOURIER_REG_WEIGHT, FOURIER_SCALE_RAD = 5.0, np.radians(5.0)
COUPLE_REG_WEIGHT, COUPLE_SCALE = 5.0, 0.1
COUPLE_BOUND = 0.3


def make_residual(angles, pos_mm, quat_xyzw):
    n = len(angles)

    def residual(x):
        base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c = unpack(x)
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c)
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
            fourier_ab * (FOURIER_REG_WEIGHT / FOURIER_SCALE_RAD),
            couple_c * (COUPLE_REG_WEIGHT / COUPLE_SCALE),
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def per_pose_errors(angles, pos_mm, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return errs


def rms_of(angles, pos_mm, x):
    errs = per_pose_errors(angles, pos_mm, x)
    return np.sqrt(np.mean(errs ** 2)), errs


def build_bounds():
    lower = np.concatenate([
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf]*3, [-np.inf]*3, [-np.inf]*3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT),
        -0.05 * np.ones(3), -0.05 * np.ones(2), -0.05 * np.ones(3),
        -np.radians(15.0) * np.ones(2),
        -COUPLE_BOUND * np.ones(N_COUPLE),
    ])
    upper = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf]*3, [np.inf]*3, [np.inf]*3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
        np.radians(15.0) * np.ones(2),
        COUPLE_BOUND * np.ones(N_COUPLE),
    ])
    return lower, upper


def x0_zero():
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE)
    ])


def fit(angles, pos_mm, quat_xyzw):
    residual = make_residual(angles, pos_mm, quat_xyzw)
    lower, upper = build_bounds()
    result = least_squares(residual, x0_zero(), method="trf", x_scale="jac",
                            bounds=(lower, upper), max_nfev=5000)
    return result


# ---------------------------------------------------------------------------
# 3. Blocked CV with trajectory-aware, round-robin-diversified folds
# ---------------------------------------------------------------------------
print(f"\n=== Blocked CV with trajectory-aware folds ===")
pooled_test_errs = []
pooled_test_pose_ids = []
bad_region_errs = []
for k in range(K):
    test_mask = fold_of_row == k
    train_mask = ~test_mask
    result = fit(angles_all[train_mask], pos_mm_all[train_mask], quat_xyzw_all[train_mask])
    train_rms, _ = rms_of(angles_all[train_mask], pos_mm_all[train_mask], result.x)
    test_errs = per_pose_errors(angles_all[test_mask], pos_mm_all[test_mask], result.x)
    test_rms = np.sqrt(np.mean(test_errs ** 2))
    pooled_test_errs.append(test_errs)
    pooled_test_pose_ids.append(pose_id_all[test_mask])

    bad_in_fold = bad_region_mask[test_mask]
    if bad_in_fold.any():
        bad_region_errs.append(test_errs[bad_in_fold])

    print(f"  fold {k} ({test_mask.sum()} held out): train={train_rms:.2f}mm "
          f"test={test_rms:.2f}mm gap={test_rms - train_rms:+.2f}mm "
          f"({bad_in_fold.sum()} bad-region poses in this fold)")

pooled_test_errs = np.concatenate(pooled_test_errs)
pooled_rms = np.sqrt(np.mean(pooled_test_errs ** 2))
print(f"\nPooled trajectory-aware blocked-CV test RMS: {pooled_rms:.2f}mm "
      f"(reference, contiguous-index folds: 9.87mm)")

if bad_region_errs:
    bad_region_errs = np.concatenate(bad_region_errs)
    bad_rms = np.sqrt(np.mean(bad_region_errs ** 2))
    print(f"Bad-region (pose_id 191-227) test RMS, now spread across "
          f"{len(bad_region_errs)} out-of-fold predictions from multiple "
          f"folds: {bad_rms:.2f}mm (reference, old fold 5 alone: 17.08mm)")
