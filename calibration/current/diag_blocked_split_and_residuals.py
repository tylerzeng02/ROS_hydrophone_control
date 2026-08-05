"""Follow-up to an external code review of this project (2026-07-29): the
review flagged that every held-out validation so far (see
diag_shoulder_pitch_fourier_fit.py) uses a *random* 80/20 split, but
TARGET_POSES is recorded as ~9-40-pose continuous per-joint trajectories --
consecutive entries differ by only tens of ticks on one joint. A random
split puts a near-twin of nearly every test pose into the training set, so
"held-out" RMS mostly measures interpolation between densely-sampled
neighbors, not generalization to a genuinely novel configuration. That
would explain why test RMS has repeatedly come out close to (or below)
train RMS across this investigation -- the signature of a leaky split, not
of a well-generalizing model.

This script re-validates the current best model (36 params + shoulder_pitch
1/rev Fourier term, see diag_shoulder_pitch_fourier_fit.py) two ways instead:

1. Blocked K-fold CV: fold boundaries are contiguous index ranges (pose_id
   order == collection order == trajectory order), so each held-out fold is
   a contiguous chunk of a trajectory, not scattered individual poses with
   neighbors leaking into training. This is the "spatially/temporally
   blocked" alternative the review suggested when a precise
   leave-one-trajectory-out segmentation isn't readily available.
2. Per-pose residual histogram from the full-dataset fit, to check the
   review's second claim: whether 8.187mm is a fairly uniform floor or a
   few outlier poses (occlusion-degraded but not error-gated by
   MAX_NDI_ERROR) dragging up an otherwise smaller bulk.

Model code here is intentionally duplicated from
diag_shoulder_pitch_fourier_fit.py (not imported) -- that file is a
top-level script with no __main__ guard, so importing it would re-run its
own multi-start fit and prints as a side effect. Every other diag_*.py in
this project follows the same duplicate-not-import convention.
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
    pose_id_all = np.array([int(row["pose_id"]) for row in csv.DictReader(f)])
assert len(pose_id_all) == n_all
assert np.all(np.diff(pose_id_all) >= 0), (
    "pose_id is not monotonically non-decreasing in the CSV -- the blocked "
    "split below assumes row order == collection/trajectory order."
)
print(f"Dataset: {n_all} poses (pose_id {pose_id_all.min()}-{pose_id_all.max()})\n")

SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX, SHOULDER_YAW_IDX, WRIST_PITCH_IDX = 1, 3, 2, 5
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


def predict(angles_row, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row * joint_scales + params.joint_offsets
    corrected_angles = corrected_angles.copy()
    raw_sp = angles_row[SHOULDER_PITCH_IDX]
    corrected_angles[SHOULDER_PITCH_IDX] = (
        corrected_angles[SHOULDER_PITCH_IDX]
        + fourier_ab[0] * np.sin(raw_sp)
        + fourier_ab[1] * np.cos(raw_sp)
    )
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
    fourier_ab = x[27 + N_SCALE + N_TILT:29 + N_SCALE + N_TILT]
    return base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab


TILT_REG_WEIGHT, TILT_SCALE_RAD, SCALE_REG_WEIGHT = 5.0, np.radians(8.0), 20.0
FOURIER_REG_WEIGHT = 5.0
FOURIER_SCALE_RAD = np.radians(5.0)


def make_residual(angles, pos_mm, quat_xyzw):
    n = len(angles)

    def residual(x):
        base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab = unpack(x)
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab)
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
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def per_pose_errors(angles, pos_mm, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab)
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
    ])
    upper = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf]*3, [np.inf]*3, [np.inf]*3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
        np.radians(15.0) * np.ones(2),
    ])
    return lower, upper


def x0_zero(angles, pos_mm, quat_xyzw):
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2)
    ])


def fit(angles, pos_mm, quat_xyzw):
    residual = make_residual(angles, pos_mm, quat_xyzw)
    lower, upper = build_bounds()
    result = least_squares(residual, x0_zero(angles, pos_mm, quat_xyzw),
                            method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
    return result


# ---------------------------------------------------------------------------
# 1. Blocked K-fold CV (contiguous index ranges, not random rows)
# ---------------------------------------------------------------------------
print("=== Blocked K-fold validation (contiguous pose_id ranges, not random rows) ===")
K = 8
fold_bounds = np.linspace(0, n_all, K + 1).astype(int)
pooled_test_errs = []
for k in range(K):
    lo, hi = fold_bounds[k], fold_bounds[k + 1]
    test_mask = np.zeros(n_all, dtype=bool)
    test_mask[lo:hi] = True
    train_mask = ~test_mask

    result = fit(angles_all[train_mask], pos_mm_all[train_mask], quat_xyzw_all[train_mask])
    train_rms, _ = rms_of(angles_all[train_mask], pos_mm_all[train_mask], result.x)
    test_errs = per_pose_errors(angles_all[test_mask], pos_mm_all[test_mask], result.x)
    test_rms = np.sqrt(np.mean(test_errs ** 2))
    pooled_test_errs.append(test_errs)

    print(f"  fold {k} (pose_id {pose_id_all[lo]}-{pose_id_all[hi-1]}, "
          f"{hi-lo} held out): train={train_rms:.2f}mm  test={test_rms:.2f}mm  "
          f"gap={test_rms - train_rms:+.2f}mm")

pooled_test_errs = np.concatenate(pooled_test_errs)
pooled_rms = np.sqrt(np.mean(pooled_test_errs ** 2))
print(f"\nPooled blocked-CV test RMS (honest, out-of-block, {n_all} poses total): "
      f"{pooled_rms:.2f}mm")
print("Compare against the random-80/20-split test RMS in "
      "diag_shoulder_pitch_fourier_fit.py and the 8.187mm full-dataset-fit "
      "figure -- if this number is meaningfully higher than both, the random "
      "split was leaking.\n")

# ---------------------------------------------------------------------------
# 2. Per-pose residual histogram from the full-dataset fit
# ---------------------------------------------------------------------------
print("=== Per-pose residual distribution (full-dataset fit, all 298 poses) ===")
full_result = fit(angles_all, pos_mm_all, quat_xyzw_all)
full_rms, full_errs = rms_of(angles_all, pos_mm_all, full_result.x)
print(f"Full-dataset RMS: {full_rms:.3f}mm (reference: 8.187mm)\n")

edges = [0, 2, 4, 6, 8, 10, 15, 20, 30, np.inf]
counts, _ = np.histogram(full_errs, bins=edges)
print("Residual histogram (mm):")
for i, c in enumerate(counts):
    lo, hi = edges[i], edges[i + 1]
    hi_label = f"{hi:.0f}" if np.isfinite(hi) else "inf"
    bar = "#" * c
    print(f"  {lo:>4.0f}-{hi_label:<4}: {c:3d} {bar}")

pctiles = [50, 75, 90, 95, 99, 100]
print("\nPercentiles (mm):", ", ".join(
    f"p{p}={np.percentile(full_errs, p):.2f}" for p in pctiles
))

n_worst = 10
worst_idx = np.argsort(full_errs)[::-1][:n_worst]
print(f"\nWorst {n_worst} poses (pose_id, error mm) -- cross-check these against "
      f"moving_camera_error/fixed_camera_error/moving_relative_fixed_error in "
      f"the CSV for that row (open finding #3 in CLAUDE.md/the review: quality "
      f"gate only screens single-sample noise, not a biased-but-consistent pose):")
for idx in worst_idx:
    print(f"  pose_id={pose_id_all[idx]:3d}  error={full_errs[idx]:6.2f}mm")
