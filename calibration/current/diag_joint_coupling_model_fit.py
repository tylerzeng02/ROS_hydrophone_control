"""Fix #2 from the blocked-split follow-up (2026-07-29): add an actual
cross-joint coupling correction term to the FK model, rather than just
checking residual-vs-product correlation as CLAUDE.md's earlier
diag_joint_coupling_refresh.py / diag_partial_coupling_check.py did.

Rationale for keeping this small (3 terms, not the full 21-pair matrix
CLAUDE.md flagged as "a much bigger, less-constrained parameter space...
with its own likely identifiability problems"): the earlier screening
already found the 3 strongest raw correlations, and diag_partial_coupling_
check.py's ambiguous result was itself confounded by testing single-joint
vs pairwise correlation over the WHOLE 298-pose dataset -- which is
dominated by single-joint sweeps that carry zero coupling signal by
construction. Restricting to the top 3 previously-flagged pairs keeps this
identifiable given that only a narrow pose_id band (~191-227) actually
exercises multi-joint-simultaneous motion.

Coupling pairs and which joint's effective angle absorbs the correction
(the Cyton is cable-driven, so an upstream joint's motion mechanically
perturbing a downstream joint's true angle is physically plausible --
this "downstream absorbs it" choice is a modeling assumption, not a
measured fact):
  (shoulder_roll, shoulder_yaw)  -> correction on shoulder_yaw  (0,2)->2
  (shoulder_yaw,  elbow_yaw)     -> correction on elbow_yaw     (2,4)->4
  (shoulder_pitch,elbow_pitch)   -> correction on elbow_pitch   (1,3)->3

Validates against the *blocked* (not random) split this time, since the
random 80/20 split was just shown to leak. Baseline to beat, from the
prior run (diag_blocked_split_and_residuals.py):
  full-dataset RMS (no coupling):        8.187 mm
  pooled blocked-CV test RMS:            11.85 mm
  fold 5 (pose_id 191-227) test RMS:     20.48 mm  <- the multi-joint region
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

SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6

# (source_i, source_j, target_joint_that_absorbs_the_correction)
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
# Layout: base(19) + scale(7) + tilt(14) + o_elbow(3) + o_sy_xz(2) + o_wp(3)
#         + fourier(2) + coupling(N_COUPLE)
OFF_BASE = 0
OFF_SCALE = 19
OFF_TILT = OFF_SCALE + N_SCALE
OFF_OELBOW = OFF_TILT + N_TILT
OFF_OSY = OFF_OELBOW + 3
OFF_OWP = OFF_OSY + 2
OFF_FOURIER = OFF_OWP + 3
OFF_COUPLE = OFF_FOURIER + 2
TOTAL_PARAMS = OFF_COUPLE + N_COUPLE


def unpack(x):
    base_params = x[OFF_BASE:OFF_BASE + 19]
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
COUPLE_REG_WEIGHT, COUPLE_SCALE = 5.0, 0.1  # rad^-1; regularizes toward zero (no coupling)
COUPLE_BOUND = 0.3  # rad^-1


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


def x0_zero(angles, pos_mm, quat_xyzw):
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE)
    ])


def fit(angles, pos_mm, quat_xyzw):
    residual = make_residual(angles, pos_mm, quat_xyzw)
    lower, upper = build_bounds()
    result = least_squares(residual, x0_zero(angles, pos_mm, quat_xyzw),
                            method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
    return result


assert TOTAL_PARAMS == 19 + N_SCALE + N_TILT + 3 + 2 + 3 + 2 + N_COUPLE
print(f"Model: {TOTAL_PARAMS} params (36 base+tilt+scale+origin+fourier + {N_COUPLE} coupling)\n")

# ---------------------------------------------------------------------------
# 1. Multi-start robustness + condition number (full dataset)
# ---------------------------------------------------------------------------
print("=== Multi-start robustness (full 298-pose dataset) ===")
rng = np.random.default_rng(555)
n_trials = 4
rms_results = []
last_result = None
for trial in range(n_trials):
    x0 = x0_zero(angles_all, pos_mm_all, quat_xyzw_all)
    if trial > 0:
        perturb = rng.uniform(-0.02, 0.02, size=TOTAL_PARAMS)
        x0 = x0 + perturb
        lower, upper = build_bounds()
        x0 = np.clip(x0, lower, upper)
    lower, upper = build_bounds()
    result = least_squares(make_residual(angles_all, pos_mm_all, quat_xyzw_all), x0,
                            method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
    rms, _ = rms_of(angles_all, pos_mm_all, result.x)
    rms_results.append(rms)
    last_result = result
    print(f"  trial {trial}: RMS={rms:.3f} mm (status={result.status}, nfev={result.nfev})")

S = np.linalg.svd(last_result.jac, compute_uv=False)
cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
print(f"\nRMS range: {min(rms_results):.3f} - {max(rms_results):.3f} mm  (reference w/o coupling: 8.187mm)")
print(f"Condition number: {cond:.1f} " +
      ("(healthy)" if cond < 1e4 else "(WARNING: high, check for new degeneracy)"))

_, _, _, _, _, _, _, couple_c = unpack(last_result.x)
names = ["shoulder_roll*shoulder_yaw -> shoulder_yaw",
         "shoulder_yaw*elbow_yaw -> elbow_yaw",
         "shoulder_pitch*elbow_pitch -> elbow_pitch"]
print("\nFitted coupling coefficients (rad^-1) and regularization-bound context:")
for name, c in zip(names, couple_c):
    print(f"  {name}: {c:+.4f}  (bound ±{COUPLE_BOUND})")

# ---------------------------------------------------------------------------
# 2. Blocked K-fold CV -- the honest test, especially fold 5
# ---------------------------------------------------------------------------
print("\n=== Blocked K-fold validation (same 8 contiguous folds as before) ===")
K = 8
fold_bounds = np.linspace(0, n_all, K + 1).astype(int)
pooled_test_errs = []
fold5_test_rms = None
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
    if k == 5:
        fold5_test_rms = test_rms

    print(f"  fold {k} (pose_id {pose_id_all[lo]}-{pose_id_all[hi-1]}, "
          f"{hi-lo} held out): train={train_rms:.2f}mm  test={test_rms:.2f}mm  "
          f"gap={test_rms - train_rms:+.2f}mm")

pooled_test_errs = np.concatenate(pooled_test_errs)
pooled_rms = np.sqrt(np.mean(pooled_test_errs ** 2))
print(f"\nPooled blocked-CV test RMS WITH coupling: {pooled_rms:.2f}mm "
      f"(reference WITHOUT coupling: 11.85mm)")
print(f"Fold 5 (the multi-joint-simultaneous region) test RMS WITH coupling: "
      f"{fold5_test_rms:.2f}mm (reference WITHOUT coupling: 20.48mm)")
if fold5_test_rms is not None and fold5_test_rms < 20.48 - 2.0:
    print("-> Real improvement in the region coupling was supposed to help. "
          "Evidence FOR fix #2 (missing structural term).")
elif pooled_rms < 11.85 - 0.5:
    print("-> Modest aggregate improvement without a clear fold-5-specific "
          "win. Weak/mixed evidence.")
else:
    print("-> No meaningful improvement. Evidence AGAINST fix #2 as the "
          "explanation for fold 5 -- points back toward fix #1 (coverage "
          "gap) or a session-specific issue in that pose_id range.")
