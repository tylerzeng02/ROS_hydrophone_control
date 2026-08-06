"""Lumped gravity/elastostatic deflection test (2026-07-30) -- the cheap
first test discussed as an alternative to full elastostatic calibration
(which needs real per-link mass/CoM data we don't have). Gravity torque at
any joint is LINEAR in the unknown mass-moment terms, so instead of needing
real masses, this fits an effective per-joint deflection coefficient
directly from the NDI position data already in hand -- same trf/bounds/
multi-start/blocked-CV methodology used for every other correction in this
project (tilt, scale, coupling).

Model: approximate all downstream mass as one equivalent point mass at the
current end-effector position (a standard first-pass simplification when
real per-link mass data isn't available). For each joint i, compute a
purely geometric "gravity moment arm" scalar from the FK chain itself (no
new hardware/measurement needed):

    lever_i(q)     = p_ee(q) - p_i(q)              (vector, meters)
    geometric_i(q) = (lever_i(q) x gravity_dir) . axis_i(q)

where p_i is joint i's pivot position and axis_i its rotation axis, both
in the FK chain's own (base-relative) frame, and gravity_dir = (0,0,-1) is
assumed to be the FK chain's own nominal "down" direction (i.e. assuming
base_rpy's fitted correction is small enough that base_link's nominal Z
axis is still a good proxy for true vertical -- a reasonable approximation
consistent with the other simplifications already accepted in this model).

The actual joint-angle deflection added is c_i * geometric_i(q), where c_i
lumps (effective mass * g / joint stiffness) into ONE fit coefficient per
joint (rad/m) -- 7 new parameters, same order of magnitude as the other
validated additions. Applied on top of the current-best 53-param model (38
base + 3 confirmed coupling terms).

Reference numbers to beat (3-coupling-term model, no gravity term):
  full-dataset RMS:            6.874mm
  blocked-CV pooled test RMS:  9.87mm
  fold 5 (pose_id 191-227):    17.08mm
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

SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6

COUPLE_TERMS = [
    (SHOULDER_ROLL_IDX, SHOULDER_YAW_IDX, SHOULDER_YAW_IDX),
    (SHOULDER_YAW_IDX, ELBOW_YAW_IDX, ELBOW_YAW_IDX),
    (SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX, ELBOW_PITCH_IDX),
]
N_COUPLE = len(COUPLE_TERMS)
N_GRAVITY = 7

PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    PERP.append((u, v))


def tilted_axes(tilt_all):
    axes = []
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        u, v = PERP[i]
        perturbed = a + tilt_all[i, 0] * u + tilt_all[i, 1] * v
        axes.append(perturbed / np.linalg.norm(perturbed))
    return axes


def fk_combined(joint_angles_rad, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes = tilted_axes(tilt_all)
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX:
            origin = origin + o_wp
        if i == ELBOW_PITCH_IDX:
            origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        joint_rotation = Rotation.from_rotvec(axes[i] * joint_angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(joint_rotation, origin)
    return T


def fk_with_joint_frames(joint_angles_rad, tilt_all, o_elbow, o_sy_xz, o_wp):
    """Like fk_combined, but also returns each joint's pivot position and
    (tilted) rotation axis direction, both expressed in the FK chain's own
    base-relative frame, evaluated BEFORE that joint's own rotation is
    applied (T_old @ Translate(origin_i)) -- i.e. the pivot point and axis
    orientation gravity actually acts about."""
    axes = tilted_axes(tilt_all)
    T = np.eye(4)
    p_list, a_list = [], []
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX:
            origin = origin + o_wp
        if i == ELBOW_PITCH_IDX:
            origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        p_i = T[:3, :3] @ origin + T[:3, 3]
        a_i = T[:3, :3] @ axes[i]
        p_list.append(p_i)
        a_list.append(a_i)
        joint_rotation = Rotation.from_rotvec(axes[i] * joint_angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(joint_rotation, origin)
    return T, p_list, a_list


GRAVITY_DIR = np.array([0.0, 0.0, -1.0])


def gravity_geometric_terms(corrected_angles, tilt_all, o_elbow, o_sy_xz, o_wp):
    T_fk, p_list, a_list = fk_with_joint_frames(corrected_angles, tilt_all, o_elbow, o_sy_xz, o_wp)
    p_ee = T_fk[:3, 3]
    geom = np.zeros(7)
    for i in range(7):
        lever = p_ee - p_list[i]
        geom[i] = np.dot(np.cross(lever, GRAVITY_DIR), a_list[i])
    return geom


N_TILT, N_SCALE = 14, 7
OFF_SCALE = 19
OFF_TILT = OFF_SCALE + N_SCALE
OFF_OELBOW = OFF_TILT + N_TILT
OFF_OSY = OFF_OELBOW + 3
OFF_OWP = OFF_OSY + 2
OFF_FOURIER = OFF_OWP + 3
OFF_COUPLE = OFF_FOURIER + 2
OFF_GRAVITY = OFF_COUPLE + N_COUPLE
TOTAL_PARAMS = OFF_GRAVITY + N_GRAVITY


def unpack(x):
    base_params = x[0:19]
    joint_scales = x[OFF_SCALE:OFF_SCALE + N_SCALE]
    tilt_all = x[OFF_TILT:OFF_TILT + N_TILT].reshape(7, 2)
    o_elbow = x[OFF_OELBOW:OFF_OELBOW + 3]
    o_sy_xz = x[OFF_OSY:OFF_OSY + 2]
    o_wp = x[OFF_OWP:OFF_OWP + 3]
    fourier_ab = x[OFF_FOURIER:OFF_FOURIER + 2]
    couple_c = x[OFF_COUPLE:OFF_COUPLE + N_COUPLE]
    gravity_c = x[OFF_GRAVITY:OFF_GRAVITY + N_GRAVITY]
    return base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c


def predict(angles_row, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c):
    params = ck.unpack_params(base_params)
    corrected_angles = (angles_row * joint_scales + params.joint_offsets).copy()
    raw_sp = angles_row[SHOULDER_PITCH_IDX]
    corrected_angles[SHOULDER_PITCH_IDX] += (
        fourier_ab[0] * np.sin(raw_sp) + fourier_ab[1] * np.cos(raw_sp)
    )
    for k, (i, j, tgt) in enumerate(COUPLE_TERMS):
        corrected_angles[tgt] += couple_c[k] * angles_row[i] * angles_row[j]

    geom = gravity_geometric_terms(corrected_angles, tilt_all, o_elbow, o_sy_xz, o_wp)
    corrected_angles_g = corrected_angles + gravity_c * geom

    T_fk = fk_combined(corrected_angles_g, tilt_all, o_elbow, o_sy_xz, o_wp)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


TILT_REG_WEIGHT, TILT_SCALE_RAD, SCALE_REG_WEIGHT = 5.0, np.radians(8.0), 20.0
FOURIER_REG_WEIGHT, FOURIER_SCALE_RAD = 5.0, np.radians(5.0)
COUPLE_REG_WEIGHT, COUPLE_SCALE = 5.0, 0.1
COUPLE_BOUND = 0.3
GRAVITY_REG_WEIGHT, GRAVITY_SCALE = 5.0, 0.1  # rad/m
GRAVITY_BOUND = 0.5  # rad/m


def make_residual(angles, pos_mm, quat_xyzw):
    n = len(angles)

    def residual(x):
        base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c = unpack(x)
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c)
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
            gravity_c * (GRAVITY_REG_WEIGHT / GRAVITY_SCALE),
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def per_pose_errors(angles, pos_mm, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c)
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
        -GRAVITY_BOUND * np.ones(N_GRAVITY),
    ])
    upper = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf]*3, [np.inf]*3, [np.inf]*3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
        np.radians(15.0) * np.ones(2),
        COUPLE_BOUND * np.ones(N_COUPLE),
        GRAVITY_BOUND * np.ones(N_GRAVITY),
    ])
    return lower, upper


def x0_zero():
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE + N_GRAVITY)
    ])


def fit(angles, pos_mm, quat_xyzw):
    residual = make_residual(angles, pos_mm, quat_xyzw)
    lower, upper = build_bounds()
    result = least_squares(residual, x0_zero(), method="trf", x_scale="jac",
                            bounds=(lower, upper), max_nfev=5000)
    return result


print(f"Model: {TOTAL_PARAMS} params (38 base + 3 confirmed coupling + {N_GRAVITY} lumped gravity-deflection terms)\n")

print("=== Multi-start (full 298-pose dataset) ===")
rng = np.random.default_rng(555)
rms_results = []
last_result = None
for trial in range(4):
    x0 = x0_zero()
    if trial > 0:
        lower, upper = build_bounds()
        x0 = np.clip(x0 + rng.uniform(-0.02, 0.02, size=TOTAL_PARAMS), lower, upper)
    lower, upper = build_bounds()
    result = least_squares(make_residual(angles_all, pos_mm_all, quat_xyzw_all), x0,
                            method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
    rms, _ = rms_of(angles_all, pos_mm_all, result.x)
    rms_results.append(rms)
    last_result = result
    print(f"  trial {trial}: RMS={rms:.3f}mm (status={result.status}, nfev={result.nfev})")

S = np.linalg.svd(last_result.jac, compute_uv=False)
cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
print(f"\nRMS range: {min(rms_results):.3f}-{max(rms_results):.3f}mm (reference, no gravity term: 6.874mm)")
print(f"Condition number: {cond:.1f} " + ("(healthy)" if cond < 1e4 else "(WARNING: high, check for new degeneracy)"))

base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c = unpack(last_result.x)
joint_names = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"]
print("\nFitted gravity-deflection coefficients (rad/m, bound +/-0.5):")
for name, c in zip(joint_names, gravity_c):
    print(f"  {name:<16}{c:+.4f}")

print("\nCoupling coefficients for comparison against the no-gravity model "
      "(+0.0018/+0.0219/-0.0165 rad^-1) -- large shifts would flag confound "
      "between coupling and gravity terms:")
for c in couple_c:
    print(f"  {c:+.4f}")

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
    print(f"  fold {k} (pose_id {pose_id_all[lo]}-{pose_id_all[hi-1]}): "
          f"train={train_rms:.2f}mm test={test_rms:.2f}mm gap={test_rms-train_rms:+.2f}mm")

pooled_test_errs = np.concatenate(pooled_test_errs)
pooled_rms = np.sqrt(np.mean(pooled_test_errs ** 2))
print(f"\nPooled blocked-CV test RMS WITH gravity term: {pooled_rms:.2f}mm (reference: 9.87mm)")
print(f"Fold 5 test RMS WITH gravity term: {fold5_test_rms:.2f}mm (reference: 17.08mm)")
if fold5_test_rms is not None and fold5_test_rms < 17.08 - 2.0:
    print("-> Real improvement in the fold-5 region. Evidence FOR gravity/elastostatic deflection.")
elif pooled_rms < 9.87 - 0.5:
    print("-> Modest aggregate improvement without a clear fold-5-specific win. Mixed evidence.")
else:
    print("-> No meaningful improvement. This lumped single-point-mass approximation "
          "doesn't explain fold 5 -- either the real per-link mass distribution matters "
          "(this approximation is too crude) or gravity/elastostatic isn't the right lead.")
