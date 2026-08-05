"""Verify specific, checkable claims from external feedback (2026-07-31)
before acting on any of it. Checks, in order:
  1. The bimodal in-sample claim: full=6.87mm, bulk(non-191-227)=6.11mm,
     bad region(191-227)=10.84mm, using the SAME 53-param coupling-only
     model (38 base + 3 coupling, no gravity) they claim to have used.
  2. The "not single-joint sweeps" claim: median 4 joints move per
     transition, 65% of transitions move >=4 joints -- this directly
     contradicts this project's own repeated characterization of the
     dataset (including this very repo's diag script docstrings), so it
     needs real verification, not just trust.
  3. New lead #1: does per-pose residual correlate with moving-fixed
     marker separation distance (the claimed orientation-amplification
     metrology effect)?
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
actual_ticks_all = np.array([[float(r[f"actual_tick_{i}"]) for i in range(7)] for r in rows])
moving_xyz = np.array([[float(r["moving_camera_tx_mm"]), float(r["moving_camera_ty_mm"]), float(r["moving_camera_tz_mm"])] for r in rows])
fixed_xyz = np.array([[float(r["fixed_camera_tx_mm"]), float(r["fixed_camera_ty_mm"]), float(r["fixed_camera_tz_mm"])] for r in rows])
separation_mm = np.linalg.norm(moving_xyz - fixed_xyz, axis=1)

print(f"Marker separation (mm) percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: {np.percentile(separation_mm, p):.1f}")
print(f"  mean: {separation_mm.mean():.1f}, max: {separation_mm.max():.1f}\n")

# ---------------------------------------------------------------------------
# Claim 2 check FIRST (cheap, no fitting needed): joints moved per transition
# ---------------------------------------------------------------------------
TICK_THRESHOLD = 10.0  # ticks -- matches the threshold already used elsewhere
deltas = np.abs(np.diff(actual_ticks_all, axis=0))
n_joints_moved = (deltas > TICK_THRESHOLD).sum(axis=1)
print("=== Claim: 'not single-joint sweeps' (median 4 joints/transition, 65% >=4) ===")
print(f"Joints moved per transition (>{TICK_THRESHOLD} ticks), n={len(n_joints_moved)} transitions:")
print(f"  median: {np.median(n_joints_moved):.1f}")
print(f"  mean: {n_joints_moved.mean():.2f}")
frac_ge4 = (n_joints_moved >= 4).mean()
print(f"  fraction with >=4 joints moving: {frac_ge4*100:.1f}%")
for k in range(8):
    frac = (n_joints_moved == k).mean()
    print(f"    exactly {k} joints: {frac*100:5.1f}%")

# ---------------------------------------------------------------------------
# Claim 1 check: reproduce the 53-param (coupling, NO gravity) in-sample
# bimodal split
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


def unpack(x):
    base_params = x[:19]
    joint_scales = x[19:19 + N_SCALE]
    tilt_all = x[19 + N_SCALE:19 + N_SCALE + N_TILT].reshape(7, 2)
    o_elbow = x[19 + N_SCALE + N_TILT:22 + N_SCALE + N_TILT]
    o_sy_xz = x[22 + N_SCALE + N_TILT:24 + N_SCALE + N_TILT]
    o_wp = x[24 + N_SCALE + N_TILT:27 + N_SCALE + N_TILT]
    fourier_ab = x[27 + N_SCALE + N_TILT:29 + N_SCALE + N_TILT]
    couple_c = x[29 + N_SCALE + N_TILT:29 + N_SCALE + N_TILT + N_COUPLE]
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
COUPLE_REG_WEIGHT, COUPLE_SCALE, COUPLE_BOUND = 5.0, 0.1, 0.3


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


def build_bounds():
    lower = np.concatenate([
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf]*3, [-np.inf]*3, [-np.inf]*3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT),
        -0.05 * np.ones(3), -0.05 * np.ones(2), -0.05 * np.ones(3),
        -np.radians(15.0) * np.ones(2), -COUPLE_BOUND * np.ones(N_COUPLE),
    ])
    upper = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf]*3, [np.inf]*3, [np.inf]*3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
        np.radians(15.0) * np.ones(2), COUPLE_BOUND * np.ones(N_COUPLE),
    ])
    return lower, upper


base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE)
])
lower, upper = build_bounds()
print("\nFitting 53-param coupling-only model (no gravity) on full dataset...")
result = least_squares(make_residual(angles_all, pos_mm_all, quat_xyzw_all), x0,
                        method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)

errs = np.zeros(n_all)
for i in range(n_all):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c = unpack(result.x)
    T_pred = predict(angles_all[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c)
    errs[i] = np.linalg.norm(pos_mm_all[i] - T_pred[:3, 3] * 1000.0)

bad_mask = (pose_id_all >= 191) & (pose_id_all <= 227)
rms_full = np.sqrt(np.mean(errs**2))
rms_bulk = np.sqrt(np.mean(errs[~bad_mask]**2))
rms_bad = np.sqrt(np.mean(errs[bad_mask]**2))
print(f"\n=== Claim: bimodal in-sample split ===")
print(f"Full dataset RMS: {rms_full:.2f}mm (claimed: 6.87mm)")
print(f"Bulk (excl. 191-227) RMS: {rms_bulk:.2f}mm (claimed: 6.11mm)")
print(f"Bad region (191-227) RMS: {rms_bad:.2f}mm (claimed: 10.84mm)")

# ---------------------------------------------------------------------------
# New lead #1 check: residual vs marker separation correlation
# ---------------------------------------------------------------------------
print(f"\n=== New lead #1: residual vs marker separation ===")
corr_sep = np.corrcoef(separation_mm, errs)[0, 1]
print(f"corr(separation_mm, residual_error) = {corr_sep:+.3f}  (n={n_all})")
print(f"Bulk mean separation: {separation_mm[~bad_mask].mean():.1f}mm, bad-region mean: {separation_mm[bad_mask].mean():.1f}mm")

print("\nError vs separation bin:")
bins = [0, 1600, 1800, 2000, 2200, 100000]
for i in range(len(bins) - 1):
    mask = (separation_mm >= bins[i]) & (separation_mm < bins[i+1])
    if mask.sum() > 0:
        print(f"  [{bins[i]:5d}-{bins[i+1]:5d}) mm: n={mask.sum():3d}  mean_err={errs[mask].mean():6.2f}mm  median={np.median(errs[mask]):6.2f}mm")
