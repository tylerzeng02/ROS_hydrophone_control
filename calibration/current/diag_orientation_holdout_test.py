"""The honest test of orientation generalization (2026-07-30): every split
tried so far (random, blocked, trajectory-aware, and the wrist_roll check)
turned out to have training data that already spans the held-out set's own
range in whatever dimension mattered -- so none of them could actually
reveal an orientation-extrapolation failure even if one existed. This
builds the one split that doesn't have that problem: identify the poses
that are genuine orientation OUTLIERS (large median angular distance from
every other pose in the dataset, computed in diag_orientation_coverage_
check.py), then train on everything else and test ONLY on those outliers.
By construction, their close orientation neighbors mostly don't exist in
the rest of the dataset (that's what "outlier" means here), so this is a
real test of extrapolation to a novel orientation neighborhood, not
interpolation within one already seen.

Uses the current-best, already-confirmed model (38 base params + 3
coupling terms, no gravity term -- that's a separate open test).
Reference numbers: full-dataset RMS 6.874mm, blocked-CV pooled test RMS
9.87mm (fold 5 specifically: 17.08mm).
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

# ---------------------------------------------------------------------------
# Identify genuine orientation outliers
# ---------------------------------------------------------------------------
rots = Rotation.from_quat(quat_xyzw_all)
mats = rots.as_matrix()
median_dist_to_others = np.zeros(n_all)
for i in range(n_all):
    rel = mats[i].T @ mats
    traces = np.trace(rel, axis1=1, axis2=2)
    cos_angle = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angle))
    median_dist_to_others[i] = np.median(angles_deg)

THRESHOLD_DEG = 140.0
outlier_mask = median_dist_to_others > THRESHOLD_DEG
print(f"Orientation outliers (median distance to rest > {THRESHOLD_DEG} deg): "
      f"{outlier_mask.sum()} poses")
print(f"pose_ids: {sorted(pose_id_all[outlier_mask].tolist())}")
print(f"Non-outlier poses (training pool): {(~outlier_mask).sum()}\n")

# ---------------------------------------------------------------------------
# Current-best model (38 base + 3 confirmed coupling terms)
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


def x0_zero(angles, pos_mm, quat_xyzw):
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE)
    ])


train_mask = ~outlier_mask
test_mask = outlier_mask

residual = make_residual(angles_all[train_mask], pos_mm_all[train_mask], quat_xyzw_all[train_mask])
lower, upper = build_bounds()
result = least_squares(residual, x0_zero(angles_all[train_mask], pos_mm_all[train_mask], quat_xyzw_all[train_mask]),
                        method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)

train_rms, _ = rms_of(angles_all[train_mask], pos_mm_all[train_mask], result.x)
test_errs = per_pose_errors(angles_all[test_mask], pos_mm_all[test_mask], result.x)
test_rms = np.sqrt(np.mean(test_errs ** 2))

print(f"Trained on {train_mask.sum()} non-outlier poses, tested on "
      f"{test_mask.sum()} genuine orientation-outlier poses (never seen in training):")
print(f"  train RMS: {train_rms:.2f}mm")
print(f"  test (orientation-outlier) RMS: {test_rms:.2f}mm")
print(f"  gap: {test_rms - train_rms:+.2f}mm")
print(f"\nReference: blocked-CV pooled test RMS (ordinary poses, same model): 9.87mm")
print(f"Reference: fold 5 (multi-joint-simultaneous region): 17.08mm")
print("\nPer-outlier-pose error:")
outlier_pose_ids = pose_id_all[test_mask]
order = np.argsort(-test_errs)
for idx in order:
    print(f"  pose_id={outlier_pose_ids[idx]:3d}  error={test_errs[idx]:6.2f}mm")

if test_rms > 20.0:
    print("\n-> Large, genuine extrapolation failure to novel orientations. "
          "This IS a real, previously-invisible gap -- more diverse orientation "
          "data collection would likely help.")
elif test_rms > 9.87 + 3.0:
    print("\n-> Real but moderate degradation predicting novel orientations -- "
          "some gap exists, worth more data but not a dominant error source.")
else:
    print("\n-> No meaningful degradation on genuinely novel orientations -- "
          "the model generalizes fine here; orientation coverage is NOT a "
          "significant remaining error source.")
