"""Round 2 of joint-coupling screening (2026-07-30): now that
diag_joint_coupling_model_fit.py validated 3 coupling terms as real (RMS
8.187->6.874mm, fold5 20.48->17.08mm), fold 5 is STILL the worst fold by a
wide margin. Check whether a 4th coupling pair is hiding in what's left.

Methodology fix vs. the earlier "ambiguous" round (diag_joint_coupling_
refresh.py / diag_partial_coupling_check.py): those screened residual-vs-
product correlation across the WHOLE 298-pose dataset, which is dominated
by single-joint sweeps that carry zero coupling signal by construction --
a real effect concentrated in one region can get diluted into "ambiguous"
by hundreds of irrelevant rows. This screens using:
  (a) residuals from the model that ALREADY has the 3 confirmed terms
      (so we're hunting for what's left, not re-finding those 3), and
  (b) correlation computed BOTH on the full dataset AND restricted to just
      the fold-5 region (pose_id 191-227) where multi-joint motion
      actually occurs -- a real 4th term should show up more clearly in
      (b) than in (a) if it does at all.
This is a screening pass only. Any candidate that stands out still needs
a real refit + blocked-CV check (this project's own prior lesson: pairwise
correlation is necessary but not sufficient -- the actual fit is the real
test), which is a follow-up script, not this one.
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

JOINT_NAMES = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow_pitch",
               "elbow_yaw", "wrist_pitch", "wrist_roll"]
SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6

EXISTING_PAIRS = {
    frozenset((SHOULDER_ROLL_IDX, SHOULDER_YAW_IDX)),
    frozenset((SHOULDER_YAW_IDX, ELBOW_YAW_IDX)),
    frozenset((SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX)),
}

# --- Reproduce the validated 3-coupling-term model exactly (duplicated from
# diag_joint_coupling_model_fit.py per this project's diag-script convention) ---
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


base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE)
])
lower, upper = build_bounds()
print("Fitting the validated 3-coupling-term model (reference: RMS should land near 6.874mm)...")
result = least_squares(make_residual(angles_all, pos_mm_all, quat_xyzw_all), x0,
                        method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
errs = per_pose_errors(angles_all, pos_mm_all, result.x)
rms = np.sqrt(np.mean(errs ** 2))
print(f"Confirmed RMS: {rms:.3f}mm\n")

fold5_mask = (pose_id_all >= 191) & (pose_id_all <= 227)
print(f"Fold-5 region: {fold5_mask.sum()} poses (pose_id 191-227)\n")

# --- Screen all remaining pairs ---
pairs = []
for i in range(7):
    for j in range(i + 1, 7):
        if frozenset((i, j)) in EXISTING_PAIRS:
            continue
        pairs.append((i, j))

print(f"Screening {len(pairs)} remaining joint pairs (21 total - 3 already modeled)\n")
print(f"{'pair':<35}{'full-dataset corr':>20}{'fold-5-only corr':>20}")
rows = []
for i, j in pairs:
    prod_full = angles_all[:, i] * angles_all[:, j]
    corr_full = np.corrcoef(prod_full, errs)[0, 1]
    prod_f5 = angles_all[fold5_mask, i] * angles_all[fold5_mask, j]
    corr_f5 = np.corrcoef(prod_f5, errs[fold5_mask])[0, 1]
    rows.append((i, j, corr_full, corr_f5))

rows.sort(key=lambda r: -abs(r[3]))
for i, j, corr_full, corr_f5 in rows:
    name = f"{JOINT_NAMES[i]}*{JOINT_NAMES[j]}"
    print(f"{name:<35}{corr_full:>20.3f}{corr_f5:>20.3f}")

# --- Also check: does each joint's OWN angle (no product) correlate with
# fold-5 residuals, to flag the same "single-joint-dominates" confound that
# made the original screening ambiguous ---
print("\nSingle-joint-angle correlation with fold-5 residuals (confound check):")
single_rows = []
for i in range(7):
    corr = np.corrcoef(angles_all[fold5_mask, i], errs[fold5_mask])[0, 1]
    single_rows.append((i, corr))
single_rows.sort(key=lambda r: -abs(r[1]))
for i, corr in single_rows:
    print(f"  {JOINT_NAMES[i]:<20}{corr:>10.3f}")

print("\nCandidates with |fold-5 corr| > 0.3 AND not dominated by a single-joint "
      "confound above are worth a real refit test; print above and inspect "
      "before picking one automatically.")
