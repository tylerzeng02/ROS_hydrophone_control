"""Check whether the fold-5 region's stubborn error is explained by
Jacobian-based error amplification rather than a missing correction term
(2026-07-30). A fixed-size joint-angle error doesn't produce a fixed-size
position error -- it's scaled by the FK Jacobian (roughly proportional to
how extended/far-reaching the current configuration is, the same lever-
arm physics already used for the gravity model). If fold 5's poses are
systematically more "amplifying" configurations, that would explain why
coupling, gravity, and the GP residual layer each helped partially without
ever closing the gap -- they remove different pieces of the underlying
joint-angle-level error, but whatever's left keeps getting scaled up by
geometry in that specific region regardless of which correction is used.

Computes, per pose: (1) simple "reach" (Euclidean distance from base
origin to the nominal FK end-effector position) as a cheap proxy, and (2)
the actual numerical FK Jacobian's largest singular value (operator norm --
the true worst-case amplification factor from a joint-angle perturbation
to a position error) via finite differences. Correlates both against the
current-best model's (38 base + 3 coupling + 7 gravity, no GP) per-pose
residual error, across the whole 298-pose dataset, and specifically checks
whether fold 5 (pose_id 191-227) stands out on either metric.
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
# Current-best physical model (38 base + 3 coupling + 7 gravity) -- reused
# to get per-pose residual errors to correlate against.
# ---------------------------------------------------------------------------
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
GRAVITY_REG_WEIGHT, GRAVITY_SCALE = 5.0, 0.1
GRAVITY_BOUND = 0.5


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


def x0_zero(angles, pos_mm, quat_xyzw):
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE + N_GRAVITY)
    ])


print("Fitting current-best model on full dataset to get per-pose residuals...")
residual_fn = make_residual(angles_all, pos_mm_all, quat_xyzw_all)
lower = np.concatenate([
    -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf]*3, [-np.inf]*3, [-np.inf]*3,
    0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT),
    -0.05 * np.ones(3), -0.05 * np.ones(2), -0.05 * np.ones(3),
    -np.radians(15.0) * np.ones(2), -COUPLE_BOUND * np.ones(N_COUPLE), -GRAVITY_BOUND * np.ones(N_GRAVITY),
])
upper = np.concatenate([
    np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf]*3, [np.inf]*3, [np.inf]*3,
    1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
    0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
    np.radians(15.0) * np.ones(2), COUPLE_BOUND * np.ones(N_COUPLE), GRAVITY_BOUND * np.ones(N_GRAVITY),
])
result = least_squares(residual_fn, x0_zero(angles_all, pos_mm_all, quat_xyzw_all),
                        method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
best_x = result.x

per_pose_err = np.zeros(n_all)
for i in range(n_all):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c = unpack(best_x)
    T_pred = predict(angles_all[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c)
    per_pose_err[i] = np.linalg.norm(pos_mm_all[i] - T_pred[:3, 3] * 1000.0)
full_rms = np.sqrt(np.mean(per_pose_err ** 2))
print(f"Confirmed full-dataset RMS: {full_rms:.3f}mm (reference: 5.876mm)\n")

# ---------------------------------------------------------------------------
# Reach and Jacobian norm, per pose, using the NOMINAL (uncorrected) FK --
# this is about the arm's own geometry/configuration, not the fitted
# correction, so nominal FK is the right thing to use here.
# ---------------------------------------------------------------------------
def nominal_fk_position(angles_row):
    T = ck.forward_kinematics(angles_row)
    T_tool = ck.build_tool_transform(np.zeros(3), np.zeros(3))
    return (T @ T_tool)[:3, 3]


def numerical_jacobian(angles_row, eps=1e-4):
    J = np.zeros((3, 7))
    p0 = nominal_fk_position(angles_row)
    for j in range(7):
        perturbed = angles_row.copy()
        perturbed[j] += eps
        p1 = nominal_fk_position(perturbed)
        J[:, j] = (p1 - p0) / eps
    return J


reach_all = np.zeros(n_all)
jac_norm_all = np.zeros(n_all)
for i in range(n_all):
    p = nominal_fk_position(angles_all[i])
    reach_all[i] = np.linalg.norm(p)  # meters, distance from base origin
    J = numerical_jacobian(angles_all[i])
    s = np.linalg.svd(J, compute_uv=False)
    jac_norm_all[i] = s[0]  # largest singular value = operator norm (m per rad)

reach_mm = reach_all * 1000.0

print("=== Correlation with per-pose residual error (whole dataset, n=298) ===")
corr_reach = np.corrcoef(reach_mm, per_pose_err)[0, 1]
corr_jac = np.corrcoef(jac_norm_all, per_pose_err)[0, 1]
print(f"reach (mm) vs error:              corr = {corr_reach:+.3f}")
print(f"Jacobian largest singular value vs error: corr = {corr_jac:+.3f}")

fold5_mask = (pose_id_all >= 191) & (pose_id_all <= 227)
print(f"\n=== Fold 5 (pose_id 191-227, n={fold5_mask.sum()}) vs rest (n={(~fold5_mask).sum()}) ===")
print(f"reach (mm):      fold5 mean={reach_mm[fold5_mask].mean():.1f}  rest mean={reach_mm[~fold5_mask].mean():.1f}")
print(f"Jacobian norm:   fold5 mean={jac_norm_all[fold5_mask].mean():.4f}  rest mean={jac_norm_all[~fold5_mask].mean():.4f}")
print(f"residual error:  fold5 mean={per_pose_err[fold5_mask].mean():.2f}mm  rest mean={per_pose_err[~fold5_mask].mean():.2f}mm")

print("\nPercentile comparison (reach, mm):")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: fold5={np.percentile(reach_mm[fold5_mask], p):.1f}  rest={np.percentile(reach_mm[~fold5_mask], p):.1f}")

if corr_reach > 0.3 or corr_jac > 0.3:
    print("\n-> Real positive correlation between reach/Jacobian-norm and "
          "residual error across the whole dataset -- supports the "
          "amplification hypothesis as a genuine, previously-unexamined "
          "contributor.")
else:
    print("\n-> Weak or no correlation -- the amplification hypothesis "
          "doesn't hold up against the data; fold 5's error isn't "
          "explained by this configuration simply being more extended.")
