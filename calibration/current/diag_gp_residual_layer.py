"""Gaussian Process residual layer on top of the current-best physical
model (2026-07-30) -- the literature-backed "second stage" (physical
calibration first, then a data-driven model of whatever residual is left,
fit on joint configuration -> position error). Per the discussion before
building this: GPs/NNs interpolate from nearby training examples, they
don't extrapolate any better than a physical model into a region with no
real local support -- and the trajectory-aware fold test already showed
fold 5's poses stay just as badly predicted regardless of what else
surrounds them in training, which is a warning sign, not a promising one,
for this approach fully fixing fold 5 specifically. Built and validated
anyway, honestly, with the same blocked-CV discipline as every other
addition this session -- NOT the full-dataset fit number, which cannot
reveal an overfit residual model (a flexible enough GP can always drive
in-sample residual near zero regardless of whether it generalizes).

Methodology, per fold (out of the same 8 contiguous-index folds used
throughout this session):
  1. Fit the current-best 60-param physical model (38 base + 3 coupling +
     7 gravity) on the training poses only.
  2. Compute the physical model's residual VECTOR (dx,dy,dz in mm, not
     just magnitude) on the training poses.
  3. Standardize training joint angles (mean/std from TRAIN only, no
     leakage from test).
  4. Fit one GP per output dimension (dx,dy,dz) -- RBF kernel with
     per-joint (ARD) length scales plus a WhiteKernel noise term (the
     regularizer that keeps this from just memorizing training noise),
     hyperparameters optimized via marginal likelihood with multiple
     restarts.
  5. Predict the GP correction for the held-out test poses' joint angles
     (standardized using TRAIN's own mean/std) and add it to the physical
     model's test predictions.
  6. Record test RMS after correction, AND the nearest-neighbor distance
     (in standardized 7D joint-angle space) from each test pose to its
     closest training pose -- a direct, quantitative check of the "fold 5
     lacks local support" hypothesis, not just a guess.

Reference (physical model only, no GP): full-dataset RMS 5.876mm; blocked-
CV pooled test RMS 8.91mm; fold 5 test RMS 15.86mm.
"""
import csv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

with open(CSV_PATH, newline="") as f:
    pose_id_all = np.array([int(row["pose_id"]) for row in csv.DictReader(f)])

# ---------------------------------------------------------------------------
# Current-best physical model: 38 base params + 3 coupling + 7 gravity
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


def predicted_positions_mm(angles, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c = unpack(x)
    out = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c)
        out[i] = T_pred[:3, 3] * 1000.0
    return out


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


def x0_zero(angles, pos_mm, quat_xyzw):
    base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
        np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE + N_GRAVITY)
    ])


def fit_physical(angles, pos_mm, quat_xyzw):
    residual = make_residual(angles, pos_mm, quat_xyzw)
    lower, upper = build_bounds()
    result = least_squares(residual, x0_zero(angles, pos_mm, quat_xyzw), method="trf",
                            x_scale="jac", bounds=(lower, upper), max_nfev=5000)
    return result


# ---------------------------------------------------------------------------
# Blocked CV with a GP residual layer fit per-fold on TRAIN only
# ---------------------------------------------------------------------------
GP_KERNEL_TEMPLATE = lambda n_dims: (
    ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3))
)

K = 8
fold_bounds = np.linspace(0, n_all, K + 1).astype(int)
pooled_test_errs_gp = []
pooled_test_errs_phys = []
fold5_gp_test_rms = None
fold5_phys_test_rms = None

print("=== Blocked CV: physical model + GP residual layer ===\n")
for k in range(K):
    lo, hi = fold_bounds[k], fold_bounds[k + 1]
    test_mask = np.zeros(n_all, dtype=bool)
    test_mask[lo:hi] = True
    train_mask = ~test_mask

    angles_tr, pos_tr, quat_tr = angles_all[train_mask], pos_mm_all[train_mask], quat_xyzw_all[train_mask]
    angles_te, pos_te = angles_all[test_mask], pos_mm_all[test_mask]

    phys_result = fit_physical(angles_tr, pos_tr, quat_tr)
    phys_pred_tr = predicted_positions_mm(angles_tr, phys_result.x)
    phys_pred_te = predicted_positions_mm(angles_te, phys_result.x)

    phys_test_errs = np.linalg.norm(pos_te - phys_pred_te, axis=1)
    phys_test_rms = np.sqrt(np.mean(phys_test_errs ** 2))
    pooled_test_errs_phys.append(phys_test_errs)

    residual_vec_tr = pos_tr - phys_pred_tr  # (n_train, 3) mm

    angle_mean = angles_tr.mean(axis=0)
    angle_std = angles_tr.std(axis=0)
    angle_std[angle_std < 1e-6] = 1.0
    angles_tr_std = (angles_tr - angle_mean) / angle_std
    angles_te_std = (angles_te - angle_mean) / angle_std

    gp_pred_te = np.zeros((len(angles_te), 3))
    for dim in range(3):
        kernel = GP_KERNEL_TEMPLATE(7)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True, alpha=1e-6, random_state=0)
        gp.fit(angles_tr_std, residual_vec_tr[:, dim])
        gp_pred_te[:, dim] = gp.predict(angles_te_std)

    corrected_pred_te = phys_pred_te + gp_pred_te
    gp_test_errs = np.linalg.norm(pos_te - corrected_pred_te, axis=1)
    gp_test_rms = np.sqrt(np.mean(gp_test_errs ** 2))
    pooled_test_errs_gp.append(gp_test_errs)

    # Nearest-neighbor distance in standardized joint-angle space: direct
    # check of whether this fold's test poses have real local support in
    # its training set, rather than assuming it.
    nn_dists = np.array([
        np.min(np.linalg.norm(angles_tr_std - row, axis=1)) for row in angles_te_std
    ])

    if k == 5:
        fold5_gp_test_rms = gp_test_rms
        fold5_phys_test_rms = phys_test_rms

    print(f"  fold {k} (pose_id {pose_id_all[lo]}-{pose_id_all[hi-1]}, {hi-lo} held out): "
          f"physical-only test={phys_test_rms:.2f}mm  +GP test={gp_test_rms:.2f}mm  "
          f"nn_dist(median/max)={np.median(nn_dists):.2f}/{np.max(nn_dists):.2f}")

pooled_phys = np.sqrt(np.mean(np.concatenate(pooled_test_errs_phys) ** 2))
pooled_gp = np.sqrt(np.mean(np.concatenate(pooled_test_errs_gp) ** 2))

print(f"\nPooled blocked-CV test RMS, physical model only: {pooled_phys:.2f}mm (reference: 8.91mm)")
print(f"Pooled blocked-CV test RMS, physical + GP residual: {pooled_gp:.2f}mm")
print(f"\nFold 5 physical-only: {fold5_phys_test_rms:.2f}mm (reference: 15.86mm)")
print(f"Fold 5 physical + GP: {fold5_gp_test_rms:.2f}mm")

if fold5_gp_test_rms is not None and fold5_gp_test_rms < fold5_phys_test_rms - 3.0:
    print("\n-> Real, meaningful improvement in fold 5 specifically from the GP layer.")
elif pooled_gp < pooled_phys - 0.5:
    print("\n-> Modest aggregate improvement without a clear fold-5-specific win -- "
          "consistent with the GP helping smooth/smaller residuals elsewhere but not "
          "having enough local support to fix the fold-5 region, as anticipated.")
else:
    print("\n-> No meaningful improvement, or worse. The GP has no real local support "
          "advantage over the physical model in the region that matters -- confirms "
          "this is a local-data-scarcity problem, not a model-flexibility problem.")
