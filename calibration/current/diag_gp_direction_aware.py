"""Direction/history-aware GP residual layer (2026-07-30) -- extends the
validated plain-angle GP residual layer (diag_gp_residual_layer.py,
8.91mm -> 7.81mm pooled blocked-CV) with a feature the current model is
structurally blind to: which direction each joint arrived from.

Grounded specifically in this project's own documented history, not just
general hysteresis-modeling literature: quick_calibration_test_combined_
298.csv was collected BEFORE the unidirectional backlash-compensation fix
existed, so each pose's achieved configuration has a real, direction-
dependent backlash bias baked into the ground truth -- and the plain GP,
seeing only final joint angles, has no way to represent that. Literature
(neural-network gear-backlash compensation, Preisach+RNN configuration-
specific hysteresis modeling) supports history/direction as the key input
a naive position-only model is missing; kept to a lightweight per-joint
direction-SIGN feature (not a full previous-pose vector or RNN) given the
modest 298-pose sample size -- backlash is specifically a function of
approach DIRECTION, not the magnitude of whatever move preceded it, so
this targets the actual hypothesis without doubling the GP's input
dimensionality with mostly-redundant magnitude information.

Direction feature is computed ONCE from the TRUE physical collection order
(CSV row order, confirmed monotonic in pose_id = true capture sequence),
before any CV splitting -- so it always reflects the real predecessor pose
for a given row, never an artificial adjacency introduced by which fold
that row happens to land in. Thresholded at 5 ticks so ordinary settling
noise on a stationary joint doesn't get miscounted as a real direction
change.

Reference (plain 7D-angle-only GP on top of the same 60-param physical
model): pooled blocked-CV test RMS 7.81mm; fold 5 test RMS 14.54mm.
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
    rows = list(csv.DictReader(f))
pose_id_all = np.array([int(r["pose_id"]) for r in rows])
actual_ticks_all = np.array([[float(r[f"actual_tick_{i}"]) for i in range(7)] for r in rows])
assert np.all(np.diff(pose_id_all) >= 0), "row order must be true capture order"

# ---------------------------------------------------------------------------
# Direction-sign feature: computed ONCE from true capture order, before any
# CV splitting, so it always reflects the real physical predecessor.
# ---------------------------------------------------------------------------
TICK_THRESHOLD = 5.0
tick_deltas = np.zeros((n_all, 7))
tick_deltas[1:] = actual_ticks_all[1:] - actual_ticks_all[:-1]
direction_sign_all = np.zeros((n_all, 7))
direction_sign_all[tick_deltas > TICK_THRESHOLD] = 1.0
direction_sign_all[tick_deltas < -TICK_THRESHOLD] = -1.0
# pose 0 has no true predecessor -- neutral (0) by construction already.

n_increasing = np.sum(direction_sign_all == 1.0)
n_decreasing = np.sum(direction_sign_all == -1.0)
n_neutral = np.sum(direction_sign_all == 0.0)
print(f"Direction-sign feature over all {n_all}x7 joint-transitions: "
      f"{n_increasing} increasing, {n_decreasing} decreasing, {n_neutral} neutral/negligible\n")

# ---------------------------------------------------------------------------
# Current-best physical model: 38 base params + 3 coupling + 7 gravity
# (identical to diag_gp_residual_layer.py)
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
# Blocked CV with a direction-aware GP residual layer fit per-fold on TRAIN
# only -- input is [standardized angles (7), direction-sign (7)] = 14 dims.
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

print("=== Blocked CV: physical model + direction-aware GP residual layer ===\n")
for k in range(K):
    lo, hi = fold_bounds[k], fold_bounds[k + 1]
    test_mask = np.zeros(n_all, dtype=bool)
    test_mask[lo:hi] = True
    train_mask = ~test_mask

    angles_tr, pos_tr, quat_tr = angles_all[train_mask], pos_mm_all[train_mask], quat_xyzw_all[train_mask]
    angles_te, pos_te = angles_all[test_mask], pos_mm_all[test_mask]
    dir_tr, dir_te = direction_sign_all[train_mask], direction_sign_all[test_mask]

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

    # 14D input: standardized angles + direction-sign (already in {-1,0,1},
    # left unstandardized so the GP sees it as a distinct discrete feature).
    features_tr = np.concatenate([angles_tr_std, dir_tr], axis=1)
    features_te = np.concatenate([angles_te_std, dir_te], axis=1)

    gp_pred_te = np.zeros((len(angles_te), 3))
    for dim in range(3):
        kernel = GP_KERNEL_TEMPLATE(14)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True, alpha=1e-6, random_state=0)
        gp.fit(features_tr, residual_vec_tr[:, dim])
        gp_pred_te[:, dim] = gp.predict(features_te)

    corrected_pred_te = phys_pred_te + gp_pred_te
    gp_test_errs = np.linalg.norm(pos_te - corrected_pred_te, axis=1)
    gp_test_rms = np.sqrt(np.mean(gp_test_errs ** 2))
    pooled_test_errs_gp.append(gp_test_errs)

    if k == 5:
        fold5_gp_test_rms = gp_test_rms
        fold5_phys_test_rms = phys_test_rms

    print(f"  fold {k} (pose_id {pose_id_all[lo]}-{pose_id_all[hi-1]}, {hi-lo} held out): "
          f"physical-only test={phys_test_rms:.2f}mm  +direction-GP test={gp_test_rms:.2f}mm")

pooled_phys = np.sqrt(np.mean(np.concatenate(pooled_test_errs_phys) ** 2))
pooled_gp = np.sqrt(np.mean(np.concatenate(pooled_test_errs_gp) ** 2))

print(f"\nPooled blocked-CV test RMS, physical model only: {pooled_phys:.2f}mm")
print(f"Pooled blocked-CV test RMS, physical + direction-aware GP: {pooled_gp:.2f}mm")
print(f"Reference, physical + plain angle-only GP (no direction feature): 7.81mm")
print(f"\nFold 5 physical-only: {fold5_phys_test_rms:.2f}mm")
print(f"Fold 5 physical + direction-aware GP: {fold5_gp_test_rms:.2f}mm")
print(f"Reference, fold 5 with plain angle-only GP: 14.54mm")

if pooled_gp < 7.81 - 0.3:
    print("\n-> Real improvement over the plain angle-only GP -- the direction/"
          "history feature is adding genuine information the position-only "
          "model was missing.")
elif pooled_gp > 7.81 + 0.3:
    print("\n-> Worse than the plain angle-only GP -- likely the added "
          "dimensionality cost (14D vs 7D, same ~260-poses-per-fold training "
          "set) outweighed the direction feature's benefit here.")
else:
    print("\n-> No meaningful difference from the plain angle-only GP -- "
          "direction-dependent backlash is not a significant additional "
          "signal beyond what the physical model + angle-only GP already "
          "captured.")
