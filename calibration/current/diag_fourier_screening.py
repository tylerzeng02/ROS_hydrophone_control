"""Screening pass (idea #1 from literature research): published robot
calibration work shows joint-angle-DEPENDENT (Fourier-series) kinematic
error terms explain substantially more error than constant-offset models
alone (~97% vs ~79% in one study) -- gear-tooth-meshing effects that
repeat periodically as a joint rotates, distinct from the constant
offset/scale/tilt/origin corrections already in our model.

This screens ALL 7 joints (not just shoulder_yaw, which was checked
earlier and found inconclusive due to sparse data at extreme angles) for
a periodic residual-vs-own-angle pattern, using the current best model's
residuals on the full 298-pose dataset for maximum statistical power.
Cheap, uses existing data, no new hardware needed.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)
print(f"Dataset: {n_all} poses\n")

ELBOW_PITCH_IDX, SHOULDER_YAW_IDX, WRIST_PITCH_IDX = 3, 2, 5
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


def predict(angles_row, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row * joint_scales + params.joint_offsets
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
    return base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp


TILT_REG_WEIGHT, TILT_SCALE_RAD, SCALE_REG_WEIGHT = 5.0, np.radians(8.0), 20.0


def make_residual(angles, pos_mm, quat_xyzw):
    n = len(angles)

    def residual(x):
        base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp = unpack(x)
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
        for i in range(n):
            T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp)
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
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


def build_bounds():
    lower = np.concatenate([
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf]*3, [-np.inf]*3, [-np.inf]*3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT),
        -0.05 * np.ones(3), -0.05 * np.ones(2), -0.05 * np.ones(3),
    ])
    upper = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf]*3, [np.inf]*3, [np.inf]*3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
    ])
    return lower, upper


print("Fitting current best (36-param) model on full dataset...")
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all)
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3)
])
lower, upper = build_bounds()
result = least_squares(residual_full, x0, method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp = unpack(result.x)

errs = np.zeros(n_all)
for i in range(n_all):
    T_pred = predict(angles_all[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp)
    errs[i] = np.linalg.norm(pos_mm_all[i] - T_pred[:3, 3] * 1000.0)
print(f"Confirmed baseline RMS: {np.sqrt(np.mean(errs**2)):.3f} mm\n")


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


JOINT_NAMES_SHORT = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw",
                     "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"]

print("=== Per-joint Fourier screening: does a periodic term in THIS joint's")
print("own angle explain more residual error than a flat/linear baseline? ===\n")
print(f"{'joint':16s} {'range(deg)':>11s} {'R2 linear':>10s} {'R2 quad':>9s} "
      f"{'R2 1/rev':>9s} {'R2 2/rev':>9s} {'R2 3/rev':>9s}  best")

candidates = []
for j in range(7):
    ang_deg = np.degrees(angles_all[:, j])
    span = ang_deg.max() - ang_deg.min()

    A1 = np.vstack([ang_deg, np.ones_like(ang_deg)]).T
    c1, *_ = np.linalg.lstsq(A1, errs, rcond=None)
    r2_lin = r2(errs, A1 @ c1)

    A2 = np.vstack([ang_deg**2, ang_deg, np.ones_like(ang_deg)]).T
    c2, *_ = np.linalg.lstsq(A2, errs, rcond=None)
    r2_quad = r2(errs, A2 @ c2)

    rad = angles_all[:, j]
    r2_periods = []
    for k in (1, 2, 3):
        Ak = np.vstack([np.sin(k*rad), np.cos(k*rad), np.ones_like(rad)]).T
        ck_, *_ = np.linalg.lstsq(Ak, errs, rcond=None)
        r2_periods.append(r2(errs, Ak @ ck_))

    best_r2 = max([r2_lin, r2_quad] + r2_periods)
    best_label = ["linear", "quad", "1/rev", "2/rev", "3/rev"][
        np.argmax([r2_lin, r2_quad] + r2_periods)
    ]
    candidates.append((best_r2, j, best_label))

    print(f"{JOINT_NAMES_SHORT[j]:16s} {span:11.1f} {r2_lin:10.3f} {r2_quad:9.3f} "
          f"{r2_periods[0]:9.3f} {r2_periods[1]:9.3f} {r2_periods[2]:9.3f}  {best_label}")

candidates.sort(reverse=True)
print(f"\nStrongest candidate: {JOINT_NAMES_SHORT[candidates[0][1]]} "
      f"(R^2={candidates[0][0]:.3f}, best form={candidates[0][2]})")
print(f"Second strongest: {JOINT_NAMES_SHORT[candidates[1][1]]} "
      f"(R^2={candidates[1][0]:.3f}, best form={candidates[1][2]})")
