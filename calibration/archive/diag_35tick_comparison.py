"""Fit the new 35-tick backlash-margin dataset (build/quick_calibration_
test.csv, collected 2026-07-30) with the EXACT same 36-param model and
methodology used for the original old-vs-new backlash comparison in
CLAUDE.md, for a true apples-to-apples RMS number.

Reference (same 36-param model, same fitting method, different movement
code only):
  OLD (uncompensated) movement:        8.27mm
  NEW, 40-tick overshoot:               8.43mm
  NEW, 20-tick overshoot:               8.32mm
  NEW, 35-tick overshoot (this run):    ?
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)
print(f"Dataset: {n_all} poses\n")

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


def rms_of(angles, pos_mm, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
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


base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all)
N_EXTRA = N_SCALE + N_TILT + 3 + 2 + 3 + 2

rng = np.random.default_rng(555)
n_trials = 6
rms_results = []
last_result = None
for trial in range(n_trials):
    if trial == 0:
        x0 = np.concatenate([
            ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
            np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2)
        ])
    else:
        joint_perturb = np.radians(rng.uniform(-10, 10, size=7))
        tool_xyz_perturb = rng.uniform(-0.03, 0.03, size=3)
        tool_rpy_perturb = np.radians(rng.uniform(-60, 60, size=3))
        base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
        base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
        scale_perturb = 1.0 + rng.uniform(-0.02, 0.02, size=N_SCALE)
        extra_perturb = rng.uniform(-0.02, 0.02, size=N_TILT + 3 + 2 + 3 + 2)
        x0 = np.concatenate([
            ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                            base_xyz_perturb, base_rpy_perturb),
            scale_perturb, extra_perturb
        ])
    lower, upper = build_bounds()
    result = least_squares(residual_full, x0, method="trf", x_scale="jac",
                            bounds=(lower, upper), max_nfev=5000)
    rms, _ = rms_of(angles_all, pos_mm_all, result.x)
    rms_results.append(rms)
    last_result = result
    print(f"  trial {trial}: RMS={rms:.3f} mm (status={result.status}, nfev={result.nfev})")

print(f"\nRMS range: {min(rms_results):.3f} - {max(rms_results):.3f} mm")
print(f"\n35-tick result: {rms_results[0]:.3f}mm")
print(f"Reference -- OLD (uncompensated): 8.27mm")
print(f"Reference -- NEW, 20-tick: 8.32mm")
print(f"Reference -- NEW, 40-tick: 8.43mm")
