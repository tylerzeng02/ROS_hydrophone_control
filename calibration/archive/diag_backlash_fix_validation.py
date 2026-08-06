"""Direct, apples-to-apples validation of the backlash-compensation fix in
dynamixel_motor.cpp: the SAME 49 poses (by pose_id), fit with the SAME
36-param model (19 base + 7 axis tilts + 3 origin corrections + 7 joint
gear-ratio scales) and the SAME trf/x_scale='jac'/bounds methodology --
the only difference is whether the poses were collected with the OLD
(uncompensated) or NEW (backlash-compensated) movement code. If the fix
is working, the NEW dataset should give a meaningfully lower RMS.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

OLD_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
NEW_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_backlashfix_51pose.csv"
NEW_CSV_V2 = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_backlashfix_v2_20tick.csv"
NEW_CSV_MOTOR3_ONLY = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_motor3only.csv"

POSE_IDS = [1,7,13,19,25,31,37,43,49,55,61,67,73,79,85,91,97,103,109,115,121,
            127,133,145,151,157,163,169,175,181,187,193,205,211,217,223,229,
            235,241,247,253,259,265,271,277,283,289,295,301]


def load_subset_by_pose_id(csv_path, pose_ids):
    import csv as csv_mod
    angles, pos_mm, quat = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        rows = {int(row["pose_id"]): row for row in reader}
    for pid in pose_ids:
        row = rows[pid]
        a = np.array([float(row[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)])
        p = np.array([float(row["moving_relative_fixed_tx_mm"]),
                      float(row["moving_relative_fixed_ty_mm"]),
                      float(row["moving_relative_fixed_tz_mm"])])
        q0 = float(row["moving_relative_fixed_q0"])
        qx = float(row["moving_relative_fixed_qx"])
        qy = float(row["moving_relative_fixed_qy"])
        qz = float(row["moving_relative_fixed_qz"])
        angles.append(a); pos_mm.append(p); quat.append([qx, qy, qz, q0])
    return np.array(angles), np.array(pos_mm), np.array(quat)


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


def rms_of(angles, pos_mm, x):
    base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp = unpack(x)
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


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


def fit_and_report(label, csv_path):
    angles, pos_mm, quat = load_subset_by_pose_id(csv_path, POSE_IDS)
    n = len(angles)
    print(f"=== {label}: {n} poses ===")

    base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat[0])
    residual_full = make_residual(angles, pos_mm, quat)

    rng = np.random.default_rng(2024)
    rms_results = []
    last_result = None
    for trial in range(5):
        if trial == 0:
            x0 = np.concatenate([
                ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
                np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3)
            ])
        else:
            joint_perturb = np.radians(rng.uniform(-10, 10, size=7))
            tool_xyz_perturb = rng.uniform(-0.03, 0.03, size=3)
            tool_rpy_perturb = np.radians(rng.uniform(-60, 60, size=3))
            base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
            base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
            scale_perturb = 1.0 + rng.uniform(-0.02, 0.02, size=N_SCALE)
            extra_perturb = rng.uniform(-0.02, 0.02, size=N_TILT + 3 + 2 + 3)
            x0 = np.concatenate([
                ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                                base_xyz_perturb, base_rpy_perturb),
                scale_perturb, extra_perturb
            ])
        lower, upper = build_bounds()
        result = least_squares(residual_full, x0, method="trf", x_scale="jac",
                                bounds=(lower, upper), max_nfev=5000)
        rms, _ = rms_of(angles, pos_mm, result.x)
        rms_results.append(rms)
        last_result = result
        print(f"  trial {trial}: RMS={rms:.3f} mm (status={result.status}, nfev={result.nfev})")

    S = np.linalg.svd(last_result.jac, compute_uv=False)
    cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
    print(f"  RMS range: {min(rms_results):.3f} - {max(rms_results):.3f} mm, condition number: {cond:.1f}\n")
    return min(rms_results)


rms_old = fit_and_report("OLD movement code (pre-backlash-fix)", OLD_CSV)
rms_new = fit_and_report("NEW movement code (40-tick overshoot, all joints)", NEW_CSV)
rms_new_v2 = fit_and_report("NEW movement code (20-tick overshoot, all joints)", NEW_CSV_V2)
rms_motor3_only = fit_and_report("NEW movement code (20-tick overshoot, MOTOR 3 ONLY)", NEW_CSV_MOTOR3_ONLY)

print("=== Summary ===")
print(f"Same {len(POSE_IDS)} poses, same model, same fitting method:")
print(f"  OLD (uncompensated) movement:              {rms_old:.2f} mm")
print(f"  NEW (40-tick, all joints) movement:         {rms_new:.2f} mm  ({rms_new - rms_old:+.2f} mm)")
print(f"  NEW (20-tick, all joints) movement:          {rms_new_v2:.2f} mm  ({rms_new_v2 - rms_old:+.2f} mm)")
print(f"  NEW (20-tick, MOTOR 3 ONLY) movement:        {rms_motor3_only:.2f} mm  ({rms_motor3_only - rms_old:+.2f} mm)")
