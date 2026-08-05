"""Sets up a real IK-style validation test: pick target positions (in the
fixed-marker/NDI coordinate frame we've been measuring in all session),
numerically invert our fully-corrected model (offset+scale+tilt+origin+
shoulder_pitch Fourier+tool+base frame -- everything validated this
session) to find joint angles that should reach each target, then convert
to raw ticks ready to command on the real arm and check with NDI.

Targets are derived from NEW joint-angle configurations (not in the
298-pose training set) so they're guaranteed reachable, but the inverse
solve starts from a DIFFERENT initial guess than those "ground truth"
angles -- so it's a genuine search, not just trivially returning its
starting point.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)

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


def predict(measured_angles, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab):
    params = ck.unpack_params(base_params)
    corrected_angles = measured_angles * joint_scales + params.joint_offsets
    corrected_angles = corrected_angles.copy()
    raw_sp = measured_angles[SHOULDER_PITCH_IDX]
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
FOURIER_REG_WEIGHT, FOURIER_SCALE_RAD = 5.0, np.radians(5.0)


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


print("Fitting the full validated model (36 params + shoulder_pitch Fourier term)...")
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
residual_full = make_residual(angles_all, pos_mm_all, quat_xyzw_all)
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.ones(N_SCALE), np.zeros(N_TILT + 3 + 2 + 3 + 2)
])
lower, upper = build_bounds()
result = least_squares(residual_full, x0, method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
model_params = result.x
base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab = unpack(model_params)

errs = np.zeros(len(angles_all))
for i in range(len(angles_all)):
    T_pred = predict(angles_all[i], base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab)
    errs[i] = np.linalg.norm(pos_mm_all[i] - T_pred[:3, 3] * 1000.0)
print(f"Confirmed model RMS: {np.sqrt(np.mean(errs**2)):.3f} mm\n")


def model_predict_pos(joint_angles_rad):
    T = predict(joint_angles_rad, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab)
    return T[:3, 3] * 1000.0


def joint_bounds_rad():
    lo = np.zeros(7); hi = np.zeros(7)
    for j in range(7):
        tmin, tmax = ck.JOINT_TICK_RANGES[j]
        # Keep 150-tick margin off each hard limit for safety.
        lo[j] = (tmin + 150 - ck.NOMINAL_ZERO_TICKS[j]) / ck.TICKS_PER_RADIAN
        hi[j] = (tmax - 150 - ck.NOMINAL_ZERO_TICKS[j]) / ck.TICKS_PER_RADIAN
    return lo, hi


lo_rad, hi_rad = joint_bounds_rad()

# Three NEW joint configurations (not in the training set), scattered
# within each joint's safe range, used ONLY to generate guaranteed-
# reachable targets -- the inverse solve below does NOT get to see these.
rng = np.random.default_rng(20260729)
ground_truth_configs = []
for _ in range(3):
    cfg = lo_rad + rng.uniform(0.2, 0.8, size=7) * (hi_rad - lo_rad)
    ground_truth_configs.append(cfg)

targets_mm = [model_predict_pos(cfg) for cfg in ground_truth_configs]

print("=== Target positions (derived from new, held-out joint configs) ===")
for i, t in enumerate(targets_mm):
    print(f"  Target {i+1}: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm")

print("\n=== Numerically inverting the model to find joint angles for each target ===")
print("(starting from a DIFFERENT initial guess than the ground-truth config --")
print(" this is a genuine search, not trivially returning its starting point)\n")

solved_configs = []
for i, target in enumerate(targets_mm):
    def ik_residual(theta):
        pos = model_predict_pos(theta)
        return pos - target

    # Start from the center of the joint range -- deliberately NOT the
    # ground-truth config used to generate the target.
    x0_ik = (lo_rad + hi_rad) / 2.0
    ik_result = least_squares(ik_residual, x0_ik, bounds=(lo_rad, hi_rad), max_nfev=5000)
    solved_pos = model_predict_pos(ik_result.x)
    residual_mm = np.linalg.norm(solved_pos - target)
    solved_configs.append(ik_result.x)

    print(f"Target {i+1}: inverse-solve residual = {residual_mm:.4f} mm "
          f"(should be ~0 -- confirms the numerical inverse converged correctly)")
    print(f"  solved angles (deg): {np.round(np.degrees(ik_result.x), 2)}")

print("\n=== Converting solved angles to raw ticks for the real robot ===")
for i, cfg in enumerate(solved_configs):
    ticks = []
    for j in range(7):
        tick = ck.NOMINAL_ZERO_TICKS[j] + cfg[j] * ck.TICKS_PER_RADIAN
        ticks.append(int(round(tick)))
        tmin, tmax = ck.JOINT_TICK_RANGES[j]
        if not (tmin <= ticks[-1] <= tmax):
            print(f"  WARNING: target {i+1} motor {j} tick {ticks[-1]} outside safe range [{tmin},{tmax}]!")
    print(f"Target {i+1} raw ticks: {{{', '.join(str(t) for t in ticks)}}}")
    print(f"  (intended marker position: [{targets_mm[i][0]:.2f}, {targets_mm[i][1]:.2f}, {targets_mm[i][2]:.2f}] mm)")
