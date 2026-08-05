"""Generate joint-angle poses for the displacement-from-home test
(2026-07-30): home position + a series of increasing Cartesian offsets
along one direction, to directly check whether position error grows
roughly proportionally with commanded displacement (the signature of a
residual joint gear-ratio SCALE error, distinct from the already-ruled-out
Jacobian/manipulability amplification and travel-distance hypotheses).

Uses the same numerical-FK-inversion approach already used for the
earlier IK round-trip validation poses (TARGET_POSES 316-318 in
test_five_pose_ndi_capture.cpp) rather than a live IK solver, since
external/trac_ik is an empty/uninitialized directory in this checkout
(see CLAUDE.md) and was never available for these tests either.

Uses the current-best fitted model (38 base + 3 coupling + 7 gravity) as
the FK ground truth for inversion, so the computed joint angles represent
the model's best available estimate of what will actually put the real
arm at each intended Cartesian target -- consistent with how 316-318 were
generated.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

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


print("Fitting current-best model on full dataset (used as FK ground truth for IK)...")
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
model_result = least_squares(residual_fn, x0_zero(angles_all, pos_mm_all, quat_xyzw_all),
                              method="trf", x_scale="jac", bounds=(lower, upper), max_nfev=5000)
best_x = model_result.x
base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c = unpack(best_x)

def fk_predict_mm(angles_row):
    T_pred = predict(angles_row, base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp, fourier_ab, couple_c, gravity_c)
    return T_pred[:3, 3] * 1000.0

# ---------------------------------------------------------------------------
# HOME: a comfortable, safe, roughly-centered configuration (NOT all-zero --
# all-zero is often near a joint-limit-adjacent or awkward posture; use the
# dataset's own mean pose instead, which is guaranteed reachable and mid-
# range, matching what a real "home" position would look like in practice).
# ---------------------------------------------------------------------------
home_angles = angles_all.mean(axis=0)
home_pos_mm = fk_predict_mm(home_angles)
print(f"\nHome position (mm): {home_pos_mm}")
print(f"Home joint angles (rad): {home_angles}")

# Direction: +Y ("right"), matching the large Y-range already seen in this
# arm's workspace (prior IK targets spanned Y from -556mm to +265mm).
DIRECTION = np.array([0.0, 1.0, 0.0])
DISPLACEMENTS_MM = [5, 10, 15, 20, 25, 30]

JOINT_LOWER_RAD = np.array([
    (ck.JOINT_TICK_RANGES[i][0] - ck.NOMINAL_ZERO_TICKS[i]) / (4096.0 / (2.0 * np.pi))
    for i in range(7)
])
JOINT_UPPER_RAD = np.array([
    (ck.JOINT_TICK_RANGES[i][1] - ck.NOMINAL_ZERO_TICKS[i]) / (4096.0 / (2.0 * np.pi))
    for i in range(7)
])


def solve_ik(target_mm, x0_angles):
    # 7-DOF arm, only 3 position constraints -- underdetermined/redundant,
    # which method='lm' (MINPACK-based) cannot handle (needs residuals >=
    # variables). 'trf' has no such restriction and supports bounds, so it
    # stays within jointCalibrations' safe range and (combined with warm-
    # starting from the previous solved pose) picks a continuous, physically
    # sensible solution branch rather than an arbitrary redundant one.
    def resid(angles):
        pos = fk_predict_mm(angles)
        return pos - target_mm
    x0_clipped = np.clip(x0_angles, JOINT_LOWER_RAD, JOINT_UPPER_RAD)
    result = least_squares(resid, x0_clipped, method="trf",
                            bounds=(JOINT_LOWER_RAD, JOINT_UPPER_RAD), max_nfev=2000)
    return result.x, np.linalg.norm(resid(result.x))

TICKS_PER_RADIAN = 4096.0 / (2.0 * np.pi)

def angles_to_ticks(angles_row):
    ticks = []
    for i in range(7):
        joint = ck.JOINT_TICK_RANGES[i]
        zero_tick = ck.NOMINAL_ZERO_TICKS[i]
        tick = int(round(zero_tick + angles_row[i] * TICKS_PER_RADIAN))
        ticks.append(tick)
        if tick < joint[0] or tick > joint[1]:
            print(f"  WARNING: joint {i} tick {tick} outside calibrated range {joint}")
    return ticks

print(f"\n{'displacement_mm':>16}{'ik_residual_mm':>16}   ticks")
poses_out = []
home_ticks = angles_to_ticks(home_angles)
poses_out.append((0, home_ticks))
print(f"{0:16d}{0.0:16.4f}   {home_ticks}")

x0 = home_angles.copy()
for d in DISPLACEMENTS_MM:
    target = home_pos_mm + d * DIRECTION
    solved_angles, ik_resid = solve_ik(target, x0)
    ticks = angles_to_ticks(solved_angles)
    poses_out.append((d, ticks))
    print(f"{d:16d}{ik_resid:16.4f}   {ticks}")
    x0 = solved_angles  # warm-start next solve from this one

print("\n// Paste into TARGET_POSES:")
for d, ticks in poses_out:
    print(f"    {{{{{', '.join(str(t) for t in ticks)}}}}},  // home + {d}mm along +Y")
