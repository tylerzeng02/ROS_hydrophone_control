"""Refresh of the stale diag_joint_coupling.py: checks correlation between
per-pose residual error (from the CURRENT best 36-param model: 19 base + 7
tilts + 3 origins + 7 joint scales, 8.257mm) and pairwise joint-angle
products, on the full current 298-pose dataset. A real coupling model
(full pairwise cross-term matrix) is a much larger parameter addition than
scale was (up to 21 new terms) -- only worth the expensive parametric fit
if this cheap correlation check flags a specific suspicious pair first.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

ELBOW_PITCH_IDX = 3
SHOULDER_YAW_IDX = 2
WRIST_PITCH_IDX = 5

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


N_TILT = 14
N_SCALE = 7
N_EXTRA = N_SCALE + N_TILT + 3 + 2 + 3


def unpack(x):
    base_params = x[:19]
    joint_scales = x[19:19 + N_SCALE]
    tilt_all = x[19 + N_SCALE:19 + N_SCALE + N_TILT].reshape(7, 2)
    o_elbow = x[19 + N_SCALE + N_TILT:22 + N_SCALE + N_TILT]
    o_sy_xz = x[22 + N_SCALE + N_TILT:24 + N_SCALE + N_TILT]
    o_wp = x[24 + N_SCALE + N_TILT:27 + N_SCALE + N_TILT]
    return base_params, joint_scales, tilt_all, o_elbow, o_sy_xz, o_wp


TILT_REG_WEIGHT = 5.0
TILT_SCALE_RAD = np.radians(8.0)
SCALE_REG_WEIGHT = 20.0


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


print("Refitting current best model (36 params: base+tilt+origin+scale) to get residuals...")
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
rms = np.sqrt(np.mean(errs**2))
print(f"Confirmed RMS: {rms:.3f} mm (status={result.status}, nfev={result.nfev})\n")

print("Correlation of per-pose residual error against pairwise joint-angle PRODUCTS")
print("(checks for coupling/cross-talk no per-joint-independent parameter could capture):\n")

results = []
for i in range(ck.N_JOINTS):
    for j in range(i + 1, ck.N_JOINTS):
        product = angles_all[:, i] * angles_all[:, j]
        corr = np.corrcoef(product, errs)[0, 1]
        results.append((abs(corr), corr, ck.JOINT_NAMES[i], ck.JOINT_NAMES[j]))

results.sort(reverse=True)
print(f"{'joint pair':50s} {'correlation':>12s}")
for abscorr, corr, name_i, name_j in results:
    print(f"{name_i + ' * ' + name_j:50s} {corr:12.3f}")

print("\nFor reference, correlation of residual error against individual joint angles:")
for i in range(ck.N_JOINTS):
    corr = np.corrcoef(angles_all[:, i], errs)[0, 1]
    print(f"  {ck.JOINT_NAMES[i]:24s} {corr:8.3f}")
