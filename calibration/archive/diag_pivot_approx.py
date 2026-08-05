import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)

WRIST_PITCH_IDX = 5
WRIST_ROLL_IDX = 6
ELBOW_PITCH_IDX = 3
SHOULDER_YAW_IDX = 2
axis0 = ck.JOINT_AXES[WRIST_PITCH_IDX] / np.linalg.norm(ck.JOINT_AXES[WRIST_PITCH_IDX])
arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
u0 = np.cross(axis0, arbitrary); u0 /= np.linalg.norm(u0)
v0 = np.cross(axis0, u0)
def fk_combined(joint_angles_rad, tilt_uv, o_elbow, o_sy, o_wp):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX:
            perturbed = a + tilt_uv[0] * u0 + tilt_uv[1] * v0
            a = perturbed / np.linalg.norm(perturbed)
            origin = origin + o_wp
        if i == ELBOW_PITCH_IDX:
            origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + o_sy
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


def predict(angles_row, base_params, tilt_uv, o_elbow, o_sy, o_wp):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, tilt_uv, o_elbow, o_sy, o_wp)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def unpack(x):
    return x[:19], x[19:21], x[21:24], x[24:27], x[27:30]


def residual(x):
    base_params, tilt_uv, o_elbow, o_sy, o_wp = unpack(x)
    params = ck.unpack_params(base_params)
    pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict(angles[i], base_params, tilt_uv, o_elbow, o_sy, o_wp)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        orient_res[i] = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec() * 100.0 * 0.3
    reg = np.concatenate([
        params.joint_offsets * (8.0 / np.radians(8.0)),
        params.tool_xyz * (1.0 / 0.01),
        params.tool_rpy * (1.0 / np.radians(10.0)),
    ])
    return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3)
])
result = least_squares(residual, x0, method="lm", max_nfev=200000)

base_params, tilt_uv, o_elbow, o_sy, o_wp = unpack(result.x)
errs = np.zeros(n)
for i in range(n):
    T_pred = predict(angles[i], base_params, tilt_uv, o_elbow, o_sy, o_wp)
    errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
rms = np.sqrt(np.mean(errs ** 2))
print(f"Current best model RMS (jointly re-optimized): {rms:.2f} mm\n")

# Non-wrist configuration vector (joints 0-4 only)
non_wrist = angles[:, :5]

# For each pose, find its nearest neighbor by non-wrist distance only
nn_idx = np.zeros(n, dtype=int)
nn_non_wrist_dist = np.zeros(n)
nn_wrist_dist_deg = np.zeros(n)
for i in range(n):
    dists = np.linalg.norm(non_wrist - non_wrist[i], axis=1)
    dists[i] = np.inf
    j = np.argmin(dists)
    nn_idx[i] = j
    nn_non_wrist_dist[i] = dists[j]
    wrist_diff = angles[i, 5:7] - angles[j, 5:7]
    nn_wrist_dist_deg[i] = np.degrees(np.linalg.norm(wrist_diff))

print(f"Nearest-neighbor non-wrist distance (deg-equivalent, radians shown): "
      f"min={nn_non_wrist_dist.min():.3f} median={np.median(nn_non_wrist_dist):.3f} "
      f"max={nn_non_wrist_dist.max():.3f}\n")

# Keep only pairs where the non-wrist configuration is reasonably close
# (bottom 25% of nearest-neighbor non-wrist distances) -- these approximate
# "same shoulder/elbow config, different wrist" pairs.
threshold = np.percentile(nn_non_wrist_dist, 25)
mask = nn_non_wrist_dist <= threshold
print(f"Using {mask.sum()} poses whose nearest non-wrist-neighbor is within "
      f"{np.degrees(threshold):.1f} deg (deg-equivalent)\n")

wrist_diff_isolated = nn_wrist_dist_deg[mask]
err_isolated = errs[mask]
err_diff_isolated = np.abs(errs[mask] - errs[nn_idx[mask]])

corr_err = np.corrcoef(wrist_diff_isolated, err_isolated)[0, 1]
corr_errdiff = np.corrcoef(wrist_diff_isolated, err_diff_isolated)[0, 1]

print(f"Correlation of wrist-angle-difference with this pose's own error: {corr_err:.3f}")
print(f"Correlation of wrist-angle-difference with |error difference between the pair|: {corr_errdiff:.3f}")
print(f"\nWrist-angle-difference range in this subset: {wrist_diff_isolated.min():.1f} - {wrist_diff_isolated.max():.1f} deg")
