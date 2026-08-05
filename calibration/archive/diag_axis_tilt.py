import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)

# For each joint's nominal axis, build two unit vectors perpendicular to it,
# so a small tilt can be expressed as a 2D perturbation in that plane
# (rotating "around the axis itself" is meaningless/unidentifiable, so we
# deliberately only allow the 2 perpendicular degrees of freedom).
perp_vectors = []
for axis in ck.JOINT_AXES:
    axis = axis / np.linalg.norm(axis)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    perp_vectors.append((u, v))


def tilted_axes(tilt_params):
    # tilt_params: (7, 2) small angles (radians) per joint
    axes = np.zeros((ck.N_JOINTS, 3))
    for i in range(ck.N_JOINTS):
        axis = ck.JOINT_AXES[i]
        u, v = perp_vectors[i]
        perturbed = axis + tilt_params[i, 0] * u + tilt_params[i, 1] * v
        axes[i] = perturbed / np.linalg.norm(perturbed)
    return axes


def forward_kinematics_tilted(joint_angles_rad, axes):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        joint_rotation = Rotation.from_rotvec(axes[i] * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, ck.JOINT_ORIGINS_M[i])
        T = T @ T_joint
    return T


N_TILT_PARAMS = ck.N_JOINTS * 2  # 14
TILT_REG_WEIGHT = 15.0  # mm-equivalent penalty per radian of axis tilt -- small-perturbation prior
TILT_BOUND_GUESS_RAD = np.radians(5.0)  # just for regularization scaling, not a hard bound


def residual_with_tilt(x):
    base_params = x[:19]
    tilt_flat = x[19:]
    tilt_params = tilt_flat.reshape(ck.N_JOINTS, 2)
    axes = tilted_axes(tilt_params)

    params = ck.unpack_params(base_params)
    n_local = len(angles)
    pos_res = np.zeros((n_local, 3))
    orient_res = np.zeros((n_local, 3))

    for i in range(n_local):
        corrected_angles = angles[i] + params.joint_offsets
        T_fk = forward_kinematics_tilted(corrected_angles, axes)
        T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
        T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
        T_pred = T_base @ T_fk @ T_tool

        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3

    reg = np.concatenate([
        params.joint_offsets * (8.0 / np.radians(8.0)),
        params.tool_xyz * (1.0 / 0.01),
        params.tool_rpy * (1.0 / np.radians(10.0)),
        tilt_flat * (TILT_REG_WEIGHT / TILT_BOUND_GUESS_RAD),
    ])

    return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0_base = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
x0 = np.concatenate([x0_base, np.zeros(N_TILT_PARAMS)])

result = least_squares(residual_with_tilt, x0, method="lm", max_nfev=200000)

# Compute RMS using only the position residuals (first 3*n entries)
n_pos = 3 * n
pos_res_final = result.fun[:n_pos].reshape(n, 3)
errs = np.linalg.norm(pos_res_final, axis=1)
rms = np.sqrt(np.mean(errs ** 2))

tilt_params_fitted = result.x[19:].reshape(ck.N_JOINTS, 2)

print(f"Fit WITH per-joint axis-tilt correction (14 extra params):")
print(f"  Position RMS: {rms:.2f} mm (median {np.median(errs):.2f}, max {errs.max():.2f})")
print(f"\nFitted axis tilts (degrees, 2 components per joint -- perpendicular to nominal axis):")
for i in range(ck.N_JOINTS):
    tilt_deg = np.degrees(tilt_params_fitted[i])
    tilt_mag_deg = np.degrees(np.linalg.norm(tilt_params_fitted[i]))
    print(f"  {ck.JOINT_NAMES[i]:24s} tilt=({tilt_deg[0]:+6.2f}, {tilt_deg[1]:+6.2f}) deg, magnitude={tilt_mag_deg:.2f} deg")

print(f"\nFor reference, baseline fit (no axis tilt) RMS was 15.43 mm")
