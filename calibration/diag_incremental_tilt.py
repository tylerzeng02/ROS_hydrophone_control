import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)

perp_vectors = []
for axis in ck.JOINT_AXES:
    axis = axis / np.linalg.norm(axis)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    perp_vectors.append((u, v))


def fk_with_one_tilt(joint_angles_rad, target_joint, tilt_uv):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        axis = ck.JOINT_AXES[i]
        if i == target_joint:
            u, v = perp_vectors[i]
            perturbed = axis + tilt_uv[0] * u + tilt_uv[1] * v
            axis = perturbed / np.linalg.norm(perturbed)
        joint_rotation = Rotation.from_rotvec(axis * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, ck.JOINT_ORIGINS_M[i])
        T = T @ T_joint
    return T


def make_residual(target_joint):
    def residual(x):
        base_params = x[:19]
        tilt_uv = x[19:21]
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3))
        orient_res = np.zeros((n, 3))
        for i in range(n):
            corrected_angles = angles[i] + params.joint_offsets
            T_fk = fk_with_one_tilt(corrected_angles, target_joint, tilt_uv)
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
        ])
        return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])
    return residual


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0_base = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)

print(f"Baseline (19-param, no tilt): RMS = 15.43 mm\n")
print(f"{'joint':24s} {'RMS(mm)':>9s} {'tilt(deg)':>10s} {'min singular val':>18s} {'condition #':>14s}")

for j in range(ck.N_JOINTS):
    x0 = np.concatenate([x0_base, np.zeros(2)])
    result = least_squares(make_residual(j), x0, method="lm", max_nfev=100000)

    n_pos = 3 * n
    pos_res_final = result.fun[:n_pos].reshape(n, 3)
    errs = np.linalg.norm(pos_res_final, axis=1)
    rms = np.sqrt(np.mean(errs ** 2))

    tilt_uv = result.x[19:21]
    tilt_mag_deg = np.degrees(np.linalg.norm(tilt_uv))

    S = np.linalg.svd(result.jac, compute_uv=False)
    min_sv = S[-1]
    cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")

    flag = "  <-- well-identified" if min_sv > 1.0 else "  <-- STILL DEGENERATE"
    print(f"{ck.JOINT_NAMES[j]:24s} {rms:9.2f} {tilt_mag_deg:10.2f} {min_sv:18.6f} {cond:14.1f}{flag}")
