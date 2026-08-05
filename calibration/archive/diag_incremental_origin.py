import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)

# Start from the wrist_pitch tilt already validated -- keep it fixed in the
# model while testing each joint's origin-delta individually on top, so we
# build the model incrementally rather than throwing everything in at once.
WRIST_PITCH_IDX = 5
axis0 = ck.JOINT_AXES[WRIST_PITCH_IDX] / np.linalg.norm(ck.JOINT_AXES[WRIST_PITCH_IDX])
arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
u0 = np.cross(axis0, arbitrary); u0 /= np.linalg.norm(u0)
v0 = np.cross(axis0, u0)
WRIST_PITCH_TILT_UV = np.radians(np.array([-0.63, -4.55]))  # from the validated fit


def fk_with_base_tilt_and_origin(joint_angles_rad, target_joint, origin_delta):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        if i == WRIST_PITCH_IDX:
            perturbed = a + WRIST_PITCH_TILT_UV[0] * u0 + WRIST_PITCH_TILT_UV[1] * v0
            a = perturbed / np.linalg.norm(perturbed)
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == target_joint:
            origin = origin + origin_delta
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


def make_residual(target_joint):
    def residual(x):
        base_params = x[:19]
        origin_delta = x[19:22]
        params = ck.unpack_params(base_params)
        pos_res = np.zeros((n, 3))
        orient_res = np.zeros((n, 3))
        for i in range(n):
            corrected_angles = angles[i] + params.joint_offsets
            T_fk = fk_with_base_tilt_and_origin(corrected_angles, target_joint, origin_delta)
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

print("Baseline (wrist_pitch tilt only, no origin delta): RMS = 13.99 mm\n")
print(f"{'joint':24s} {'RMS(mm)':>9s} {'origin delta(mm)':>28s} {'min singular val':>18s} {'condition #':>14s}")

for j in range(ck.N_JOINTS):
    x0 = np.concatenate([x0_base, np.zeros(3)])
    result = least_squares(make_residual(j), x0, method="lm", max_nfev=100000)

    n_pos = 3 * n
    pos_res_final = result.fun[:n_pos].reshape(n, 3)
    errs = np.linalg.norm(pos_res_final, axis=1)
    rms = np.sqrt(np.mean(errs ** 2))

    origin_delta_mm = result.x[19:22] * 1000.0

    S = np.linalg.svd(result.jac, compute_uv=False)
    min_sv = S[-1]
    cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")

    flag = "  <-- well-identified" if min_sv > 1.0 else "  <-- DEGENERATE"
    print(f"{ck.JOINT_NAMES[j]:24s} {rms:9.2f} {str(np.round(origin_delta_mm,1)):>28s} {min_sv:18.6f} {cond:14.1f}{flag}")
