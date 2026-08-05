import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

# Standard minimal error model: a full 6-DOF rigid-body perturbation
# (3 rotation + 3 translation) inserted right after each joint's nominal
# transform -- this is the "local POE" / CPC structure from the
# calibration literature, proven complete + minimal for interior joints.
#
# Joint 0 (shoulder_roll): NO perturbation at all -- provably fully
# redundant with the base transform (any perturbation here is
# indistinguishable from changing base_xyz/base_rpy for every pose).
#
# Joint 6 (wrist_roll): perturbation rotation is restricted to the 2
# directions PERPENDICULAR to its own axis only -- rotation about its own
# axis is provably redundant with tool_rpy, the same way joint 0's full
# perturbation is redundant with the base transform. Translation (3 DOF)
# is kept in full, since (unlike joint 0) real upstream rotation from
# joints 0-5 varies pose-to-pose, making it genuinely observable -- this
# matches what we found empirically (wrist_roll's origin delta was
# well-conditioned, just not very useful).
#
# Joints 1-5 (middle joints): full 6-DOF perturbation each.

MIDDLE_JOINTS = [1, 2, 3, 4, 5]
LAST_JOINT = 6

perp_vectors = {}
for i in [LAST_JOINT]:
    axis = ck.JOINT_AXES[i] / np.linalg.norm(ck.JOINT_AXES[i])
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    perp_vectors[i] = (u, v)


def fk_minimal(joint_angles_rad, perturbations):
    """perturbations: dict joint_idx -> (R_perturb 3x3, t_perturb (3,))"""
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        joint_rotation = Rotation.from_rotvec(ck.JOINT_AXES[i] * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, ck.JOINT_ORIGINS_M[i])
        T = T @ T_joint
        if i in perturbations:
            R_p, t_p = perturbations[i]
            T = T @ ck.homogeneous_transform(R_p, t_p)
    return T


N_MIDDLE = len(MIDDLE_JOINTS) * 6   # 30
N_LAST = 2 + 3                      # 5 (2 perpendicular rotation + 3 translation)
N_TOTAL_EXTRA = N_MIDDLE + N_LAST   # 35


def unpack_perturbations(extra_params):
    perturbations = {}
    idx = 0
    for j in MIDDLE_JOINTS:
        rotvec = extra_params[idx:idx + 3]
        t = extra_params[idx + 3:idx + 6]
        R_p = Rotation.from_rotvec(rotvec).as_matrix()
        perturbations[j] = (R_p, t)
        idx += 6
    # last joint: 2 perpendicular rotation components + 3 translation
    u, v = perp_vectors[LAST_JOINT]
    rot_uv = extra_params[idx:idx + 2]
    t_last = extra_params[idx + 2:idx + 5]
    rotvec_last = rot_uv[0] * u + rot_uv[1] * v
    R_p_last = Rotation.from_rotvec(rotvec_last).as_matrix()
    perturbations[LAST_JOINT] = (R_p_last, t_last)
    idx += 5
    assert idx == N_TOTAL_EXTRA
    return perturbations


def predict(angles_row, base_params, extra_params):
    params = ck.unpack_params(base_params)
    perturbations = unpack_perturbations(extra_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_minimal(corrected_angles, perturbations)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def residual_no_reg(x, angles, pos_mm, quat_xyzw):
    base_params = x[:19]
    extra_params = x[19:19 + N_TOTAL_EXTRA]
    n = len(angles)
    pos_res = np.zeros((n, 3))
    orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict(angles[i], base_params, extra_params)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3
    return np.concatenate([pos_res.ravel(), orient_res.ravel()])


def rms_of(angles, pos_mm, x):
    base_params = x[:19]
    extra_params = x[19:19 + N_TOTAL_EXTRA]
    errs = np.zeros(len(angles))
    for i in range(len(angles)):
        T_pred = predict(angles[i], base_params, extra_params)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


print(f"Total parameters: 19 base + {N_TOTAL_EXTRA} perturbation = {19 + N_TOTAL_EXTRA}\n")

base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.zeros(N_TOTAL_EXTRA)
])

result = least_squares(lambda x: residual_no_reg(x, angles_all, pos_mm_all, quat_xyzw_all),
                        x0, method="lm", max_nfev=300000)
rms, errs = rms_of(angles_all, pos_mm_all, result.x)

S = np.linalg.svd(result.jac, compute_uv=False)
cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
n_near_zero = (S < 1.0).sum()

print(f"RMS (unregularized, all 190 poses): {rms:.2f} mm (median {np.median(errs):.2f}, max {errs.max():.2f})")
print(f"Condition number: {cond:.1f}")
print(f"Singular values < 1.0: {n_near_zero} out of {len(S)}")
print(f"Smallest 5 singular values: {S[-5:]}")
