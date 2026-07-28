import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

perp_vectors = []
for axis in ck.JOINT_AXES:
    axis = axis / np.linalg.norm(axis)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    perp_vectors.append((u, v))


def tilted_axes(tilt_params):
    axes = np.zeros((ck.N_JOINTS, 3))
    for i in range(ck.N_JOINTS):
        axis = ck.JOINT_AXES[i]
        u, v = perp_vectors[i]
        perturbed = axis + tilt_params[i, 0] * u + tilt_params[i, 1] * v
        axes[i] = perturbed / np.linalg.norm(perturbed)
    return axes


def fk_combined(joint_angles_rad, axes, origin_deltas_m):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        joint_rotation = Rotation.from_rotvec(axes[i] * joint_angles_rad[i]).as_matrix()
        origin = ck.JOINT_ORIGINS_M[i] + origin_deltas_m[i]
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


N_TILT = ck.N_JOINTS * 2
N_ORIGIN = ck.N_JOINTS * 3


def residual_no_reg(x, angles, pos_mm, quat_xyzw):
    base_params = x[:19]
    tilt_flat = x[19:19 + N_TILT]
    origin_flat = x[19 + N_TILT:]
    tilt_params = tilt_flat.reshape(ck.N_JOINTS, 2)
    origin_deltas = origin_flat.reshape(ck.N_JOINTS, 3)
    axes = tilted_axes(tilt_params)
    params = ck.unpack_params(base_params)
    n = len(angles)
    pos_res = np.zeros((n, 3))
    orient_res = np.zeros((n, 3))
    for i in range(n):
        corrected_angles = angles[i] + params.joint_offsets
        T_fk = fk_combined(corrected_angles, axes, origin_deltas)
        T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
        T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
        T_pred = T_base @ T_fk @ T_tool
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3
    return np.concatenate([pos_res.ravel(), orient_res.ravel()])


print("Step 1: Fit the full 54-param model on all 190 poses (unregularized, to get a clean Jacobian)")
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.zeros(N_TILT), np.zeros(N_ORIGIN)
])
result = least_squares(lambda x: residual_no_reg(x, angles_all, pos_mm_all, quat_xyzw_all),
                        x0, method="lm", max_nfev=300000)

J = result.jac  # rows: [pos(3*n), orient(3*n)] -- no reg rows since residual_no_reg has none
n_params = J.shape[1]
print(f"Jacobian shape: {J.shape}\n")

# Per-pose 6-row block (3 position + 3 orientation), pulled from the position
# block (rows 3i:3i+3) and the orientation block (rows 3n+3i:3n+3i+3).
def pose_block(i):
    pos_rows = J[3 * i: 3 * i + 3, :]
    orient_rows = J[3 * n_all + 3 * i: 3 * n_all + 3 * i + 3, :]
    return np.vstack([pos_rows, orient_rows])

print("Step 2: Greedy observability-maximizing pose selection (E-optimality: maximize min singular value)")
N_SELECT = 40
selected = []
remaining = list(range(n_all))

# Seed with the pose whose own block has the largest singular value spread
first = max(remaining, key=lambda i: np.linalg.svd(pose_block(i), compute_uv=False)[0])
selected.append(first)
remaining.remove(first)

cumulative = pose_block(first)

while len(selected) < N_SELECT:
    best_idx, best_min_sv = None, -1
    for i in remaining:
        candidate = np.vstack([cumulative, pose_block(i)])
        sv = np.linalg.svd(candidate, compute_uv=False)
        min_sv = sv[-1] if len(sv) >= n_params else 0.0
        if min_sv > best_min_sv:
            best_min_sv = min_sv
            best_idx = i
    selected.append(best_idx)
    remaining.remove(best_idx)
    cumulative = np.vstack([cumulative, pose_block(best_idx)])

selected_sv = np.linalg.svd(cumulative, compute_uv=False)
print(f"Selected {len(selected)} poses via observability-maximizing greedy selection")
print(f"Min singular value: {selected_sv[-1]:.4f}   Condition number: {selected_sv[0]/selected_sv[-1]:.1f}\n")

print("Step 3: Refit the 54-param model using ONLY these observability-selected poses")
sel_idx = np.array(selected)
angles_sel = angles_all[sel_idx]
pos_sel = pos_mm_all[sel_idx]
quat_sel = quat_xyzw_all[sel_idx]

base_xyz0s, base_rpy0s = ck.initial_base_guess(angles_sel[0], pos_sel[0], quat_sel[0])
x0s = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0s, base_rpy0s),
    np.zeros(N_TILT), np.zeros(N_ORIGIN)
])
result_sel = least_squares(lambda x: residual_no_reg(x, angles_sel, pos_sel, quat_sel),
                            x0s, method="lm", max_nfev=300000)

n_pos_sel = 3 * len(sel_idx)
errs_sel = np.linalg.norm(result_sel.fun[:n_pos_sel].reshape(len(sel_idx), 3), axis=1)
rms_sel = np.sqrt(np.mean(errs_sel ** 2))

S_sel = np.linalg.svd(result_sel.jac, compute_uv=False)
print(f"RMS on the 40 observability-selected poses: {rms_sel:.2f} mm")
print(f"Condition number: {S_sel[0]/S_sel[-1]:.1f}")
print(f"Number of singular values < 1.0: {(S_sel < 1.0).sum()} out of {len(S_sel)}")
print(f"\nFor reference: the earlier 54-param fit on all 190 poses (joint-range-selected)")
print(f"had condition number ~1.27e10 with 20 near-zero singular values.")
