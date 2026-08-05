"""Fast diagnostic (single Jacobian evaluation, no optimization) to check
WHY the extended fit was converging so slowly. Evaluates the residual
Jacobian at the already-validated all-7-tilt solution (8.98mm) with the
two new params (shoulder_pitch-Y, elbow_yaw-Z) held at 0, and compares
their column norms (signal strength) against the existing, known-good
origin params. A much smaller column norm would mean the data barely
constrains that direction -- a "weak but not exactly zero" signal, which
causes exactly this kind of slow/wandering convergence in an unregularized
least_squares fit (as opposed to true unidentifiability, which we already
ruled out via the exact-orthogonality check).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

SHOULDER_PITCH_IDX = 1
ELBOW_PITCH_IDX = 3
SHOULDER_YAW_IDX = 2
ELBOW_YAW_IDX = 4
WRIST_PITCH_IDX = 5

PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    PERP.append((u, v))


def fk_combined(joint_angles_rad, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z):
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
        if i == SHOULDER_PITCH_IDX:
            origin = origin + np.array([0.0, o_sp_y[0], 0.0])
        if i == ELBOW_YAW_IDX:
            origin = origin + np.array([0.0, 0.0, o_ey_z[0]])
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(joint_rotation, origin)
    return T


def predict(angles_row, base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


N_TILT = 14


def unpack(x):
    base_params = x[:19]
    tilt_all = x[19:19 + N_TILT].reshape(7, 2)
    o_elbow = x[19 + N_TILT:22 + N_TILT]
    o_sy_xz = x[22 + N_TILT:24 + N_TILT]
    o_wp = x[24 + N_TILT:27 + N_TILT]
    o_sp_y = x[27 + N_TILT:28 + N_TILT]
    o_ey_z = x[28 + N_TILT:29 + N_TILT]
    return base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z


def residual_vec(x, angles, pos_mm, quat_xyzw):
    base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z = unpack(x)
    n = len(angles)
    pos_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict(angles[i], base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
    return pos_res.ravel()  # position-only, for a quick signal-strength check


print("Step 1: quickly re-fit the KNOWN-GOOD 41-param (all-7-tilt) model alone")
print("(no new params) to get a realistic operating point -- should converge fast,")
print("matching the 8.98mm result already validated.\n")

# reuse the already-known-good result from diag_tilt_sweep_fit.py by re-fitting
# quickly with a modest nfev cap (should converge well within it if healthy)
N_EXTRA_OLD = N_TILT + 3 + 2 + 3  # 22, no new params
base_xyz0, base_rpy0 = ck.initial_base_guess(angles_all[0], pos_mm_all[0], quat_xyzw_all[0])


def residual_old(x, angles, pos_mm, quat_xyzw):
    x_full = np.concatenate([x, [0.0, 0.0]])  # o_sp_y=0, o_ey_z=0
    n = len(angles)
    base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z = unpack(x_full)
    pos_res = np.zeros((n, 3)); orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict(angles[i], base_params, tilt_all, o_elbow, o_sy_xz, o_wp, o_sp_y, o_ey_z)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        orient_res[i] = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec() * 100.0 * 0.3
    params = ck.unpack_params(base_params)
    reg = np.concatenate([
        params.joint_offsets * (8.0 / np.radians(8.0)),
        params.tool_xyz * (1.0 / 0.01),
        params.tool_rpy * (1.0 / np.radians(10.0)),
        tilt_all.ravel() * (5.0 / np.radians(8.0)),
    ])
    return np.concatenate([pos_res.ravel(), orient_res.ravel(), reg])


x0_old = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    np.zeros(N_EXTRA_OLD)
])
import time
t0 = time.time()
result_old = least_squares(lambda x: residual_old(x, angles_all, pos_mm_all, quat_xyzw_all),
                            x0_old, method="lm", max_nfev=20000)
elapsed = time.time() - t0
print(f"  status={result_old.status}  nfev={result_old.nfev}  time={elapsed:.1f}s")
rms_check = np.sqrt(np.mean(residual_old(result_old.x, angles_all, pos_mm_all, quat_xyzw_all)[:298*3]**2))
print(f"  (rough position RMS check, unnormalized): {rms_check:.2f}\n")

print("Step 2: at this converged point, evaluate the Jacobian of the FULL")
print("43-param position-only residual (new params o_sp_y, o_ey_z = 0) and")
print("compare column norms -- weak columns explain slow convergence when free.\n")

x_full_point = np.concatenate([result_old.x, [0.0, 0.0]])

eps = 1e-6
base_res = residual_vec(x_full_point, angles_all, pos_mm_all, quat_xyzw_all)
n_params = len(x_full_point)
col_norms = np.zeros(n_params)
for j in range(n_params):
    xp = x_full_point.copy(); xp[j] += eps
    xm = x_full_point.copy(); xm[j] -= eps
    col = (residual_vec(xp, angles_all, pos_mm_all, quat_xyzw_all) -
           residual_vec(xm, angles_all, pos_mm_all, quat_xyzw_all)) / (2 * eps)
    col_norms[j] = np.linalg.norm(col)

param_names = (["joint_off_%d" % i for i in range(7)] + ["tool_x", "tool_y", "tool_z"] +
               ["tool_r", "tool_p", "tool_yaw"] + ["base_x", "base_y", "base_z"] +
               ["base_r", "base_p", "base_yaw"] +
               [f"tilt{j}_{c}" for j in range(7) for c in range(2)] +
               ["o_elbow_x", "o_elbow_y", "o_elbow_z", "o_sy_x", "o_sy_z",
                "o_wp_x", "o_wp_y", "o_wp_z", "o_sp_y (NEW)", "o_ey_z (NEW)"])

order = np.argsort(col_norms)
print(f"{'param':16s} {'column norm':>12s}")
for idx in order:
    print(f"{param_names[idx]:16s} {col_norms[idx]:12.2f}")
