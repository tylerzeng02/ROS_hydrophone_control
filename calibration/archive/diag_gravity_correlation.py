import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)

WRIST_PITCH_IDX = 5
ELBOW_PITCH_IDX = 3
SHOULDER_YAW_IDX = 2
axis0 = ck.JOINT_AXES[WRIST_PITCH_IDX] / np.linalg.norm(ck.JOINT_AXES[WRIST_PITCH_IDX])
arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
u0 = np.cross(axis0, arbitrary); u0 /= np.linalg.norm(u0)
v0 = np.cross(axis0, u0)
WRIST_PITCH_TILT_UV = np.radians(np.array([-0.63, -4.55]))
ELBOW_PITCH_ORIGIN_DELTA = np.array([-2.08, 4.46, 6.59]) / 1000.0
SHOULDER_YAW_ORIGIN_DELTA = np.array([-1.18, 0.04, 4.19]) / 1000.0


def fk_combined(joint_angles_rad, tilt_uv, origin_elbow, origin_shoulder_yaw):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        if i == WRIST_PITCH_IDX:
            perturbed = a + tilt_uv[0] * u0 + tilt_uv[1] * v0
            a = perturbed / np.linalg.norm(perturbed)
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == ELBOW_PITCH_IDX:
            origin = origin + origin_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + origin_shoulder_yaw
        joint_rotation = Rotation.from_rotvec(a * joint_angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
    return T


def predict(angles_row, base_params, tilt_uv, origin_elbow, origin_shoulder_yaw):
    params = ck.unpack_params(base_params)
    corrected_angles = angles_row + params.joint_offsets
    T_fk = fk_combined(corrected_angles, tilt_uv, origin_elbow, origin_shoulder_yaw)
    T_tool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool


def residual(x):
    base_params = x[:19]
    tilt_uv = x[19:21]
    origin_elbow = x[21:24]
    origin_shoulder_yaw = x[24:27]
    params = ck.unpack_params(base_params)
    pos_res = np.zeros((n, 3))
    orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = predict(angles[i], base_params, tilt_uv, origin_elbow, origin_shoulder_yaw)
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


def rms_full(x):
    base_params = x[:19]
    tilt_uv = x[19:21]
    origin_elbow = x[21:24]
    origin_shoulder_yaw = x[24:27]
    errs = np.zeros(n)
    for i in range(n):
        T_pred = predict(angles[i], base_params, tilt_uv, origin_elbow, origin_shoulder_yaw)
        errs[i] = np.linalg.norm(pos_mm[i] - T_pred[:3, 3] * 1000.0)
    return np.sqrt(np.mean(errs ** 2)), errs


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([
    ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0),
    WRIST_PITCH_TILT_UV, ELBOW_PITCH_ORIGIN_DELTA, SHOULDER_YAW_ORIGIN_DELTA
])
result = least_squares(residual, x0, method="lm", max_nfev=200000)
rms, errs = rms_full(result.x)
print(f"Fully-corrected model, JOINTLY re-optimized (27 params): RMS = {rms:.2f} mm")
print(f"(for reference, sequential-incremental validation gave 11.99 mm)\n")

# Gravity-loading proxy: horizontal (XY-plane) distance from base to the
# nominal end-effector position for each pose. Unlike the earlier "extension"
# check (raw 3D distance from origin), this specifically captures moment arm
# under gravity (Z assumed vertical/up per the URDF convention) -- reaching
# straight up gives high 3D extension but near-zero horizontal moment arm;
# reaching horizontally gives high moment arm at the same 3D extension.
horiz_reach_mm = np.zeros(n)
total_reach_mm = np.zeros(n)
for i in range(n):
    T_fk = ck.forward_kinematics(angles[i])
    ee_pos = T_fk[:3, 3] * 1000.0
    horiz_reach_mm[i] = np.linalg.norm(ee_pos[:2])  # sqrt(x^2+y^2)
    total_reach_mm[i] = np.linalg.norm(ee_pos)

corr_horiz = np.corrcoef(horiz_reach_mm, errs)[0, 1]
corr_total = np.corrcoef(total_reach_mm, errs)[0, 1]

print(f"Correlation of error with HORIZONTAL reach (gravity moment-arm proxy): {corr_horiz:.3f}")
print(f"Correlation of error with TOTAL 3D reach (for comparison):             {corr_total:.3f}")
print(f"\nHorizontal reach range: {horiz_reach_mm.min():.1f} - {horiz_reach_mm.max():.1f} mm\n")

order = np.argsort(horiz_reach_mm)
quartile_size = n // 4
print("Horizontal-reach quartile -> mean error (mm):")
for q in range(4):
    start = q * quartile_size
    end = (q + 1) * quartile_size if q < 3 else n
    idx = order[start:end]
    print(f"  Q{q+1} (reach {horiz_reach_mm[idx].min():.0f}-{horiz_reach_mm[idx].max():.0f}mm): "
          f"mean error = {errs[idx].mean():.2f} mm, n={len(idx)}")

# Also a rough mass-weighted torque proxy using the URDF's per-link masses
# (values are round placeholder-looking numbers, not precisely measured --
# treat this as a secondary, cruder check, not the primary one)
LINK_MASSES = [50.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # shoulder_roll..wrist_roll
torque_proxy = np.zeros(n)
for i in range(n):
    T = np.eye(4)
    cumulative_torque = 0.0
    for j in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[j]
        origin = ck.JOINT_ORIGINS_M[j]
        joint_rotation = Rotation.from_rotvec(a * angles[i, j]).as_matrix()
        T_joint = ck.homogeneous_transform(joint_rotation, origin)
        T = T @ T_joint
        joint_pos_mm = T[:3, 3] * 1000.0
        cumulative_torque += LINK_MASSES[j] * np.linalg.norm(joint_pos_mm[:2])
    torque_proxy[i] = cumulative_torque

corr_torque = np.corrcoef(torque_proxy, errs)[0, 1]
print(f"\nCorrelation of error with mass-weighted cumulative torque proxy: {corr_torque:.3f}")
