import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)


def residual_no_reg(x):
    # Same as ck.residual_function but with the regularization block
    # entirely removed -- pure position + orientation residuals only, no
    # pull toward zero on joint_offsets/tool_xyz/tool_rpy.
    params = ck.unpack_params(x)
    pos_res = np.zeros((n, 3))
    orient_res = np.zeros((n, 3))
    for i in range(n):
        T_pred = ck.predict_relative_pose(angles[i], params)
        pos_res[i] = pos_mm[i] - T_pred[:3, 3] * 1000.0
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_res[i] = Rotation.from_matrix(R_err).as_rotvec() * 100.0 * 0.3
    return np.concatenate([pos_res.ravel(), orient_res.ravel()])


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)

result_reg = least_squares(lambda x: ck.residual_function(x, angles, pos_mm, quat_xyzw),
                            x0, method="lm", max_nfev=100000)
result_noreg = least_squares(residual_no_reg, x0, method="lm", max_nfev=100000)

rms_reg, _ = ck.rms_position_error_mm(result_reg.x, angles, pos_mm)
rms_noreg, _ = ck.rms_position_error_mm(result_noreg.x, angles, pos_mm)

p_reg = ck.unpack_params(result_reg.x)
p_noreg = ck.unpack_params(result_noreg.x)

print("WITH regularization (current default):")
print(f"  RMS: {rms_reg:.2f} mm")
print(f"  tool_xyz (mm): {p_reg.tool_xyz * 1000}")
print(f"  tool_rpy (deg): {np.degrees(p_reg.tool_rpy)}")
print(f"  joint_offsets (deg): {np.degrees(p_reg.joint_offsets)}")

print("\nWITHOUT regularization (joint_offsets/tool_xyz/tool_rpy free to go anywhere):")
print(f"  RMS: {rms_noreg:.2f} mm")
print(f"  tool_xyz (mm): {p_noreg.tool_xyz * 1000}")
print(f"  tool_rpy (deg): {np.degrees(p_noreg.tool_rpy)}")
print(f"  joint_offsets (deg): {np.degrees(p_noreg.joint_offsets)}")

print(f"\nRMS difference: {rms_reg - rms_noreg:+.3f} mm")
print("(if ~0, regularization isn't meaningfully constraining anything;")
print(" if large, regularization was holding a parameter back from its true value)")
