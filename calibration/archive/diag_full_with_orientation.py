import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])

# Use the REAL residual_function from calibrate_kinematics.py (position +
# down-weighted orientation + regularization on joint offsets/tool), but
# with NO bounds, to see the true unconstrained best case.
result = least_squares(
    ck.residual_function,
    x0,
    method="lm",
    args=(angles, pos_mm, quat_xyzw),
    verbose=0,
    max_nfev=50000,
)

rms_pos, pos_errs = ck.rms_position_error_mm(result.x, angles, pos_mm)
rms_orient, orient_errs = ck.rms_orientation_error_deg(result.x, angles, quat_xyzw)

params = ck.unpack_params(result.x)
print("Full fit with position+orientation residual, NO bounds:")
print("  joint_offsets (deg):", np.degrees(params.joint_offsets))
print("  tool_xyz (mm):", params.tool_xyz * 1000.0)
print("  tool_rpy (deg):", np.degrees(params.tool_rpy))
print("  base_xyz (mm):", params.base_xyz * 1000.0)
print("  base_rpy (deg):", np.degrees(params.base_rpy))
print()
print("Position RMS (mm):", rms_pos)
print("Orientation RMS (deg):", rms_orient)
print("Position error min/median/max:", pos_errs.min(), np.median(pos_errs), pos_errs.max())
print("Orientation error min/median/max:", orient_errs.min(), np.median(orient_errs), orient_errs.max())
