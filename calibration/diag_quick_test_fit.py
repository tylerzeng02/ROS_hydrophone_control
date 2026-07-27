import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)
print(f"Loaded {n} poses from quick_calibration_test.csv\n")


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)

rms, errs = ck.rms_position_error_mm(result.x, angles, pos_mm)
orient_rms, orient_errs = ck.rms_orientation_error_deg(result.x, angles, quat_xyzw)

params = ck.unpack_params(result.x)

print("=== Quick test (15 poses, single sitting) fit ===")
print(f"Position RMS:    {rms:.3f} mm  (median {np.median(errs):.3f}, max {errs.max():.3f})")
print(f"Orientation RMS: {orient_rms:.3f} deg")
print()
print("Fitted joint offsets (deg):", np.degrees(params.joint_offsets))
print("Fitted tool_xyz (mm):", params.tool_xyz * 1000.0)
print("Fitted tool_rpy (deg):", np.degrees(params.tool_rpy))
print("Fitted base_xyz (mm):", params.base_xyz * 1000.0)
print("Fitted base_rpy (deg):", np.degrees(params.base_rpy))
print()

# Compare directly against the original 174-pose fit's RMS for reference
print(f"For reference: original 174-pose (multi-hour session) unbounded fit RMS was ~17.24mm")
print(f"Arm's independently-measured repeatability floor: ~0.5mm")
print()

ck.identifiability_report(result)
