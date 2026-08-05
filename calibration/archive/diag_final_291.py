import csv
import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_deduped.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)
print(f"Loaded {n} poses\n")


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)
params = ck.unpack_params(result.x)

rms, errs = ck.rms_position_error_mm(result.x, angles, pos_mm)
print(f"Baseline (19-param) fit RMS: {rms:.2f} mm (median {np.median(errs):.2f}, max {errs.max():.2f})")
print(f"tool_xyz (mm): {params.tool_xyz * 1000.0}")
print(f"tool_rpy (deg): {np.degrees(params.tool_rpy)}")
print(f"base_xyz (mm): {params.base_xyz * 1000.0}")
print(f"base_rpy (deg): {np.degrees(params.base_rpy)}")

S = np.linalg.svd(result.jac, compute_uv=False)
cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
print(f"Condition number: {cond:.1f}")
