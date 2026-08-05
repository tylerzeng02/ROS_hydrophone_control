import csv
import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total poses so far: {len(rows)}")
subset = rows[:70]
print(f"Using first {len(subset)} poses (pose_ids {subset[0]['pose_id']}..{subset[-1]['pose_id']})\n")


def rows_to_arrays(rows):
    angles = np.array([[float(r[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)] for r in rows])
    pos_mm = np.array([[float(r["moving_relative_fixed_tx_mm"]),
                         float(r["moving_relative_fixed_ty_mm"]),
                         float(r["moving_relative_fixed_tz_mm"])] for r in rows])
    quat_xyzw = np.array([[float(r["moving_relative_fixed_qx"]),
                            float(r["moving_relative_fixed_qy"]),
                            float(r["moving_relative_fixed_qz"]),
                            float(r["moving_relative_fixed_q0"])] for r in rows])
    return angles, pos_mm, quat_xyzw


angles, pos_mm, quat_xyzw = rows_to_arrays(subset)


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)
params = ck.unpack_params(result.x)

rms, errs = ck.rms_position_error_mm(result.x, angles, pos_mm)

print(f"Position RMS: {rms:.2f} mm (median {np.median(errs):.2f}, max {errs.max():.2f})")
print(f"tool_xyz (mm): {params.tool_xyz * 1000.0}")
print(f"tool_rpy (deg): {np.degrees(params.tool_rpy)}")

S = np.linalg.svd(result.jac, compute_uv=False)
cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
print(f"Condition number: {cond:.1f}")

print(f"\nFor reference, all 103 poses so far: RMS = 12.16 mm")
