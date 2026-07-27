import csv
import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total poses: {len(rows)}")
subset = rows[:45]
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
orient_rms, _ = ck.rms_orientation_error_deg(result.x, angles, quat_xyzw)

print(f"Position RMS: {rms:.2f} mm (median {np.median(errs):.2f}, max {errs.max():.2f})")
print(f"Orientation RMS: {orient_rms:.2f} deg")
print(f"tool_xyz (mm): {params.tool_xyz * 1000.0}")
print(f"tool_rpy (deg): {np.degrees(params.tool_rpy)}")
print(f"tool_xyz magnitude (mm): {np.linalg.norm(params.tool_xyz)*1000:.1f}")

ck.identifiability_report(result)
