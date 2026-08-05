import numpy as np
import csv
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)

pose_ids = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pose_ids.append(int(row["pose_id"]))
pose_ids = np.array(pose_ids)


def unbounded_fit(idx):
    a = angles[idx]
    p = pos_mm[idx]
    q = quat_xyzw[idx]

    def residual(x):
        return ck.residual_function(x, a, p, q)

    base_xyz0, base_rpy0 = ck.initial_base_guess(a[0], p[0], q[0])
    x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
    result = least_squares(residual, x0, method="lm", max_nfev=100000)
    rms, errs = ck.rms_position_error_mm(result.x, a, p)
    return result, rms, errs


sets = {
    "all (n=%d)" % len(angles): np.arange(len(angles)),
    "first half, pose_id<100": np.where(pose_ids < 100)[0],
    "second half, pose_id>=100": np.where(pose_ids >= 100)[0],
}

results = {}
for name, idx in sets.items():
    result, rms, errs = unbounded_fit(idx)
    results[name] = result
    print(f"{name}: n={len(idx)}  train RMS={rms:.2f} mm  median={np.median(errs):.2f}  max={errs.max():.2f}")

# Cross-check: does the second-half-fitted model also do well on the first half, and vice versa?
print("\nCross-generalization check:")
idx_first = sets["first half, pose_id<100"]
idx_second = sets["second half, pose_id>=100"]

rms_2nd_model_on_1st, _ = ck.rms_position_error_mm(results["second half, pose_id>=100"].x, angles[idx_first], pos_mm[idx_first])
rms_1st_model_on_2nd, _ = ck.rms_position_error_mm(results["first half, pose_id<100"].x, angles[idx_second], pos_mm[idx_second])
print(f"  Model fit on 2nd half, evaluated on 1st half: {rms_2nd_model_on_1st:.2f} mm")
print(f"  Model fit on 1st half, evaluated on 2nd half: {rms_1st_model_on_2nd:.2f} mm")
