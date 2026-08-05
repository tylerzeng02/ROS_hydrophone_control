import csv
import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

# Load with pose_id preserved, in file order
pose_ids = []
angles_list = []
pos_list = []
quat_list = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pose_ids.append(int(row["pose_id"]))
        angles_list.append([float(row[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)])
        pos_list.append([
            float(row["moving_relative_fixed_tx_mm"]),
            float(row["moving_relative_fixed_ty_mm"]),
            float(row["moving_relative_fixed_tz_mm"]),
        ])
        q0 = float(row["moving_relative_fixed_q0"])
        qx = float(row["moving_relative_fixed_qx"])
        qy = float(row["moving_relative_fixed_qy"])
        qz = float(row["moving_relative_fixed_qz"])
        quat_list.append([qx, qy, qz, q0])

pose_ids = np.array(pose_ids)
angles = np.array(angles_list)
pos_mm = np.array(pos_list)
quat_xyzw = np.array(quat_list)

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])
result = least_squares(
    ck.residual_function, x0, method="lm",
    args=(angles, pos_mm, quat_xyzw), verbose=0, max_nfev=50000
)
rms_pos, pos_errs = ck.rms_position_error_mm(result.x, angles, pos_mm)

print("Per-pose position error vs pose_id (in file/capture order):")
print(f"{'pose_id':>8} {'error_mm':>10}")
for pid, err in zip(pose_ids, pos_errs):
    marker = ""
    if pid in (36, 37, 84, 85):
        marker = "  <-- session boundary"
    print(f"{pid:>8} {err:>10.2f}{marker}")

# Group stats around the two known resume boundaries (pose 37, pose 85)
def group_stats(mask, label):
    e = pos_errs[mask]
    if len(e) == 0:
        print(f"{label}: no data")
        return
    print(f"{label}: n={len(e)}, mean={e.mean():.2f}mm, median={np.median(e):.2f}mm, std={e.std():.2f}mm")

print("\nGrouped by capture session:")
group_stats(pose_ids <= 36, "Session 1 (pose_id <= 36)")
group_stats((pose_ids >= 37) & (pose_ids <= 84), "Session 2 (37-84)")
group_stats(pose_ids >= 85, "Session 3 (85+)")
