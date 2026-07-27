import numpy as np
import csv
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)

pose_ids = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pose_ids.append(int(row["pose_id"]))
pose_ids = np.array(pose_ids)

result = ck.run_calibration(angles, pos_mm, quat_xyzw, verbose=False)
rms_all, errs = ck.rms_position_error_mm(result.x, angles, pos_mm)

print("Overall RMS (fit on all data):", rms_all)
print("Overall n poses:", len(errs))

order = np.argsort(pose_ids)
pose_ids_sorted = pose_ids[order]
errs_sorted = errs[order]

print("\npose_id range -> mean error (mm), n poses")
edges = list(range(0, 220, 20))
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (pose_ids_sorted >= lo) & (pose_ids_sorted < hi)
    if mask.sum() > 0:
        print(f"  [{lo:3d},{hi:3d}): mean={errs_sorted[mask].mean():6.2f}  median={np.median(errs_sorted[mask]):6.2f}  n={mask.sum()}")

mid = pose_ids_sorted.max() // 2 if len(pose_ids_sorted) else 0
first_half_mask = pose_ids_sorted < 100
second_half_mask = pose_ids_sorted >= 100
print(f"\nFirst half (pose_id < 100): mean={errs_sorted[first_half_mask].mean():.2f} mm, n={first_half_mask.sum()}")
print(f"Second half (pose_id >= 100): mean={errs_sorted[second_half_mask].mean():.2f} mm, n={second_half_mask.sum()}")

corr = np.corrcoef(pose_ids_sorted, errs_sorted)[0, 1]
print(f"\nCorrelation of error with pose_id (index order): {corr:.3f}")
