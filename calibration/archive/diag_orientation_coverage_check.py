"""Quantify actual end-effector orientation diversity in the 298-pose
calibration dataset (2026-07-30) -- prompted by a concern that cross-
validation (random, blocked, or trajectory-aware) can only measure
generalization WITHIN the range of orientations already collected. If the
whole dataset only samples a narrow slice of possible tool orientations, no
re-splitting of that same data can ever reveal an orientation-coverage
gap -- that requires actually checking how diverse the collected
orientations are, independent of any model fit.

No fitting here -- pure geometry on moving_relative_fixed quaternions
already in the CSV.
"""
import csv
import numpy as np
from scipy.spatial.transform import Rotation

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

pose_id_all = np.array([int(r["pose_id"]) for r in rows])
quat_xyzw_all = np.array([
    [float(r["moving_relative_fixed_qx"]), float(r["moving_relative_fixed_qy"]),
     float(r["moving_relative_fixed_qz"]), float(r["moving_relative_fixed_q0"])]
    for r in rows
])
n = len(quat_xyzw_all)
rots = Rotation.from_quat(quat_xyzw_all)

# ---------------------------------------------------------------------------
# 1. Pairwise geodesic (angular) distance between every pair of poses'
#    orientations -- the most direct, reference-frame-independent measure
#    of how much orientation diversity actually exists.
# ---------------------------------------------------------------------------
rotvecs = rots.as_rotvec()  # not directly usable pairwise; compute relative rotations instead
angles_pairwise = np.zeros((n, n))
mats = rots.as_matrix()
for i in range(n):
    rel = mats[i].T @ mats  # (n,3,3): R_i^T @ R_j for all j
    # angle of each relative rotation via trace formula
    traces = np.trace(rel, axis1=1, axis2=2)
    cos_angle = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    angles_pairwise[i] = np.degrees(np.arccos(cos_angle))

iu = np.triu_indices(n, k=1)
pairwise_flat = angles_pairwise[iu]
print(f"Dataset: {n} poses")
print(f"\nPairwise orientation distance (degrees, {len(pairwise_flat)} pairs):")
print(f"  max:    {pairwise_flat.max():.2f}")
print(f"  p99:    {np.percentile(pairwise_flat, 99):.2f}")
print(f"  p90:    {np.percentile(pairwise_flat, 90):.2f}")
print(f"  median: {np.median(pairwise_flat):.2f}")
print(f"  mean:   {pairwise_flat.mean():.2f}")
print(f"  min (excluding self):  {pairwise_flat.min():.2f}")

# ---------------------------------------------------------------------------
# 2. Per-axis Euler spread (roll/pitch/yaw of the moving marker relative to
#    the fixed marker) -- shows whether some rotational DOF barely varies
#    across the whole dataset even if the aggregate angle looks diverse.
# ---------------------------------------------------------------------------
eulers_deg = rots.as_euler("xyz", degrees=True)
print("\nPer-axis Euler angle range across all 298 poses (degrees):")
for i, name in enumerate(["X (roll)", "Y (pitch)", "Z (yaw)"]):
    col = eulers_deg[:, i]
    print(f"  {name}: min={col.min():7.2f}  max={col.max():7.2f}  "
          f"range={col.max()-col.min():7.2f}  std={col.std():6.2f}")

# ---------------------------------------------------------------------------
# 3. Identify the outlier poses (largest median distance to everything
#    else) -- these are presumably the "~13 orientation-diverse" poses
#    CLAUDE.md mentions were deliberately added.
# ---------------------------------------------------------------------------
median_dist_to_others = np.median(angles_pairwise, axis=1)
order = np.argsort(-median_dist_to_others)
print(f"\nTop 20 poses by median angular distance to all other poses "
      f"(the most 'orientation-outlier' poses in the set):")
for idx in order[:20]:
    print(f"  pose_id={pose_id_all[idx]:3d}  median_dist_to_others={median_dist_to_others[idx]:6.2f} deg")

n_outliers_over_20deg = np.sum(median_dist_to_others > 20.0)
n_outliers_over_10deg = np.sum(median_dist_to_others > 10.0)
print(f"\nPoses with median distance > 20deg from the rest of the dataset: {n_outliers_over_20deg}")
print(f"Poses with median distance > 10deg from the rest of the dataset: {n_outliers_over_10deg}")
print(f"(If this count is small relative to {n} total poses, the dataset really is "
      f"dominated by one narrow orientation neighborhood, with only a handful of "
      f"outliers providing any real rotational diversity.)")
