"""Check whether fold 7's unusually low GP-residual error (4.37mm, vs.
4-10mm elsewhere) is explained by cross-batch near-duplicate poses rather
than that region genuinely being easier (2026-07-30). Blocked/contiguous-
index CV protects against ADJACENT-trajectory near-twins leaking between
train and test, but does nothing to catch a near-duplicate pose that ended
up at a completely different, non-adjacent pose_id because the 298-pose
dataset is a merge of several collection batches (per CLAUDE.md: "40 -> 190
-> 200 new + 91 reused from an earlier batch -> +13 orientation-diverse").
If one of the "91 reused" poses is a near-twin of something already in a
different part of the file, whichever fold it lands in would show
artificially low error -- a form of leakage blocked-CV was never designed
to catch.

Method: for every pose, find its nearest neighbor (Euclidean distance
across all 7 joint angles, converted to ticks for interpretability) among
ALL OTHER poses in the dataset -- not just within its own fold. A normal,
expected-close match is one where the nearest neighbor is ADJACENT in
pose_id (the next/previous step of the same hand-recorded trajectory,
which naturally differ by a small amount). A suspicious match is one that
is BOTH unusually close (much closer than typical adjacent-trajectory
spacing) AND non-adjacent in pose_id (the signature of a cross-batch
duplicate, not a normal fine trajectory step).
"""
import csv
import numpy as np
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

with open(CSV_PATH, newline="") as f:
    pose_id_all = np.array([int(row["pose_id"]) for row in csv.DictReader(f)])

TICKS_PER_RADIAN = 4096.0 / (2.0 * np.pi)
angles_ticks = angles_all * TICKS_PER_RADIAN  # (n, 7), tick-equivalent units

# Pairwise distance matrix (ticks, Euclidean across all 7 joints)
diff = angles_ticks[:, None, :] - angles_ticks[None, :, :]
dist_matrix = np.linalg.norm(diff, axis=2)
np.fill_diagonal(dist_matrix, np.inf)  # exclude self-match

nearest_idx = np.argmin(dist_matrix, axis=1)
nearest_dist = dist_matrix[np.arange(n_all), nearest_idx]
nearest_pose_id = pose_id_all[nearest_idx]
pose_id_gap = np.abs(pose_id_all - nearest_pose_id)

print(f"Dataset: {n_all} poses\n")
print("Nearest-neighbor distance (ticks, across ALL other poses) percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(nearest_dist, p):.1f}")

# Overall: how often is the nearest neighbor pose_id-adjacent (normal,
# expected) vs. far away in pose_id but still very close in angle-space
# (the suspicious duplicate signature)?
ADJACENT_THRESHOLD = 3       # pose_id gap this small = normal trajectory step
SUSPICIOUS_DIST_TICKS = 30   # closer than this is "very close" in angle-space

adjacent_mask = pose_id_gap <= ADJACENT_THRESHOLD
close_mask = nearest_dist < SUSPICIOUS_DIST_TICKS
suspicious_mask = close_mask & ~adjacent_mask

print(f"\nOverall: {adjacent_mask.sum()}/{n_all} poses' nearest neighbor is "
      f"pose_id-adjacent (gap <= {ADJACENT_THRESHOLD}) -- normal trajectory steps.")
print(f"{suspicious_mask.sum()}/{n_all} poses have a very close "
      f"(<{SUSPICIOUS_DIST_TICKS} ticks) nearest neighbor that is NOT "
      f"pose_id-adjacent -- candidate cross-batch duplicates.")

if suspicious_mask.sum() > 0:
    print("\nSuspicious near-duplicate candidates (pose_id, nearest pose_id, dist ticks):")
    order = np.argsort(nearest_dist[suspicious_mask])
    suspicious_indices = np.where(suspicious_mask)[0][order]
    for idx in suspicious_indices:
        print(f"  pose_id={pose_id_all[idx]:3d}  nearest={nearest_pose_id[idx]:3d}  "
              f"dist={nearest_dist[idx]:.1f} ticks")

# Fold 7 specifically (pose_id 265-302)
print("\n=== Fold 7 (pose_id 265-302) detail ===")
fold7_mask = (pose_id_all >= 265) & (pose_id_all <= 302)
print(f"{fold7_mask.sum()} poses\n")
for idx in np.where(fold7_mask)[0]:
    flag = " <-- SUSPICIOUS (non-adjacent + close)" if suspicious_mask[idx] else ""
    print(f"  pose_id={pose_id_all[idx]:3d}  nearest={nearest_pose_id[idx]:3d}  "
          f"gap={pose_id_gap[idx]:3d}  dist={nearest_dist[idx]:6.1f} ticks{flag}")

fold7_suspicious_count = suspicious_mask[fold7_mask].sum()
print(f"\nFold 7 suspicious (likely-duplicate) poses: {fold7_suspicious_count} "
      f"out of {fold7_mask.sum()}")
if fold7_suspicious_count > fold7_mask.sum() * 0.15:
    print("-> A meaningful fraction of fold 7 has suspiciously close, "
          "non-adjacent neighbors elsewhere in the dataset -- real leakage "
          "risk, treat fold 7's low error with real skepticism.")
else:
    print("-> Fold 7 does not show unusual duplicate-neighbor density compared "
          "to the rest of the dataset -- its low error is more likely a "
          "genuine 'this region is easier' effect than leakage.")
