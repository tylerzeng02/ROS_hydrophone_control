import re
import numpy as np
import calibrate_kinematics as ck

RAW_PATH = "target_poses_raw.txt"
N_SELECT = 35

# Poses skipped ('s' hotkey) during the original 200-pose capture -- exclude
# these from the candidate pool entirely, since some may have a persistent
# physical visibility issue at that joint configuration (arm geometry
# blocking the marker), not just transient foot traffic. Computed by diffing
# five_pose_ndi_capture.csv's captured pose_ids against the full 0-199 range.
SKIPPED_POSES = {0, 9, 17, 32, 33, 34, 49, 50, 56, 62, 74, 85, 86, 87,
                  102, 103, 104, 105, 144, 145, 146, 149, 159, 160, 161, 162}

ticks = []
with open(RAW_PATH) as f:
    for line in f:
        nums = [int(x) for x in re.findall(r"\d+", line)]
        assert len(nums) == 7, line
        ticks.append(nums)
ticks = np.array(ticks)
n = len(ticks)
print(f"Loaded {n} poses from TARGET_POSES")
print(f"Excluding {len(SKIPPED_POSES)} previously-skipped poses from the candidate pool\n")

zero_ticks = np.array(ck.NOMINAL_ZERO_TICKS)
angles_rad = (ticks - zero_ticks) / ck.TICKS_PER_RADIAN

ranges = np.array([(hi - lo) / ck.TICKS_PER_RADIAN for lo, hi in ck.JOINT_TICK_RANGES])
normalized = angles_rad / ranges

candidate_idx = np.array([i for i in range(n) if i not in SKIPPED_POSES])
candidates = normalized[candidate_idx]

# Greedy farthest-point (k-center) sampling over the candidate pool only.
centroid = candidates.mean(axis=0)
first_local = np.argmax(np.linalg.norm(candidates - centroid, axis=1))

selected_local = [first_local]
min_dist = np.linalg.norm(candidates - candidates[first_local], axis=1)

while len(selected_local) < N_SELECT:
    next_local = np.argmax(min_dist)
    selected_local.append(next_local)
    new_dist = np.linalg.norm(candidates - candidates[next_local], axis=1)
    min_dist = np.minimum(min_dist, new_dist)

selected = sorted(int(candidate_idx[i]) for i in selected_local)
print(f"Selected {len(selected)} pose indices (0-based, matches TARGET_POSES/pose_id):")
print(selected)

assert not (set(selected) & SKIPPED_POSES), "selection leaked a skipped pose!"
print("\nConfirmed: no overlap with previously-skipped poses.\n")

print("Per-joint coverage of this subset vs full 200:")
for j in range(ck.N_JOINTS):
    lo_full, hi_full = np.degrees(angles_rad[:, j].min()), np.degrees(angles_rad[:, j].max())
    sub = np.degrees(angles_rad[selected, j])
    lo_sub, hi_sub = sub.min(), sub.max()
    full_range = hi_full - lo_full
    sub_range = hi_sub - lo_sub
    pct = 100.0 * sub_range / full_range
    print(f"  {ck.JOINT_NAMES[j]:24s} subset=[{lo_sub:7.1f},{hi_sub:7.1f}]  full=[{lo_full:7.1f},{hi_full:7.1f}]  {pct:5.1f}% of full range")

cpp_list = ", ".join(str(i) for i in selected)
print(f"\nC++ initializer list:\n{{{cpp_list}}}")
