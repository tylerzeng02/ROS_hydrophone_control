import re
import numpy as np
import calibrate_kinematics as ck

RAW_PATH = "target_poses_raw.txt"

SKIPPED_POSES = {0, 9, 17, 32, 33, 34, 49, 50, 56, 62, 74, 85, 86, 87,
                  102, 103, 104, 105, 144, 145, 146, 149, 159, 160, 161, 162}

CURRENT_SELECTION = [1, 7, 16, 18, 23, 25, 27, 31, 43, 46, 47, 55, 58, 60,
                      65, 72, 73, 78, 80, 83, 92, 99, 101, 106, 111, 117,
                      131, 133, 138, 142, 150, 163, 174, 191, 199]
POSE_TO_REPLACE = 46  # 9th slot in the run sequence

ticks = []
with open(RAW_PATH) as f:
    for line in f:
        nums = [int(x) for x in re.findall(r"\d+", line)]
        assert len(nums) == 7, line
        ticks.append(nums)
ticks = np.array(ticks)
n = len(ticks)

zero_ticks = np.array(ck.NOMINAL_ZERO_TICKS)
angles_rad = (ticks - zero_ticks) / ck.TICKS_PER_RADIAN
ranges = np.array([(hi - lo) / ck.TICKS_PER_RADIAN for lo, hi in ck.JOINT_TICK_RANGES])
normalized = angles_rad / ranges

retained = [p for p in CURRENT_SELECTION if p != POSE_TO_REPLACE]
excluded = SKIPPED_POSES | set(CURRENT_SELECTION)  # don't re-pick anything already used or skipped

candidate_idx = np.array([i for i in range(n) if i not in excluded])
candidates = normalized[candidate_idx]
retained_pts = normalized[retained]

# For each candidate, its min distance to the retained set -- pick whichever
# candidate is farthest from everything already kept (best fills the gap
# left by removing POSE_TO_REPLACE).
min_dist_to_retained = np.min(
    np.linalg.norm(candidates[:, None, :] - retained_pts[None, :, :], axis=2),
    axis=1
)
best_local = np.argmax(min_dist_to_retained)
replacement = int(candidate_idx[best_local])

new_selection = sorted(retained + [replacement])

print(f"Replacing pose {POSE_TO_REPLACE} with pose {replacement}")
print(f"\nNew selection ({len(new_selection)} poses):")
print(new_selection)

print("\nPer-joint coverage of the new subset vs full 200:")
for j in range(ck.N_JOINTS):
    lo_full, hi_full = np.degrees(angles_rad[:, j].min()), np.degrees(angles_rad[:, j].max())
    sub = np.degrees(angles_rad[new_selection, j])
    lo_sub, hi_sub = sub.min(), sub.max()
    pct = 100.0 * (hi_sub - lo_sub) / (hi_full - lo_full)
    print(f"  {ck.JOINT_NAMES[j]:24s} subset=[{lo_sub:7.1f},{hi_sub:7.1f}]  full=[{lo_full:7.1f},{hi_full:7.1f}]  {pct:5.1f}% of full range")

cpp_list = ", ".join(str(i) for i in new_selection)
print(f"\nC++ initializer list:\n{{{cpp_list}}}")
