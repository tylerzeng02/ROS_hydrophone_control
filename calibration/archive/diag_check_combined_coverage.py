import csv
import numpy as np
import calibrate_kinematics as ck

NEW_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/recorded_hand_poses.csv"
OLD2_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/recorded_hand_poses_OLD2.csv"


def load_ticks(path):
    ticks = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticks.append([int(row[f"tick_{i}"]) for i in range(7)])
    return np.array(ticks)


new_ticks = load_ticks(NEW_CSV)
old2_ticks = load_ticks(OLD2_CSV)
print(f"New dataset: {len(new_ticks)} poses")
print(f"OLD2 dataset: {len(old2_ticks)} poses\n")

old2_subset = old2_ticks[99:190]  # poses 100-190 (1-indexed), 91 poses
print(f"OLD2 poses 100-190: {len(old2_subset)} poses\n")

zero_ticks = np.array(ck.NOMINAL_ZERO_TICKS)

new_angles = np.degrees((new_ticks - zero_ticks) / ck.TICKS_PER_RADIAN)
old2_subset_angles = np.degrees((old2_subset - zero_ticks) / ck.TICKS_PER_RADIAN)
old2_full_angles = np.degrees((old2_ticks - zero_ticks) / ck.TICKS_PER_RADIAN)
combined_angles = np.degrees((np.vstack([new_ticks, old2_subset]) - zero_ticks) / ck.TICKS_PER_RADIAN)

print(f"{'joint':24s} {'NEW range':>20s} {'OLD2[100:190] range':>22s} {'OLD2 full range':>20s} {'COMBINED range':>20s} {'full safe':>12s}")
for j in range(ck.N_JOINTS):
    lo_tick, hi_tick = ck.JOINT_TICK_RANGES[j]
    full_range_deg = np.degrees((hi_tick - lo_tick) / ck.TICKS_PER_RADIAN)

    new_lo, new_hi = new_angles[:, j].min(), new_angles[:, j].max()
    old2_sub_lo, old2_sub_hi = old2_subset_angles[:, j].min(), old2_subset_angles[:, j].max()
    old2_full_lo, old2_full_hi = old2_full_angles[:, j].min(), old2_full_angles[:, j].max()
    comb_lo, comb_hi = combined_angles[:, j].min(), combined_angles[:, j].max()

    new_pct = 100.0 * (new_hi - new_lo) / full_range_deg
    comb_pct = 100.0 * (comb_hi - comb_lo) / full_range_deg

    print(f"{ck.JOINT_NAMES[j]:24s} "
          f"[{new_lo:6.1f},{new_hi:6.1f}]({new_pct:4.0f}%) "
          f"[{old2_sub_lo:6.1f},{old2_sub_hi:6.1f}] "
          f"[{old2_full_lo:6.1f},{old2_full_hi:6.1f}] "
          f"[{comb_lo:6.1f},{comb_hi:6.1f}]({comb_pct:4.0f}%) "
          f"{full_range_deg:10.1f}")
