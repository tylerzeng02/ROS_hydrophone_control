import csv
import numpy as np
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/recorded_hand_poses_combined.csv"

ticks = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ticks.append([int(row[f"tick_{i}"]) for i in range(7)])
ticks = np.array(ticks)
n = len(ticks)
print(f"n poses: {n}\n")

zero_ticks = np.array(ck.NOMINAL_ZERO_TICKS)
angles_rad = (ticks - zero_ticks) / ck.TICKS_PER_RADIAN

print(f"{'joint':24s} {'range covered (deg)':>20s} {'full safe range (deg)':>22s} {'% of full range':>16s}")
for j in range(ck.N_JOINTS):
    lo_tick, hi_tick = ck.JOINT_TICK_RANGES[j]
    full_range_deg = np.degrees((hi_tick - lo_tick) / ck.TICKS_PER_RADIAN)
    joint_deg = np.degrees(angles_rad[:, j])
    covered_deg = joint_deg.max() - joint_deg.min()
    pct = 100.0 * covered_deg / full_range_deg
    print(f"{ck.JOINT_NAMES[j]:24s} {covered_deg:20.1f} {full_range_deg:22.1f} {pct:15.1f}%")
