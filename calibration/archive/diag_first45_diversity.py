import csv
import numpy as np
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

subset = rows[:45]
angles = np.array([[float(r[f"actual_rad_{i}"]) for i in range(ck.N_JOINTS)] for r in subset])

print(f"Per-joint coverage of first 45 poses vs full safe range:")
for j in range(ck.N_JOINTS):
    lo_tick, hi_tick = ck.JOINT_TICK_RANGES[j]
    full_range_deg = np.degrees((hi_tick - lo_tick) / ck.TICKS_PER_RADIAN)
    joint_deg = np.degrees(angles[:, j])
    covered_deg = joint_deg.max() - joint_deg.min()
    pct = 100.0 * covered_deg / full_range_deg
    print(f"  {ck.JOINT_NAMES[j]:24s} {covered_deg:8.1f} deg / {full_range_deg:6.1f} deg full  ({pct:5.1f}%)")
