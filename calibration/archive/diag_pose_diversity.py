import numpy as np
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)

print(f"n poses: {n}\n")
print(f"{'joint':24s} {'range covered (deg)':>20s} {'full safe range (deg)':>22s} {'% of full range':>16s}")

for j in range(ck.N_JOINTS):
    lo_tick, hi_tick = ck.JOINT_TICK_RANGES[j]
    full_range_rad = (hi_tick - lo_tick) / ck.TICKS_PER_RADIAN
    full_range_deg = np.degrees(full_range_rad)

    joint_angles_deg = np.degrees(angles[:, j])
    covered_deg = joint_angles_deg.max() - joint_angles_deg.min()

    pct = 100.0 * covered_deg / full_range_deg
    print(f"{ck.JOINT_NAMES[j]:24s} {covered_deg:20.1f} {full_range_deg:22.1f} {pct:15.1f}%")

# Also compare against the full 174-pose original dataset for reference
print("\nFor reference, same check on the original 174-pose dataset:")
angles_full, _, _ = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)
for j in range(ck.N_JOINTS):
    lo_tick, hi_tick = ck.JOINT_TICK_RANGES[j]
    full_range_deg = np.degrees((hi_tick - lo_tick) / ck.TICKS_PER_RADIAN)
    joint_angles_deg = np.degrees(angles_full[:, j])
    covered_deg = joint_angles_deg.max() - joint_angles_deg.min()
    pct = 100.0 * covered_deg / full_range_deg
    print(f"{ck.JOINT_NAMES[j]:24s} {covered_deg:20.1f} {full_range_deg:22.1f} {pct:15.1f}%")
