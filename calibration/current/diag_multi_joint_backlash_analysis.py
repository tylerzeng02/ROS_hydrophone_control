"""Analyze the multi-joint backlash NDI capture (2026-07-30):
build/quick_calibration_test.csv, 28 poses = 7 joints x 4 poses each
(below, target-from-below, above, target-from-above), captured via the
generalized record_hand_poses.cpp + test_five_pose_ndi_capture.cpp
--quick-test (TARGET_POSES index 319-346).

For each joint's group of 4, compares pose 2 (target, arriving from
below) vs pose 4 (same target tick, arriving from above) using the NDI-
measured moving_relative_fixed position -- the actual backlash gap for
that joint, the same methodology already validated for wrist_pitch
(~4.2-4.7mm) and elbow_pitch (~10mm) individually.
"""
import csv
import numpy as np

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

n = len(rows)
print(f"Loaded {n} poses\n")

JOINT_NAMES = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow_pitch",
               "elbow_yaw", "wrist_pitch", "wrist_roll"]
POSES_PER_JOINT = 4

if n != 7 * POSES_PER_JOINT:
    print(f"WARNING: expected {7 * POSES_PER_JOINT} poses, got {n} -- "
          f"some poses may have been skipped. Proceeding with what's here, "
          f"grouping strictly in sets of {POSES_PER_JOINT}.\n")

print(f"{'joint':<16}{'gap_mm':>8}  {'dx':>7}{'dy':>7}{'dz':>7}   "
      f"{'tick2':>7}{'tick4':>7}{'diff':>6}   {'err2':>6}{'err4':>6}  {'vis2/vis4'}")

results = []
n_groups = n // POSES_PER_JOINT
for g in range(n_groups):
    group = rows[g * POSES_PER_JOINT:(g + 1) * POSES_PER_JOINT]
    if len(group) < POSES_PER_JOINT:
        break
    pose2 = group[1]  # target, from below
    pose4 = group[3]  # target, from above (auto-driven to pose2's tick)

    p2 = np.array([float(pose2["moving_relative_fixed_tx_mm"]),
                   float(pose2["moving_relative_fixed_ty_mm"]),
                   float(pose2["moving_relative_fixed_tz_mm"])])
    p4 = np.array([float(pose4["moving_relative_fixed_tx_mm"]),
                   float(pose4["moving_relative_fixed_ty_mm"]),
                   float(pose4["moving_relative_fixed_tz_mm"])])
    gap = p4 - p2
    gap_mag = np.linalg.norm(gap)

    joint_name = JOINT_NAMES[g] if g < len(JOINT_NAMES) else f"joint_group_{g}"
    tick2 = float(pose2[f"actual_tick_{g}"]) if g < 7 else float("nan")
    tick4 = float(pose4[f"actual_tick_{g}"]) if g < 7 else float("nan")
    err2 = float(pose2["moving_relative_fixed_error"])
    err4 = float(pose4["moving_relative_fixed_error"])
    vis2 = pose2["moving_relative_fixed_visible_markers"]
    vis4 = pose4["moving_relative_fixed_visible_markers"]

    results.append((joint_name, gap_mag, gap, tick2, tick4, err2, err4, vis2, vis4))
    print(f"{joint_name:<16}{gap_mag:8.2f}  {gap[0]:7.2f}{gap[1]:7.2f}{gap[2]:7.2f}   "
          f"{tick2:7.0f}{tick4:7.0f}{tick4-tick2:6.0f}   "
          f"{err2:6.3f}{err4:6.3f}  {vis2}/{vis4}")

print("\nSorted by gap magnitude (largest first):")
for r in sorted(results, key=lambda r: -r[1]):
    print(f"  {r[0]:<16} {r[1]:6.2f}mm")

print(f"\nReference: wrist_pitch confirmed ~4.2-4.7mm (two independent runs), "
      f"elbow_pitch confirmed ~10.0mm. Anything comparably above the ~0.5-1mm "
      f"repeatability/NDI-noise floor here indicates real backlash at that joint.")
