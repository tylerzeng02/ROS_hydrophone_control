import csv
import numpy as np
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/recorded_hand_poses.csv"

ticks = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ticks.append([int(row[f"tick_{i}"]) for i in range(7)])
ticks = np.array(ticks)
n = len(ticks)

zero_ticks = np.array(ck.NOMINAL_ZERO_TICKS)
angles_rad = (ticks - zero_ticks) / ck.TICKS_PER_RADIAN

rotations = []
for i in range(n):
    T = ck.forward_kinematics(angles_rad[i])
    rotations.append(Rotation.from_matrix(T[:3, :3]))

all_angles = []
for i in range(n):
    rel = rotations[i].inv() * Rotation.concatenate(rotations[i + 1:]) if i + 1 < n else None
    if rel is not None:
        all_angles.extend((rel.magnitude() * 180.0 / np.pi).tolist())
all_angles = np.array(all_angles)

nn = np.full(n, np.inf)
for i in range(n):
    rel = rotations[i].inv() * Rotation.concatenate(rotations)
    ang = rel.magnitude() * 180.0 / np.pi
    ang[i] = np.inf
    nn[i] = ang.min()

print(f"Pairwise orientation distance across all {len(all_angles)} pose pairs (degrees):")
print(f"  mean = {all_angles.mean():.1f}, median = {np.median(all_angles):.1f}, "
      f"min = {all_angles.min():.1f}, max = {all_angles.max():.1f}")
print()
print(f"Nearest-neighbor orientation distance per pose (degrees):")
print(f"  mean = {nn.mean():.1f}, median = {np.median(nn):.1f}, "
      f"min = {nn.min():.1f}, max = {nn.max():.1f}")
print(f"\nFor reference, previous 190-pose dataset had: pairwise mean=56.4, nearest-neighbor mean=6.8")
