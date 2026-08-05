import numpy as np
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

new_ticks = np.array([
    [287, 852, 1265, 3034, 1920, 3345, 1710],
    [287, 852, 1266, 3088, 1765, 2502, 1493],
    [285, 852, 1265, 3087, 2263, 2477, 998],
    [286, 852, 1265, 3051, 2264, 2522, 604],
    [286, 852, 1265, 3055, 2262, 2540, 337],
    [337, 851, 1265, 3271, 2022, 1921, 3753],
    [337, 852, 1265, 3273, 2019, 1781, 3753],
    [340, 852, 1265, 3275, 1607, 1709, 3557],
    [339, 852, 1265, 3274, 1511, 1583, 3557],
    [339, 852, 1265, 3274, 1296, 1513, 3554],
    [340, 852, 1265, 3274, 1016, 1903, 3319],
    [354, 852, 1265, 3273, 2342, 2972, 1558],
    [351, 852, 1265, 3274, 2341, 3344, 1672],
])
n = len(new_ticks)

zero_ticks = np.array(ck.NOMINAL_ZERO_TICKS)
angles_rad = (new_ticks - zero_ticks) / ck.TICKS_PER_RADIAN

rotations = [Rotation.from_matrix(ck.forward_kinematics(angles_rad[i])[:3, :3]) for i in range(n)]

all_angles = []
for i in range(n):
    if i + 1 < n:
        rel = rotations[i].inv() * Rotation.concatenate(rotations[i + 1:])
        all_angles.extend((rel.magnitude() * 180.0 / np.pi).tolist())
all_angles = np.array(all_angles)

nn = np.full(n, np.inf)
for i in range(n):
    rel = rotations[i].inv() * Rotation.concatenate(rotations)
    ang = rel.magnitude() * 180.0 / np.pi
    ang[i] = np.inf
    nn[i] = ang.min()

print(f"n new poses: {n}\n")
print(f"Pairwise orientation distance among these 13 new poses (degrees):")
print(f"  mean = {all_angles.mean():.1f}, median = {np.median(all_angles):.1f}, "
      f"min = {all_angles.min():.1f}, max = {all_angles.max():.1f}\n")
print(f"Nearest-neighbor orientation distance (degrees):")
print(f"  mean = {nn.mean():.1f}, median = {np.median(nn):.1f}, min = {nn.min():.1f}, max = {nn.max():.1f}")
print(f"\nFor reference, the previous 287-pose dataset had nearest-neighbor mean ~8.3deg")
