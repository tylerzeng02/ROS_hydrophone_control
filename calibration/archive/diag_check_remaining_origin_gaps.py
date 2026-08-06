"""General rule (proven for the elbow_pitch/shoulder_yaw case): joint (i+1)'s
origin-component that lies ALONG joint i's own rotation axis is EXACTLY
degenerate with joint i's origin component in that same direction, for
every possible joint angle -- because a rotation never moves a vector on
its own axis.

This predicts a whole chain of pairwise degeneracies we haven't checked:
  joint1(shoulder_pitch)-origin-Z <-> base_xyz-Z          (already known)
  joint2(shoulder_yaw)-origin-X   <-> joint1(shoulder_pitch)-origin-X
  joint3(elbow_pitch)-origin-Y    <-> joint2(shoulder_yaw)-origin-Y   (already found/fixed)
  joint4(elbow_yaw)-origin-X      <-> joint3(elbow_pitch)-origin-X
  joint5(wrist_pitch)-origin-Y    <-> joint4(elbow_yaw)-origin-Y
  joint6(wrist_roll)-origin-X     <-> joint5(wrist_pitch)-origin-X    (moot, wrist_roll origin already excluded)

Since shoulder_yaw-X, elbow_pitch-X/Y/Z, and wrist_pitch-X/Y/Z are ALREADY
free in the current model, this predicts shoulder_pitch's X and elbow_yaw's
X and Y would each collide with an existing free parameter if added --
leaving only shoulder_pitch-Y and elbow_yaw-Z as genuinely new,
independent, safely-addable directions. Verify numerically before trusting
this.
"""
import numpy as np
import calibrate_kinematics as ck

rng = np.random.default_rng(7)


def fk_with_two_origin_deltas(angles_rad, idx_a, delta_a, idx_b, delta_b):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == idx_a:
            origin = origin + delta_a
        if i == idx_b:
            origin = origin + delta_b
        R = ck.Rotation.from_rotvec(ck.JOINT_AXES[i] * angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, origin)
    return T


def col(angles, joint_idx, axis, h=1e-6):
    d = np.zeros(3); d[axis] = h
    Tp = fk_with_two_origin_deltas(angles, joint_idx, d, -1, np.zeros(3))
    Tm = fk_with_two_origin_deltas(angles, joint_idx, -d, -1, np.zeros(3))
    return (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)


def random_angles():
    angles = np.zeros(7)
    for j in range(7):
        lo, hi = ck.JOINT_TICK_RANGES[j]
        tick = rng.uniform(lo + 100, hi - 100)
        angles[j] = (tick - ck.NOMINAL_ZERO_TICKS[j]) / ck.TICKS_PER_RADIAN
    return angles


AXIS_NAME = ["X", "Y", "Z"]
NAMES = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow_pitch",
         "elbow_yaw", "wrist_pitch", "wrist_roll"]

pairs_to_check = [
    (2, 1, "shoulder_yaw-origin-X vs shoulder_pitch-origin-X (predicted DEGENERATE)"),
    (4, 3, "elbow_yaw-origin-X vs elbow_pitch-origin-X (predicted DEGENERATE)"),
    (5, 4, "wrist_pitch-origin-Y vs elbow_yaw-origin-Y (predicted DEGENERATE)"),
]

for joint_hi, joint_lo, label in pairs_to_check:
    print(f"--- {label} ---")
    axis = 0 if "X" in label.split(" vs ")[0] else (1 if "-Y " in label or label.count("-Y") else 2)
    # figure out which axis component from the label text robustly:
    comp_name = label.split("-origin-")[1].split(" ")[0]
    axis = AXIS_NAME.index(comp_name)
    for _ in range(4):
        angles = random_angles()
        c_hi = col(angles, joint_hi, axis)
        c_lo = col(angles, joint_lo, axis)
        diff = np.linalg.norm(c_hi - c_lo)
        print(f"  {NAMES[joint_hi]}-{comp_name}: {np.round(c_hi,5)}   "
              f"{NAMES[joint_lo]}-{comp_name}: {np.round(c_lo,5)}   diff={diff:.2e}")
    print()

print("--- Sanity check: shoulder_pitch-origin-Y vs shoulder_yaw-origin-X (should be INDEPENDENT) ---")
for _ in range(3):
    angles = random_angles()
    c_a = col(angles, 1, 1)  # shoulder_pitch Y
    c_b = col(angles, 2, 0)  # shoulder_yaw X
    dot = np.dot(c_a, c_b) / (np.linalg.norm(c_a) * np.linalg.norm(c_b))
    print(f"  cos(angle between columns) = {dot:.3f}  (near +-1 = degenerate, near 0 = independent)")

print("\n--- Sanity check: elbow_yaw-origin-Z vs elbow_pitch-origin-X and wrist_pitch-origin-Y (should be INDEPENDENT) ---")
for _ in range(3):
    angles = random_angles()
    c_ey_z = col(angles, 4, 2)
    c_ep_x = col(angles, 3, 0)
    c_wp_y = col(angles, 5, 1)
    dot1 = np.dot(c_ey_z, c_ep_x) / (np.linalg.norm(c_ey_z) * np.linalg.norm(c_ep_x))
    dot2 = np.dot(c_ey_z, c_wp_y) / (np.linalg.norm(c_ey_z) * np.linalg.norm(c_wp_y))
    print(f"  cos(elbow_yaw-Z, elbow_pitch-X) = {dot1:.3f}   cos(elbow_yaw-Z, wrist_pitch-Y) = {dot2:.3f}")
