"""Verify (numerically) whether elbow_pitch-origin-Y and shoulder_yaw-origin-Y
corrections have IDENTICAL sensitivity (Jacobian column) on end-effector
position, for arbitrary joint angles -- i.e. a true structural degeneracy,
not just a data-diversity-limited correlation.

Reasoning being checked: shoulder_yaw (joint index 2) rotates about the Y
axis. A translation correction inserted at joint 3's origin (elbow_pitch),
if it points along Y, sits exactly along shoulder_yaw's rotation axis --
and a rotation never moves points that lie on its own axis. So that
Y-component should be rotated by joint 2 to itself unchanged, making it
structurally identical (as a function of ALL joint angles) to the same
Y-correction placed at joint 2's own origin, one link earlier.
"""
import numpy as np
import calibrate_kinematics as ck

rng = np.random.default_rng(0)


def fk_with_extra_origin(angles_rad, joint_idx, delta_xyz_m):
    """Forward kinematics with an extra small translation delta inserted
    into JOINT_ORIGINS_M[joint_idx] only (all other params nominal)."""
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == joint_idx:
            origin = origin + delta_xyz_m
        R = ck.Rotation.from_rotvec(ck.JOINT_AXES[i] * angles_rad[i]).as_matrix()
        T_joint = ck.homogeneous_transform(R, origin)
        T = T @ T_joint
    return T


def numeric_jacobian_column(angles_rad, joint_idx, axis, h=1e-6):
    delta = np.zeros(3)
    delta[axis] = h
    Tp = fk_with_extra_origin(angles_rad, joint_idx, delta)
    Tm = fk_with_extra_origin(angles_rad, joint_idx, -delta)
    return (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)


print("Checking d(p_end)/d(elbow_pitch_origin_Y) vs d(p_end)/d(shoulder_yaw_origin_Y)")
print("across random joint configurations (should be IDENTICAL if the")
print("degeneracy is real, since shoulder_yaw's axis is Y):\n")

for trial in range(6):
    angles = np.zeros(7)
    for j in range(7):
        lo, hi = ck.JOINT_TICK_RANGES[j]
        tick = rng.uniform(lo + 100, hi - 100)
        angles[j] = (tick - ck.NOMINAL_ZERO_TICKS[j]) / ck.TICKS_PER_RADIAN

    j_elbow_y = numeric_jacobian_column(angles, 3, 1)   # elbow_pitch origin, Y axis
    j_shoulder_y = numeric_jacobian_column(angles, 2, 1)  # shoulder_yaw origin, Y axis

    print(f"trial {trial}: shoulder_yaw_angle={np.degrees(angles[2]):+7.1f}deg")
    print(f"  d(p_end)/d(elbow_pitch_Y)    = {np.round(j_elbow_y, 6)}")
    print(f"  d(p_end)/d(shoulder_yaw_Y)   = {np.round(j_shoulder_y, 6)}")
    print(f"  difference norm: {np.linalg.norm(j_elbow_y - j_shoulder_y):.2e}\n")

# Now check the X and Z components ARE distinguishable (different sensitivity)
print("\nFor contrast, checking X-origin sensitivity (should DIFFER between")
print("the two joints, since X is not shoulder_yaw's rotation axis):\n")
for trial in range(3):
    angles = np.zeros(7)
    for j in range(7):
        lo, hi = ck.JOINT_TICK_RANGES[j]
        tick = rng.uniform(lo + 100, hi - 100)
        angles[j] = (tick - ck.NOMINAL_ZERO_TICKS[j]) / ck.TICKS_PER_RADIAN

    j_elbow_x = numeric_jacobian_column(angles, 3, 0)
    j_shoulder_x = numeric_jacobian_column(angles, 2, 0)
    print(f"trial {trial}:")
    print(f"  d(p_end)/d(elbow_pitch_X)  = {np.round(j_elbow_x, 6)}")
    print(f"  d(p_end)/d(shoulder_yaw_X) = {np.round(j_shoulder_x, 6)}")
    print(f"  difference norm: {np.linalg.norm(j_elbow_x - j_shoulder_x):.2e}")
