"""Cross-check for src/pose_dependent_correction.cpp.

Computes, in Python using the SAME math and fitted coefficients as
final_deployment_fit.py's predict() (the fourier + coupling + gravity
terms only -- i.e. cag - ca, the "correction" the C++ port also computes),
at a handful of test joint configurations. Print these numbers and diff
them by eye (or pipe both this and the matching C++ test program's output
through `diff`) against
ros/src/cyton_hardware/test/print_pose_dependent_correction.cpp's output
for the same configurations -- they should agree to ~1e-6 or better.

Run: python3 verify_pose_dependent_port.py
"""

import numpy as np

SHOULDER_ROLL, SHOULDER_PITCH, SHOULDER_YAW = 0, 1, 2
ELBOW_PITCH, ELBOW_YAW, WRIST_PITCH, WRIST_ROLL = 3, 4, 5, 6

JOINT_NAMES = [
    "shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow_pitch",
    "elbow_yaw", "wrist_pitch", "wrist_roll",
]

# Deployed axis/origin, copied verbatim from
# references/cyton_gamma_1500_trac_ik.urdf -- same source the C++ port
# copied from. Kept independent of calibrate_kinematics.py's own
# JOINT_AXES/JOINT_ORIGINS_M + tilt/origin correction machinery
# deliberately, so this check isn't just re-running the same code path
# twice.
JOINT_ORIGIN = np.array([
    [0.0, 0.0, 0.05315],
    [0.0205, 0.0, 0.12435],
    [-0.02478414, -0.0205, 0.1308452],
    [0.01656849, 0.02722018, 0.11356304],
    [-0.0171, -0.018, 0.09746],
    [0.02765348, 0.01273746, 0.07244612],
    [-0.026255, 0.0, 0.051425],
])
JOINT_AXIS_RAW = np.array([
    [0.013792, 0.014877, 0.999794],
    [0.998997, -0.043573, -0.010357],
    [-0.027678, -0.999437, 0.018973],
    [0.999677, -0.024005, 0.008304],
    [0.0, -1.0, 0.0],
    [0.999044, 0.042957, 0.008105],
    [-0.006451, -0.018572, 0.999807],
])
JOINT_AXIS = JOINT_AXIS_RAW / np.linalg.norm(JOINT_AXIS_RAW, axis=1, keepdims=True)

COUPLE_TERMS = [
    (SHOULDER_ROLL, SHOULDER_YAW, SHOULDER_YAW, 0.002638),
    (SHOULDER_YAW, ELBOW_YAW, ELBOW_YAW, -0.000372),
    (SHOULDER_PITCH, ELBOW_PITCH, ELBOW_PITCH, 0.002293),
]
FOURIER_A, FOURIER_B = -0.002404, 0.012840
GRAVITY_COEFF = np.array([0.001381, 0.002937, 0.000380, 0.011437, -0.001249, 0.000190, 0.000000])


def rodrigues(axis, angle):
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    x, y, z = axis
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def fk_frames(joint_angles):
    R = np.eye(3)
    p = np.zeros(3)
    pl, al = [], []
    for i in range(7):
        origin = JOINT_ORIGIN[i]
        p_before = R @ origin + p
        a_in_base = R @ JOINT_AXIS[i]
        pl.append(p_before)
        al.append(a_in_base)
        R = R @ rodrigues(JOINT_AXIS[i], joint_angles[i])
        p = p_before
    return p, pl, al  # p = chain end position (pre virtual_endeffector offset)


def gravity_moment_arms(joint_angles):
    pee, pl, al = fk_frames(joint_angles)
    g = np.zeros(7)
    gdir = np.array([0.0, 0.0, -1.0])
    for i in range(7):
        lever = pee - pl[i]
        g[i] = np.dot(np.cross(lever, gdir), al[i])
    return g


def compute_correction(joint_angles):
    correction = np.zeros(7)
    sp = joint_angles[SHOULDER_PITCH]
    correction[SHOULDER_PITCH] += FOURIER_A * np.sin(sp) + FOURIER_B * np.cos(sp)
    for i, j, target, coeff in COUPLE_TERMS:
        if target == ELBOW_YAW:
            continue
        correction[target] += coeff * joint_angles[i] * joint_angles[j]
    angles_for_gravity = joint_angles + correction
    g = gravity_moment_arms(angles_for_gravity)
    for i in range(7):
        if i == ELBOW_YAW:
            continue
        correction[i] += GRAVITY_COEFF[i] * g[i]
    correction[ELBOW_YAW] = 0.0
    return correction


TEST_CONFIGS = [
    np.zeros(7),
    np.array([0.3, -0.2, 0.15, -0.5, 0.02, 0.4, -0.1]),
    np.array([-0.8, 0.6, -1.0, 1.2, -0.02, -0.9, 0.7]),
    np.array([1.5, -1.0, 1.5, -1.8, 0.0, 1.7, -1.5]),
    np.array([0.05, 0.05, -0.05, 0.1, 0.01, -0.05, 0.05]),
]

if __name__ == "__main__":
    for idx, cfg in enumerate(TEST_CONFIGS):
        correction = compute_correction(cfg)
        print(f"config {idx}: angles={np.array2string(cfg, precision=4)}")
        for name, c in zip(JOINT_NAMES, correction):
            print(f"    {name:15s} correction_rad={c: .8f}")
