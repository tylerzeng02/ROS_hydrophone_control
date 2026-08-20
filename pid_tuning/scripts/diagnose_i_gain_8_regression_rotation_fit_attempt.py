"""Robust alternative to diagnose_i_gain_8_regression_offset_refit_attempt.py, after that
approach hit a real problem: the 60-param model isn't uniquely identified
from the fit data (good in-sample RMS, 1.41mm, but a fresh refit converges
to a DIFFERENT local optimum than whatever produced
i_gain_test_10pose_with_predicted_positions.csv's recorded predictions --
confirmed by cross-checking my refit's own predictions against the file's
recorded ones and finding 6.7-16.7mm disagreement even after fixing the
earlier wrong-dataset mistake). Re-deriving that exact original fit isn't
reliable, so this script sidesteps it entirely.

Approach: use ONLY the two already-collected, real, DIRECT NDI
measurements for the same 9 points -- pid_tuning/data/i_gain_0_baseline_accuracy_results.csv
(I=0) and pid_tuning/data/i_gain_8_regression_discovery_9pose_results.csv (I=8)
-- nothing about the fitted correction model is needed at all for this
part. For each point, delta_i = actual_position_I8 - actual_position_I0
is a real, measured, model-independent 3D shift caused purely by fixing
shoulder_pitch's tick-settling behavior (everything else about the setup
was identical between the two runs).

If that shift is well-explained by a pure shoulder_pitch ROTATION (which
it should be, since shoulder_pitch's true angle is the only thing that
changed), it must be proportional to the standard robot-Jacobian column
for that joint: axis_1 x (p_ee - p_1), evaluated at each point's own
configuration via the arm's own (verified, already-deployed) forward
kinematics -- copied from
calibration/current/verify_pose_dependent_port.py, which was already
independently checked against pose_dependent_correction.cpp to 8 decimal
places. This uses the STATIC, deployed axis/origin geometry only (not the
large, non-uniquely-fit correction model), so there's no local-optima
ambiguity here.

Fits a single scalar (shoulder_pitch angle correction, radians) via
closed-form weighted least squares against these 9 real 3D shifts, then
reports the equivalent zeroTick delta.

Run from calibration/current/:
    python3 diagnose_i_gain_8_regression_rotation_fit_attempt.py
"""

import csv

import numpy as np

import calibrate_kinematics as ck

SHOULDER_PITCH = 1

# FK, copied from verify_pose_dependent_port.py (already independently
# verified against pose_dependent_correction.cpp's C++ port to 8 decimal
# places -- see that script's own header comment).
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


def shoulder_pitch_jacobian_column(joint_angles):
    """Standard revolute-joint Jacobian column for joint SHOULDER_PITCH,
    evaluated at this configuration: axis x (p_ee - p_joint). This
    approximates p_ee as the wrist-frame chain-end position (matching
    fk_frames' own convention, same as the gravity term) -- close enough
    to the true marker position for a small-rotation linear approximation,
    and NOT trying to be more precise than that (this is deliberately a
    simple, robust check, not a full recalibration)."""
    pee, pl, al = fk_frames(joint_angles)
    lever = pee - pl[SHOULDER_PITCH]
    return np.cross(al[SHOULDER_PITCH], lever)  # meters, per radian


# Load both already-collected real measurement sets, matched by test_id.
with open("../data/i_gain_0_baseline_accuracy_results.csv") as f:
    base_rows = {r["test_id"]: r for r in csv.DictReader(f)}
with open("../data/i_gain_8_regression_discovery_9pose_results.csv") as f:
    new_rows = {r["test_id"]: r for r in csv.DictReader(f)}

common_ids = sorted(set(base_rows) & set(new_rows), key=int)
print(f"Matched {len(common_ids)} points present in both runs: {common_ids}\n")

deltas_mm = []
jac_cols_m = []
for tid in common_ids:
    b, n = base_rows[tid], new_rows[tid]
    actual_i0 = np.array([float(b["actual_x_mm"]), float(b["actual_y_mm"]), float(b["actual_z_mm"])])
    actual_i8 = np.array([float(n["actual_x_mm"]), float(n["actual_y_mm"]), float(n["actual_z_mm"])])
    delta = actual_i8 - actual_i0
    deltas_mm.append(delta)

    # Use the I=8 run's achieved ticks as the reference configuration --
    # it's the one closer to the arm's true commanded/intended pose (the
    # whole point of the I-gain fix).
    achieved_ticks = np.array([float(n[f"achieved_tick_{i}"]) for i in range(7)])
    ar = (achieved_ticks - np.array(ck.NOMINAL_ZERO_TICKS)) / ck.TICKS_PER_RADIAN
    jac_col = shoulder_pitch_jacobian_column(ar) * 1000.0  # m/rad -> mm/rad
    jac_cols_m.append(jac_col)

    print(f"  test_id {tid}: delta(I8-I0)={delta} mm, |delta|={np.linalg.norm(delta):.2f}mm, "
          f"jacobian_col={jac_col} mm/rad")

deltas_mm = np.array(deltas_mm)
jac_cols_m = np.array(jac_cols_m)

# Closed-form scalar least squares: minimize sum ||delta_i - theta*J_i||^2
# => theta = sum(delta_i . J_i) / sum(J_i . J_i)
numerator = np.sum(deltas_mm * jac_cols_m)
denominator = np.sum(jac_cols_m * jac_cols_m)
theta = numerator / denominator

print(f"\nFitted shoulder_pitch angle correction: {theta:.6f} rad ({np.degrees(theta):+.4f} deg)")

# How well does a pure-rotation explanation fit the observed deltas?
predicted_deltas = theta * jac_cols_m
residuals = deltas_mm - predicted_deltas
residual_norms = np.linalg.norm(residuals, axis=1)
delta_norms = np.linalg.norm(deltas_mm, axis=1)
print(f"Observed |delta|: mean={delta_norms.mean():.2f}mm, range=[{delta_norms.min():.2f}, {delta_norms.max():.2f}]mm")
print(f"Residual after removing the fitted pure-rotation component: "
      f"mean={residual_norms.mean():.2f}mm, max={residual_norms.max():.2f}mm")
print("(if the residual is small relative to the observed delta, the pure-shoulder_pitch-rotation "
      "explanation is a good fit -- if it's comparably large, something else changed too and this "
      "correction alone won't fully explain the I=0->I=8 shift.)\n")

# jointCalibrations[1] convention: radians = direction*scale*(tick-zeroTick)/TICKS_PER_RADIAN,
# direction=+1 for every joint on this arm. We want the NEW true angle
# (after I=8 fix) to equal the OLD model's implied angle PLUS theta, i.e.
# shift zeroTick so ticksToRadians(achieved_tick) increases by theta at a
# fixed tick value -- solve zeroTick_new from:
#   theta = scale*(zeroTick_old - zeroTick_new)/TICKS_PER_RADIAN
#   => zeroTick_new = zeroTick_old - theta*TICKS_PER_RADIAN/scale
sp_scale = 1.001931  # from src/robot_calibration.cpp jointCalibrations[1].scale (deployed value)
zero_tick_delta = -theta * ck.TICKS_PER_RADIAN / sp_scale
print(f"Implied change to robot_calibration.cpp jointCalibrations[1] (shoulder_pitch) zeroTick: "
      f"{zero_tick_delta:+.2f} ticks")
