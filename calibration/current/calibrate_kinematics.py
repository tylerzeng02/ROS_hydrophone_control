"""Kinematic calibration for the Cyton Gamma 1500 arm using NDI Polaris
Spectra data collected by calibration/collection/ndi_capture_and_validate.cpp.

Fits 19 parameters via nonlinear least squares against measured
end-effector poses:
  - joint_offsets (7, rad): per-joint zero-offset correction.
  - tool_xyz/tool_rpy (6): correction to the tool frame (virtual_endeffector
    -> moving marker).
  - base_xyz/base_rpy (6): fixed marker -> URDF base_link transform. Not
    regularized, unlike the other two: the marker's position relative to
    the robot base has no known nominal value, so there is nothing to pull
    it toward.
"""

import argparse
import csv
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# Kinematic chain from references/cyton_gamma_1500_trac_ik.urdf
# (base_link -> virtual_endeffector). Every joint <origin> has rpy="0 0 0",
# so each joint's transform reduces to Translate(origin_xyz) then
# Rotate(axis, angle) instead of a general rotation+translation. If the
# URDF ever gains a nonzero <origin rpy>, forward_kinematics() below stops
# matching it and needs a general rotation term added.

JOINT_NAMES = [
    "shoulder_roll_joint",
    "shoulder_pitch_joint",
    "shoulder_yaw_joint",
    "elbow_pitch_joint",
    "elbow_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
]

N_JOINTS = 7

# Hand-transcribed from the URDF, not loaded from it. JOINT_ORIGINS_M,
# JOINT_AXES, NOMINAL_ZERO_TICKS, and JOINT_TICK_RANGES below are all
# positionally aligned to JOINT_NAMES. If the URDF or
# src/robot_calibration.cpp's jointCalibrations table ever changes, these
# need a manual update; otherwise the fit silently calibrates against
# stale geometry with no error raised.
JOINT_ORIGINS_M = np.array([
    [0.0, 0.0, 0.05315],
    [0.0205, 0.0, 0.12435],
    [-0.0215, -0.0205, 0.1255],
    [0.018, 0.0206, 0.1158],
    [-0.0171, -0.018, 0.09746],
    [0.02626, 0.018, 0.0718],
    [-0.026255, 0.0, 0.051425],
])

JOINT_AXES = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

TOOL_ORIGIN_NOMINAL_M = np.array([-0.002316, 0.0079, 0.079425])  # virtual_endeffector_joint

# jointCalibrations from src/robot_calibration.cpp (id, zeroTick, direction,
# minTick, maxTick). direction is +1 for every joint on this arm. If a
# future joint needs direction=-1, TICKS_PER_RADIAN's sign convention here
# needs revisiting.
NOMINAL_ZERO_TICKS = [2048, 2048, 2066, 2108, 2078, 2048, 2048]
JOINT_TICK_RANGES = [
    (376, 3772),
    (851, 3231),
    (912, 3327),
    (829, 3274),
    (952, 3245),
    (751, 3344),
    (335, 3755),
]
TICKS_PER_RADIAN = 4096.0 / (2.0 * np.pi)


# Parameter vector layout (19 total). Order must match pack_params/
# unpack_params/build_bounds exactly. Reordering CalibParams' fields
# without updating all three silently scrambles which bound/regularization
# weight applies to which physical quantity.

N_PARAMS = 19


@dataclass
class CalibParams:
    joint_offsets: np.ndarray   # (7,) radians, added to measured joint angle
    tool_xyz: np.ndarray        # (3,) meters, correction to TOOL_ORIGIN_NOMINAL_M
    tool_rpy: np.ndarray        # (3,) radians, correction to tool orientation
    # base_xyz / base_rpy: pose of base_link expressed in the fixed
    # marker's frame, not the reverse. Flipping this direction produces a
    # wrong pose that still runs without error. See build_base_transform()
    # and predict_relative_pose()'s left-multiply order, which depends on
    # this exact direction.
    base_xyz: np.ndarray        # (3,) meters
    base_rpy: np.ndarray        # (3,) radians


def unpack_params(v: np.ndarray) -> CalibParams:
    """Splits the flat 19-element optimizer vector into named fields.

    Args:
        v: Parameter vector in pack_params()'s layout.

    Returns:
        The corresponding CalibParams.
    """
    return CalibParams(
        joint_offsets=v[0:7],
        tool_xyz=v[7:10],
        tool_rpy=v[10:13],
        base_xyz=v[13:16],
        base_rpy=v[16:19],
    )


def pack_params(
    joint_offsets: np.ndarray,
    tool_xyz: np.ndarray,
    tool_rpy: np.ndarray,
    base_xyz: np.ndarray,
    base_rpy: np.ndarray,
) -> np.ndarray:
    """Inverse of unpack_params(): flattens named fields into the vector
    least_squares() actually optimizes.

    Args:
        joint_offsets: (7,) radians.
        tool_xyz: (3,) meters.
        tool_rpy: (3,) radians.
        base_xyz: (3,) meters.
        base_rpy: (3,) radians.

    Returns:
        (19,) flat parameter vector.
    """
    return np.concatenate([joint_offsets, tool_xyz, tool_rpy, base_xyz, base_rpy])


# Forward kinematics

def homogeneous_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assembles a 4x4 homogeneous transform from a 3x3 rotation and a
    translation vector.

    Args:
        rotation_matrix: (3, 3) rotation.
        translation: (3,) translation.

    Returns:
        (4, 4) homogeneous transform.
    """
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def forward_kinematics(joint_angles_rad: np.ndarray) -> np.ndarray:
    """4x4 transform from base_link to the nominal virtual_endeffector frame.
    Tool-frame correction is not included; compose with build_tool_transform
    separately.

    Relies on every URDF joint <origin> having rpy="0 0 0" (see the comment
    above JOINT_NAMES). Each joint's local transform is built as a pure
    translation followed by a rotation about JOINT_AXES, not a general
    rotation+translation.

    Args:
        joint_angles_rad: (7,) measured or corrected joint angles, in
            JOINT_NAMES order.

    Returns:
        (4, 4) base_link -> virtual_endeffector transform.
    """
    T = np.eye(4)
    for i in range(N_JOINTS):
        joint_rotation = Rotation.from_rotvec(JOINT_AXES[i] * joint_angles_rad[i]).as_matrix()
        T_joint = homogeneous_transform(joint_rotation, JOINT_ORIGINS_M[i])
        T = T @ T_joint
    return T


def build_tool_transform(tool_xyz: np.ndarray, tool_rpy: np.ndarray) -> np.ndarray:
    """virtual_endeffector -> moving-marker transform, nominal origin plus
    the fitted correction.

    Args:
        tool_xyz: (3,) meters, correction added to TOOL_ORIGIN_NOMINAL_M.
        tool_rpy: (3,) radians, tool orientation correction.

    Returns:
        (4, 4) transform.
    """
    R = Rotation.from_euler("xyz", tool_rpy).as_matrix()
    return homogeneous_transform(R, TOOL_ORIGIN_NOMINAL_M + tool_xyz)


def build_base_transform(base_xyz: np.ndarray, base_rpy: np.ndarray) -> np.ndarray:
    """base_link -> fixed-marker transform, not the reverse: this is the
    direction that cancels base_link when left-multiplied onto
    T_fk @ T_tool in predict_relative_pose(). Flipping this direction does
    not raise an error; it fits a self-consistent but physically wrong
    base pose.

    Args:
        base_xyz: (3,) meters.
        base_rpy: (3,) radians.

    Returns:
        (4, 4) transform.
    """
    R = Rotation.from_euler("xyz", base_rpy).as_matrix()
    return homogeneous_transform(R, base_xyz)


def predict_relative_pose(measured_angles_rad: np.ndarray, params: CalibParams) -> np.ndarray:
    """Predicted moving-marker pose in the fixed marker's frame. Directly
    comparable to the "moving_relative_fixed" columns
    ndi_capture_and_validate.cpp writes, since that computation already
    removes the tracker's own arbitrary reference frame.

    Args:
        measured_angles_rad: (7,) raw measured joint angles, before
            joint_offsets are applied (this function applies them).
        params: Current parameter estimate.

    Returns:
        (4, 4) predicted fixed-marker -> moving-marker transform.
    """
    corrected_angles = measured_angles_rad + params.joint_offsets
    T_fk = forward_kinematics(corrected_angles)
    T_tool = build_tool_transform(params.tool_xyz, params.tool_rpy)
    T_base = build_base_transform(params.base_xyz, params.base_rpy)
    return T_base @ T_fk @ T_tool

# Bound sizing encodes a physical prior. joint_offset/tool bounds are tight
# because those corrections should only ever be small perturbations from
# the nominal URDF values; an offset that large would mean the URDF itself
# is wrong, not that this joint needs an 8-degree fudge. base_xyz/base_rpy
# bounds are room-scale and effectively unconstrained, because the fixed
# marker's position relative to base_link has no nominal value to bound
# around. Loosening the joint/tool bounds without also strengthening their
# regularization risks the optimizer explaining a base misalignment away
# with implausible per-joint offsets instead.
JOINT_OFFSET_BOUND_RAD = np.radians(8.0)
TOOL_XYZ_BOUND_M = 0.01
TOOL_RPY_BOUND_RAD = np.radians(10.0)
BASE_XYZ_BOUND_M = 2.0
BASE_RPY_BOUND_RAD = np.pi


def build_bounds():
    """Builds the (lower, upper) bound arrays for least_squares(), in
    pack_params() order. run_calibration() widens these further if the
    closed-form initial base guess itself falls outside them, so these
    are a starting prior, not a hard physical limit.

    Returns:
        Tuple of (lower, upper), each (19,).
    """
    lower = np.concatenate([
        -JOINT_OFFSET_BOUND_RAD * np.ones(7),
        -TOOL_XYZ_BOUND_M * np.ones(3),
        -TOOL_RPY_BOUND_RAD * np.ones(3),
        -BASE_XYZ_BOUND_M * np.ones(3),
        -BASE_RPY_BOUND_RAD * np.ones(3),
    ])
    upper = -lower
    return lower, upper


def initial_base_guess(
    first_measured_angles: np.ndarray,
    first_measured_pos_mm: np.ndarray,
    first_measured_quat_xyzw: np.ndarray,
):
    """Closed-form base-transform guess from one pose (zero joint/tool
    correction assumed), since the base transform could be anywhere in the
    room and an identity starting guess is not a useful prior. Solved by
    inverting the known partial (joint+tool) transform and combining it
    with the single measured pose.

    Args:
        first_measured_angles: (7,) joint angles from one sample pose.
        first_measured_pos_mm: (3,) measured position, millimeters.
        first_measured_quat_xyzw: (4,) measured orientation, scipy [x, y,
            z, w] order.

    Returns:
        Tuple of (base_xyz_guess (3,) meters, base_rpy_guess (3,) radians).
    """
    T_fk = forward_kinematics(first_measured_angles)
    T_tool_nominal = build_tool_transform(np.zeros(3), np.zeros(3))
    T_partial = T_fk @ T_tool_nominal

    T_measured = np.eye(4)
    T_measured[:3, :3] = Rotation.from_quat(first_measured_quat_xyzw).as_matrix()
    T_measured[:3, 3] = first_measured_pos_mm / 1000.0

    T_base_guess = T_measured @ np.linalg.inv(T_partial)
    base_xyz_guess = T_base_guess[:3, 3]
    base_rpy_guess = Rotation.from_matrix(T_base_guess[:3, :3]).as_euler("xyz")
    return base_xyz_guess, base_rpy_guess


# Residuals

# ORIENTATION_SCALE_MM makes radians and millimeters numerically
# comparable inside one least-squares cost. Without it, the optimizer
# would effectively ignore whichever residual type is smaller in raw
# magnitude. ORIENTATION_WEIGHT further de-weights orientation because
# position accuracy at the probe tip is the primary target for this arm's
# use case; orientation is fit for stability, not as the primary
# objective. The regularization weights are a fixed-size penalty appended
# once per residual call, not scaled per pose. As the pose count grows,
# their relative pull on the fit weakens automatically, since position
# and orientation residuals scale with pose count and these do not.
ORIENTATION_WEIGHT = 0.3      # orientation is de-weighted, not dropped: position accuracy is primary
ORIENTATION_SCALE_MM = 100.0  # converts radians to a scale comparable to the mm position residuals
JOINT_OFFSET_REG_WEIGHT = 8.0
TOOL_REG_WEIGHT = 1.0


def residual_function(
    param_vector: np.ndarray,
    measured_angles: np.ndarray,
    measured_pos_mm: np.ndarray,
    measured_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Full residual vector for least_squares(): per-pose position error,
    per-pose orientation error, and a fixed regularization block pulling
    joint_offsets/tool_xyz/tool_rpy toward zero. base_xyz/base_rpy are
    deliberately excluded from regularization; see the module docstring.

    Args:
        param_vector: Current (19,) optimizer estimate.
        measured_angles: (N, 7) measured joint angles.
        measured_pos_mm: (N, 3) measured positions, millimeters.
        measured_quat_xyzw: (N, 4) measured orientations, scipy order.

    Returns:
        Flat residual vector: 3N position + 3N orientation + 13
        regularization terms.
    """
    params = unpack_params(param_vector)
    n = len(measured_angles)

    pos_residuals = np.zeros((n, 3))
    orient_residuals = np.zeros((n, 3))

    for i in range(n):
        T_pred = predict_relative_pose(measured_angles[i], params)
        pred_pos_mm = T_pred[:3, 3] * 1000.0
        pos_residuals[i] = measured_pos_mm[i] - pred_pos_mm

        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(measured_quat_xyzw[i]).as_matrix()
        R_err = R_pred.T @ R_meas
        orient_residuals[i] = (
            Rotation.from_matrix(R_err).as_rotvec() * ORIENTATION_SCALE_MM * ORIENTATION_WEIGHT
        )

    reg_residual = np.concatenate([
        params.joint_offsets * (JOINT_OFFSET_REG_WEIGHT / max(JOINT_OFFSET_BOUND_RAD, 1e-9)),
        params.tool_xyz * (TOOL_REG_WEIGHT / max(TOOL_XYZ_BOUND_M, 1e-9)),
        params.tool_rpy * (TOOL_REG_WEIGHT / max(TOOL_RPY_BOUND_RAD, 1e-9)),
    ])

    return np.concatenate([pos_residuals.ravel(), orient_residuals.ravel(), reg_residual])


def rms_position_error_mm(param_vector: np.ndarray, measured_angles, measured_pos_mm):
    """RMS position error of a parameter vector against measured poses.

    Args:
        param_vector: (19,) parameters to evaluate.
        measured_angles: (N, 7) measured joint angles.
        measured_pos_mm: (N, 3) measured positions, millimeters.

    Returns:
        Tuple of (rms_mm: float, per_pose_errors_mm: (N,) array).
    """
    params = unpack_params(param_vector)
    errors = np.zeros(len(measured_angles))
    for i in range(len(measured_angles)):
        T_pred = predict_relative_pose(measured_angles[i], params)
        pred_pos_mm = T_pred[:3, 3] * 1000.0
        errors[i] = np.linalg.norm(measured_pos_mm[i] - pred_pos_mm)
    return float(np.sqrt(np.mean(errors ** 2))), errors


def rms_orientation_error_deg(param_vector: np.ndarray, measured_angles, measured_quat_xyzw):
    """Full predicted-vs-measured orientation error. Insensitive to the
    joint0<->base_rpy / joint6<->tool_rpy gauge freedom (see selftest()),
    unlike comparing those individual parameters, since it evaluates the
    combined orientation those ambiguous pairs leave unchanged.

    Args:
        param_vector: (19,) parameters to evaluate.
        measured_angles: (N, 7) measured joint angles.
        measured_quat_xyzw: (N, 4) measured orientations, scipy order.

    Returns:
        Tuple of (rms_deg: float, per_pose_errors_deg: (N,) array).
    """
    params = unpack_params(param_vector)
    errors = np.zeros(len(measured_angles))
    for i in range(len(measured_angles)):
        T_pred = predict_relative_pose(measured_angles[i], params)
        R_pred = T_pred[:3, :3]
        R_meas = Rotation.from_quat(measured_quat_xyzw[i]).as_matrix()
        errors[i] = np.degrees(Rotation.from_matrix(R_pred.T @ R_meas).magnitude())
    return float(np.sqrt(np.mean(errors ** 2))), errors


# CSV loading (columns written by calibration/collection/ndi_capture_and_validate.cpp).
# The column names below are an external contract with that C++ writer.
# A schema change there breaks this loader with a plain KeyError rather
# than a silent bug, but there is no shared schema definition enforcing it.

def load_poses_from_csv(csv_path: str):
    """Loads measured joint angles and NDI-tracked poses from a capture
    CSV.

    Args:
        csv_path: Path to a CSV written by ndi_capture_and_validate.cpp.

    Returns:
        Tuple of (measured_angles (N, 7), measured_pos_mm (N, 3),
        measured_quat_xyzw (N, 4)).
    """
    measured_angles = []
    measured_pos_mm = []
    measured_quat_xyzw = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            angles = np.array(
                [float(row[f"actual_rad_{i}"]) for i in range(N_JOINTS)]
            )
            pos_mm = np.array([
                float(row["moving_relative_fixed_tx_mm"]),
                float(row["moving_relative_fixed_ty_mm"]),
                float(row["moving_relative_fixed_tz_mm"]),
            ])
            # NdiPoseSample stores q0 (scalar/w), qx, qy, qz. scipy's
            # Rotation.from_quat expects [x, y, z, w] order. Getting this
            # reorder wrong does not raise an error; it silently produces
            # a plausible-looking but wrong rotation for every pose.
            q0 = float(row["moving_relative_fixed_q0"])
            qx = float(row["moving_relative_fixed_qx"])
            qy = float(row["moving_relative_fixed_qy"])
            qz = float(row["moving_relative_fixed_qz"])
            quat_xyzw = np.array([qx, qy, qz, q0])

            measured_angles.append(angles)
            measured_pos_mm.append(pos_mm)
            measured_quat_xyzw.append(quat_xyzw)

    return (
        np.array(measured_angles),
        np.array(measured_pos_mm),
        np.array(measured_quat_xyzw),
    )


# Calibration driver

def run_calibration(measured_angles, measured_pos_mm, measured_quat_xyzw, verbose=True):
    """Fits the 19-parameter calibration to a set of measured poses.

    Uses method="trf" (required for bounded least squares) with
    loss="soft_l1", which down-weights outlier poses (e.g. occlusion-
    degraded marker reads) instead of letting a handful of bad
    measurements dominate the fit the way a plain L2 loss would.

    Args:
        measured_angles: (N, 7) measured joint angles.
        measured_pos_mm: (N, 3) measured positions, millimeters.
        measured_quat_xyzw: (N, 4) measured orientations, scipy order.
        verbose: If True, prints scipy's per-iteration optimizer output.

    Returns:
        scipy.optimize.OptimizeResult from least_squares().
    """
    base_xyz_guess, base_rpy_guess = initial_base_guess(
        measured_angles[0], measured_pos_mm[0], measured_quat_xyzw[0]
    )

    initial_params = pack_params(
        joint_offsets=np.zeros(7),
        tool_xyz=np.zeros(3),
        tool_rpy=np.zeros(3),
        base_xyz=base_xyz_guess,
        base_rpy=base_rpy_guess,
    )

    lower, upper = build_bounds()
    # Widen the bounds if the closed-form guess itself falls outside them,
    # rather than letting least_squares() start infeasible and fail.
    lower = np.minimum(lower, initial_params - 1e-6)
    upper = np.maximum(upper, initial_params + 1e-6)

    result = least_squares(
        residual_function,
        initial_params,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        args=(measured_angles, measured_pos_mm, measured_quat_xyzw),
        verbose=2 if verbose else 0,
    )

    return result


def identifiability_report(result):
    """Prints the Jacobian's singular values and condition number at the
    fitted solution. A high condition number signals a poorly-constrained
    parameter combination, e.g. two parameters trading off against each
    other with near-equal effect on the residuals. The fit can still
    converge and look fine on RMS error alone while hiding this.

    Args:
        result: OptimizeResult from run_calibration().
    """
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    condition_number = (
        singular_values[0] / singular_values[-1] if singular_values[-1] > 1e-12 else float("inf")
    )
    print("\nIdentifiability check (Jacobian singular values at solution):")
    print(np.array2string(singular_values, precision=3, suppress_small=True))
    print(f"Condition number: {condition_number:.1f}")
    if condition_number > 1e4:
        print(
            "WARNING: high condition number -- some parameter combination is "
            "poorly constrained by this pose set. Consider more/more-varied "
            "poses before trusting the fitted values."
        )


def print_report(result, measured_angles, measured_pos_mm, train_idx, test_idx):
    """Prints fitted parameters, updated zeroTick values, and train/held-
    out RMS error, with an overfitting warning if held-out error is
    notably worse than train error.

    Args:
        result: OptimizeResult from run_calibration() (fit on train_idx
            only).
        measured_angles: (N, 7) full measured-angle set.
        measured_pos_mm: (N, 3) full measured-position set.
        train_idx: Indices used in the fit.
        test_idx: Held-out indices, or empty if none.
    """
    params = unpack_params(result.x)

    print("\n=== Fitted calibration parameters ===")
    for i, name in enumerate(JOINT_NAMES):
        offset_deg = np.degrees(params.joint_offsets[i])
        new_zero_tick = NOMINAL_ZERO_TICKS[i] - params.joint_offsets[i] * TICKS_PER_RADIAN
        print(
            f"  {name:24s} offset = {offset_deg:+7.3f} deg  "
            f"(zeroTick {NOMINAL_ZERO_TICKS[i]} -> {new_zero_tick:.1f})"
        )

    print(f"  tool_xyz (mm)  = {params.tool_xyz * 1000.0}")
    print(f"  tool_rpy (deg) = {np.degrees(params.tool_rpy)}")
    print(f"  base_xyz (mm)  = {params.base_xyz * 1000.0}")
    print(f"  base_rpy (deg) = {np.degrees(params.base_rpy)}")

    nominal_params = pack_params(
        np.zeros(7), np.zeros(3), np.zeros(3), params.base_xyz, params.base_rpy
    )

    train_rms_before, _ = rms_position_error_mm(
        nominal_params, measured_angles[train_idx], measured_pos_mm[train_idx]
    )
    train_rms_after, _ = rms_position_error_mm(
        result.x, measured_angles[train_idx], measured_pos_mm[train_idx]
    )
    print(f"\nTrain set ({len(train_idx)} poses):")
    print(f"  RMS position error, uncorrected joints: {train_rms_before:.3f} mm")
    print(f"  RMS position error, after calibration:  {train_rms_after:.3f} mm")

    if len(test_idx) > 0:
        test_rms_before, _ = rms_position_error_mm(
            nominal_params, measured_angles[test_idx], measured_pos_mm[test_idx]
        )
        test_rms_after, _ = rms_position_error_mm(
            result.x, measured_angles[test_idx], measured_pos_mm[test_idx]
        )
        print(f"\nHeld-out set ({len(test_idx)} poses, not used in the fit):")
        print(f"  RMS position error, uncorrected joints: {test_rms_before:.3f} mm")
        print(f"  RMS position error, after calibration:  {test_rms_after:.3f} mm")
        # 1.5x is a heuristic threshold, not a statistically derived one.
        # It flags gross overfitting, not marginal generalization gaps.
        if test_rms_after > 1.5 * train_rms_after:
            print(
                "WARNING: held-out error is notably worse than train error -- "
                "likely overfitting. Trust these parameters less; consider "
                "more poses or tighter regularization."
            )

    identifiability_report(result)


def calibrate_from_csv(csv_path: str, test_fraction: float = 0.2, seed: int = 0):
    """Loads a capture CSV, splits it into train/held-out sets, fits the
    calibration, and prints a report.

    The train/test split is a uniform random permutation, not
    trajectory-aware. If the CSV's poses were captured as a continuous
    sweep, adjacent poses are highly correlated, so a random split can
    leak near-duplicate configurations across the train/test boundary and
    make held-out error look better than true generalization.

    Args:
        csv_path: Path to a CSV written by ndi_capture_and_validate.cpp.
        test_fraction: Fraction of poses held out, if there are at least
            10 poses total.
        seed: RNG seed for the train/test split, for reproducibility.

    Returns:
        OptimizeResult from run_calibration().
    """
    measured_angles, measured_pos_mm, measured_quat_xyzw = load_poses_from_csv(csv_path)
    n = len(measured_angles)
    if n < 10:
        print(
            f"WARNING: only {n} poses loaded -- this is too few to reliably "
            "identify 19 parameters. Results below may not be trustworthy."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(n * test_fraction)) if n >= 10 else 0
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    result = run_calibration(
        measured_angles[train_idx], measured_pos_mm[train_idx], measured_quat_xyzw[train_idx]
    )

    print_report(result, measured_angles, measured_pos_mm, train_idx, test_idx)
    return result


# Self-test: synthetic data with known ground truth, no hardware required.
# Validates the optimizer and math independent of any capture data or
# tracker issues. Run this first when debugging a fit that looks wrong,
# to rule out a code bug before suspecting the data.

def selftest():
    """Generates synthetic poses from known ground-truth parameters, fits
    against them, and checks the fit recovers those parameters, except the
    two structurally unidentifiable ones (see the comment below).

    Returns:
        True if the fit passes all checks, False otherwise. Also printed.
    """
    rng = np.random.default_rng(1234)

    true_joint_offsets = np.radians(rng.uniform(-3.0, 3.0, size=7))
    true_tool_xyz = rng.uniform(-0.004, 0.004, size=3)
    true_tool_rpy = np.radians(rng.uniform(-3.0, 3.0, size=3))
    true_base_xyz = np.array([0.35, -0.10, 0.20]) + rng.uniform(-0.02, 0.02, size=3)
    # Pitch deliberately kept well away from +-90 deg: xyz-Euler angles are
    # non-unique (gimbal lock) right at that singularity, which would make
    # the raw rpy comparison below meaningless even for a correct fit.
    true_base_rpy = np.radians(np.array([5.0, -30.0, 10.0])) + np.radians(
        rng.uniform(-2.0, 2.0, size=3)
    )
    true_params = CalibParams(
        true_joint_offsets, true_tool_xyz, true_tool_rpy, true_base_xyz, true_base_rpy
    )

    n_poses = 60
    measured_angles = np.zeros((n_poses, N_JOINTS))
    for i in range(n_poses):
        for j in range(N_JOINTS):
            min_tick, max_tick = JOINT_TICK_RANGES[j]
            tick = rng.uniform(min_tick + 100, max_tick - 100)
            measured_angles[i, j] = (tick - NOMINAL_ZERO_TICKS[j]) / TICKS_PER_RADIAN

    measured_pos_mm = np.zeros((n_poses, 3))
    measured_quat_xyzw = np.zeros((n_poses, 4))
    position_noise_mm = 0.3
    orientation_noise_rad = np.radians(0.2)

    for i in range(n_poses):
        T_true = predict_relative_pose(measured_angles[i], true_params)
        pos_mm = T_true[:3, 3] * 1000.0 + rng.normal(0, position_noise_mm, size=3)
        noise_rotvec = rng.normal(0, orientation_noise_rad, size=3)
        R_noisy = T_true[:3, :3] @ Rotation.from_rotvec(noise_rotvec).as_matrix()
        measured_pos_mm[i] = pos_mm
        measured_quat_xyzw[i] = Rotation.from_matrix(R_noisy).as_quat()

    result = run_calibration(measured_angles, measured_pos_mm, measured_quat_xyzw, verbose=False)
    fitted = unpack_params(result.x)

    joint_offset_err_deg = np.degrees(np.abs(fitted.joint_offsets - true_joint_offsets))

    print("=== Self-test: recovered vs. ground-truth parameters ===")
    print(f"Joint offset errors (deg), all 7 joints: {joint_offset_err_deg}")
    print(f"Tool xyz error (mm): {np.abs(fitted.tool_xyz - true_tool_xyz) * 1000.0}")
    print(f"Base xyz error (mm): {np.abs(fitted.base_xyz - true_base_xyz) * 1000.0}")

    rms_pos_mm, _ = rms_position_error_mm(result.x, measured_angles, measured_pos_mm)
    rms_orient_deg, _ = rms_orientation_error_deg(result.x, measured_angles, measured_quat_xyzw)
    print(f"RMS position error vs. noisy synthetic measurements: {rms_pos_mm:.3f} mm")
    print(f"RMS orientation error vs. noisy synthetic measurements: {rms_orient_deg:.3f} deg")

    identifiability_report(result)
    # joint_offset[0]/[6] are excluded here: each is degenerate with
    # base_rpy/tool_rpy respectively (confirmed via Jacobian SVD), so no
    # fit can recover them independently. This is a structural property of
    # this kinematic chain, not a bug. Do not fix a failing self-test by
    # loosening these two joints' thresholds instead of excluding them.
    # rms_pos_mm/rms_orient_deg above already validate the combined
    # predicted pose regardless of how the fit splits credit between a
    # joint offset and its neighboring frame rotation.
    other_joint_offset_err_deg = np.delete(joint_offset_err_deg, [0, 6])

    passed = (
        np.all(other_joint_offset_err_deg < 0.5)
        and np.all(np.abs(fitted.tool_xyz - true_tool_xyz) < 0.001)
        and np.all(np.abs(fitted.base_xyz - true_base_xyz) < 0.002)
        and rms_pos_mm < 3.0 * position_noise_mm
        and rms_orient_deg < 3.0 * np.degrees(orientation_noise_rad)
    )
    print("SELF-TEST PASSED" if passed else "SELF-TEST FAILED")
    if passed:
        print(
            "(joint_offset[0] and joint_offset[6] are intentionally not "
            "checked individually -- see the comment above this line.)"
        )
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", help="Path to five_pose_ndi_capture.csv")
    parser.add_argument(
        "--selftest", action="store_true", help="Run the synthetic-data correctness check"
    )
    args = parser.parse_args()

    if args.selftest:
        passed = selftest()
        sys.exit(0 if passed else 1)

    if not args.csv_path:
        parser.error("csv_path is required unless --selftest is given")

    calibrate_from_csv(args.csv_path)


if __name__ == "__main__":
    main()
