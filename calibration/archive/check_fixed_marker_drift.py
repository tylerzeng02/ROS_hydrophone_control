"""Checks whether the fixed NDI marker's orientation has drifted since the
batch2 calibration fit that move_between_points.cpp's hardcoded
R_MOVEIT_TO_NDI rotation matrix was derived from.

Why this matters: move_between_points.cpp converts an NDI-frame delta into
a MoveIt-frame command using a ROTATION matrix baked into the C++ source,
fit once from the fixed marker's orientation during batch2 data collection.
If the fixed marker's tripod has been reoriented since then, that hardcoded
rotation is stale and every commanded move gets rotated in the wrong
direction -- regardless of whether the marker has been perfectly stable
during a later, separate calibration/test session. See the conversation
that motivated this script for the full reasoning.

Usage:
    1. Get a fresh live reading of the fixed tool's orientation quaternion.
       The easiest way: run ndi_measure (ros/src/cyton_ndi_capture), press
       Enter once at any arm pose, and read fixed_camera_q0/qx/qy/qz from
       its output CSV.
    2. python check_fixed_marker_drift.py <q0> <qx> <qy> <qz>

Reference quaternion is recomputed fresh from
build/quick_calibration_test_fixed_elbow_yaw_batch2.csv every run (not
hardcoded), so it always reflects the actual data on disk.
"""

import csv
import sys

import numpy as np

BATCH2_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_fixed_elbow_yaw_batch2.csv"

# Rough interpretation bands, from the delta-vector sensitivity estimate
# (error ~= target_distance_mm * angle_rad) at this project's typical
# move_between_points target distances (~100-150mm).
BANDS = [
    (0.2, "negligible -- within normal per-sample NDI jitter, not real drift"),
    (0.5, "small -- plausibly still just noise, but worth a second reading to confirm"),
    (1.0, "real -- likely contributing 1-3mm of directional error to move_between_points results"),
    (2.0, "significant -- likely a meaningful fraction of the observed error; consider refitting"),
    (float("inf"), "large -- R_MOVEIT_TO_NDI is very likely stale; refit before trusting this tool"),
]


def load_batch2_reference():
    qs = []
    with open(BATCH2_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = np.array([
                float(row["fixed_camera_q0"]), float(row["fixed_camera_qx"]),
                float(row["fixed_camera_qy"]), float(row["fixed_camera_qz"]),
            ])
            qs.append(q)
    qs = np.array(qs)
    ref = qs[0].copy()
    for i in range(len(qs)):
        if np.dot(qs[i], ref) < 0:
            qs[i] = -qs[i]
    mean_q = qs.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q)

    dots = np.clip(np.abs(qs @ mean_q), -1, 1)
    spread_deg = np.degrees(2 * np.arccos(dots))
    return mean_q, len(qs), spread_deg


def angle_between_quaternions_deg(q1, q2):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def interpret(angle_deg):
    for threshold, label in BANDS:
        if angle_deg <= threshold:
            return label
    return BANDS[-1][1]


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <q0> <qx> <qy> <qz>", file=sys.stderr)
        print("  (fresh fixed-tool orientation quaternion from a live NDI reading)",
              file=sys.stderr)
        sys.exit(1)

    fresh_q = [float(x) for x in sys.argv[1:5]]

    ref_q, n_rows, spread_deg = load_batch2_reference()
    print(f"Batch2 reference quaternion (from {n_rows} rows): "
          f"q0={ref_q[0]:.6f} qx={ref_q[1]:.6f} qy={ref_q[2]:.6f} qz={ref_q[3]:.6f}")
    print(f"Batch2's own internal spread (session noise floor): "
          f"mean={spread_deg.mean():.4f} deg, max={spread_deg.max():.4f} deg")

    angle = angle_between_quaternions_deg(ref_q, fresh_q)
    print(f"\nFresh reading: q0={fresh_q[0]:.6f} qx={fresh_q[1]:.6f} "
          f"qy={fresh_q[2]:.6f} qz={fresh_q[3]:.6f}")
    print(f"Angle vs. batch2 reference: {angle:.4f} deg")
    print(f"Interpretation: {interpret(angle)}")

    if angle > spread_deg.max():
        print(f"\n(This exceeds batch2's own max internal noise of {spread_deg.max():.4f} deg, "
              f"so it's unlikely to be pure measurement noise.)")


if __name__ == "__main__":
    main()
