"""Fits ONE new correction term -- an additive, angle-dependent deflection
correction to shoulder_pitch under I=8 -- on top of the EXISTING, FROZEN
48-param reduced model (offset+scale+tilt+origin), the same way gravity
deflection and joint coupling were each added previously in this project.

Methodology:
  1. Fit the 48-param reduced model fresh on the original I=0 374-pose
     dataset (frozen afterward -- not touched again).
  2. Using the partial I=8 --validate run's achieved ticks + real NDI-
     measured positions, fit a 2-param linear-in-angle correction to
     shoulder_pitch's own joint angle: extra_angle = k0 + k1*sp_angle.
     Everything else in the model stays frozen.
  3. Validate with a BLOCKED (not random) train/test split, since the I=8
     data was collected in trajectory order just like every other dataset
     in this project.
"""

import csv
import sys

import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck
import reduced_model_blocked_cv as rm

I0_BASELINE_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_fixed_elbow_yaw.csv"
I8_PARTIAL_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/validation_results_igain8_snapshot2.csv"

SHOULDER_PITCH_IDX = 1


def raw_angles_from_ticks(ticks):
    """Convert raw achieved ticks -> nominal (uncorrected) radians, matching
    the exact convention ck.load_poses_from_csv's actual_rad_i columns use
    (NOMINAL_ZERO_TICKS, scale=1 -- correction is applied later in predict())."""
    return (np.array(ticks) - np.array(ck.NOMINAL_ZERO_TICKS)) / ck.TICKS_PER_RADIAN


def load_i8_partial(csv_path):
    angles, pos_mm = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ticks = [float(row[f"achieved_tick_{i}"]) for i in range(7)]
            angles.append(raw_angles_from_ticks(ticks))
            pos_mm.append([float(row["actual_x_mm"]), float(row["actual_y_mm"]), float(row["actual_z_mm"])])
    return np.array(angles), np.array(pos_mm)


def predict_with_deflection(ar, bp, js, tl, oe, osy, owp, k0, k1):
    ar2 = ar.copy()
    ar2[SHOULDER_PITCH_IDX] = ar2[SHOULDER_PITCH_IDX] + k0 + k1 * ar[SHOULDER_PITCH_IDX]
    return rm.predict(ar2, bp, js, tl, oe, osy, owp)


def rms_with_deflection(angles, pos, bp, js, tl, oe, osy, owp, k0, k1):
    e = np.zeros(len(angles))
    for i in range(len(angles)):
        Tp = predict_with_deflection(angles[i], bp, js, tl, oe, osy, owp, k0, k1)
        e[i] = np.linalg.norm(pos[i] - Tp[:3, 3] * 1000.0)
    return np.sqrt(np.mean(e ** 2)), e


def fit_deflection_term(angles, pos, bp, js, tl, oe, osy, owp):
    def res(k):
        k0, k1 = k
        n = len(angles)
        pr = np.zeros((n, 3))
        for i in range(n):
            Tp = predict_with_deflection(angles[i], bp, js, tl, oe, osy, owp, k0, k1)
            pr[i] = pos[i] - Tp[:3, 3] * 1000.0
        return pr.ravel()

    r = least_squares(res, x0=[0.0, 0.0], method="trf", x_scale="jac",
                       bounds=([-np.radians(30), -5.0], [np.radians(30), 5.0]))
    return r.x


def main():
    print("Step 1: fitting frozen 48-param reduced model on the I=0 baseline (374 poses)...")
    angles0, pos0, quat0 = ck.load_poses_from_csv(I0_BASELINE_CSV)
    result0 = rm.fit(angles0, pos0, quat0)
    bp, js, tl, oe, osy, owp = rm.unpack(result0.x)
    baseline_rms = rm.rms(angles0, pos0, bp, js, tl, oe, osy, owp)
    print(f"  Frozen model in-sample RMS on I=0 data: {baseline_rms:.3f}mm (sanity check vs. documented 0.676mm)")

    print(f"\nStep 2: loading I=8 partial data from {I8_PARTIAL_CSV}...")
    angles8, pos8 = load_i8_partial(I8_PARTIAL_CSV)
    n = len(angles8)
    print(f"  {n} points loaded")

    # No-correction baseline: how well does the FROZEN (I=0-fit) model alone
    # predict the I=8 measurements, with zero new term?
    rms_no_term, _ = rms_with_deflection(angles8, pos8, bp, js, tl, oe, osy, owp, 0.0, 0.0)
    print(f"\nFrozen model alone (no new term) vs I=8 measurements, ALL {n} points: {rms_no_term:.2f}mm RMS")

    # Blocked split: first 70% train, last 30% test (collection order)
    n_train = int(n * 0.7)
    train_a, test_a = angles8[:n_train], angles8[n_train:]
    train_p, test_p = pos8[:n_train], pos8[n_train:]

    print(f"\nStep 3: fitting shoulder_pitch deflection term (k0 + k1*angle) on {n_train} training points...")
    k0, k1 = fit_deflection_term(train_a, train_p, bp, js, tl, oe, osy, owp)
    print(f"  Fitted: k0={np.degrees(k0):.3f}deg, k1={k1:.4f} (rad extra per rad of shoulder_pitch angle)")

    train_rms, _ = rms_with_deflection(train_a, train_p, bp, js, tl, oe, osy, owp, k0, k1)
    test_rms_no_term, _ = rms_with_deflection(test_a, test_p, bp, js, tl, oe, osy, owp, 0.0, 0.0)
    test_rms_with_term, _ = rms_with_deflection(test_a, test_p, bp, js, tl, oe, osy, owp, k0, k1)

    print(f"\n=== Results (held-out test set, {n - n_train} points, blocked split) ===")
    print(f"  Train RMS (with new term):        {train_rms:.2f}mm")
    print(f"  Test RMS, NO new term (baseline): {test_rms_no_term:.2f}mm")
    print(f"  Test RMS, WITH new term:          {test_rms_with_term:.2f}mm")
    print(f"  Improvement: {test_rms_no_term - test_rms_with_term:.2f}mm")


if __name__ == "__main__":
    main()
