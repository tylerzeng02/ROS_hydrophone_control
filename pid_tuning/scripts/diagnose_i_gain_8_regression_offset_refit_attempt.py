"""Single-parameter refit: shoulder_pitch's offset, using the 9 points
already collected with I=8 active (pid_tuning/data/i_gain_8_regression_discovery_9pose_results.csv).

CORRECTED 2026-08-14: the first version of this script hardcoded fitted
parameter values transcribed from final_deployment_fit.py's rerun on
deployed_model_training_dataset_374pose.csv -- WRONG dataset. The actual
predictions in i_gain_test_10pose_with_predicted_positions.csv (and hence
pid_tuning/data/validation_results*.csv, which all trace back to it) were generated
by gen_validation_predictions.py, whose default --fit-csv is
quick_calibration_test.csv + quick_calibration_test_batch3.csv -- a
different fit entirely. That mismatch is why the first version's own
recomputed "BEFORE" predictions didn't match the file's recorded
predicted_x/y/z_mm (off by 5-12mm per point, confirmed via direct
comparison) -- not a bug in the FK math, a wrong parameter set.

Fixed by importing gen_validation_predictions.py directly (it's properly
guarded with if __name__ == "__main__", so importing it does NOT trigger
its own main()) and reusing its exact fit()/predict() functions and file
paths -- no hand-transcription, no risk of a repeat of this exact mistake.

Context: every parameter in that model was fit using data collected under
the OLD servo behavior (I=0), where shoulder_pitch consistently undershot
its commanded tick by ~7-9 ticks. The fitted shoulder_pitch offset
silently absorbed/compensated for that consistent undershoot. Now that
I=8 fixes the undershoot at the servo level (confirmed: mean tick error
8.8 -> 0.6 ticks), the model's stale compensation is wrong in the same
direction it used to correct for, producing a fresh systematic error
(confirmed: pid_tuning/data/i_gain_0_baseline_accuracy_results.csv vs
pid_tuning/data/i_gain_8_regression_discovery_9pose_results.csv, same 9 points, ~10mm -> ~25mm
mean deviation).

This script re-fits the SAME model gen_validation_predictions.py uses
(same fit-csv files), then holds every one of its parameters FIXED and
fits ONLY an additional shoulder_pitch offset against the 9
(achieved_tick, actual_position) pairs already collected with I=8 active
-- no new data collection needed for that second step.

Run from calibration/current/:
    python3 diagnose_i_gain_8_regression_offset_refit_attempt.py
"""

import csv
import sys

import numpy as np
from scipy.optimize import least_squares

import calibrate_kinematics as ck
import gen_validation_predictions as gvp

SHOULDER_PITCH = 1

FIT_CSV_PATHS = [
    "../../build/archive/ndi_capture/quick_calibration_test.csv",
    "../../build/archive/ndi_capture/quick_calibration_test_batch3.csv",
]

print("Refitting the SAME model gen_validation_predictions.py uses "
      f"(fit-csv={FIT_CSV_PATHS})...")
angles_list, pos_list, quat_list = [], [], []
for path in FIT_CSV_PATHS:
    a, p, q = ck.load_poses_from_csv(path)
    angles_list.append(a)
    pos_list.append(p)
    quat_list.append(q)
    print(f"  loaded {len(a)} poses from {path}")
angles_all = np.concatenate(angles_list)
pos_mm_all = np.concatenate(pos_list)
quat_xyzw_all = np.concatenate(quat_list)

result = gvp.fit(angles_all, pos_mm_all, quat_xyzw_all)
bp, js, tl, oe, osy, owp, fab, cc, gc = gvp.unpack(result.x)

e = np.array([
    np.linalg.norm(pos_mm_all[i] - gvp.predict(angles_all[i], bp, js, tl, oe, osy, owp, fab, cc, gc)[:3, 3] * 1000.0)
    for i in range(len(angles_all))
])
print(f"Fit in-sample RMS: {np.sqrt(np.mean(e ** 2)):.2f}mm (sanity check only)\n")

# ---------------------------------------------------------------------
# Cross-check against the file's own recorded predictions BEFORE trusting
# anything further -- this is exactly the check that caught the wrong-
# dataset mistake last time; do not skip it again.
# ---------------------------------------------------------------------
DATA_PATH = "../data/i_gain_8_regression_discovery_9pose_results.csv"
with open(DATA_PATH) as f:
    rows = list(csv.DictReader(f))

print("Cross-check: my refit's predictions (using TARGET ticks) vs. the file's own recorded predicted_x/y/z_mm:")
max_diff = 0.0
for r in rows:
    target_ticks = np.array([float(r[f"tick_{i}"]) for i in range(7)])
    ar = gvp.ticks_to_radians(target_ticks)
    Tp = gvp.predict(ar, bp, js, tl, oe, osy, owp, fab, cc, gc)
    mine = Tp[:3, 3] * 1000.0
    file_pred = np.array([float(r["predicted_x_mm"]), float(r["predicted_y_mm"]), float(r["predicted_z_mm"])])
    diff = np.linalg.norm(mine - file_pred)
    max_diff = max(max_diff, diff)
    print(f"  test_id {r['test_id']}: mine={mine}, file={file_pred}, diff={diff:.3f}mm")
print(f"Max diff: {max_diff:.3f}mm\n")

if max_diff > 1.0:
    print("STOP: still not matching the file's recorded predictions closely (>1mm) -- "
          "do NOT trust the correction below until this is resolved.", file=sys.stderr)
    sys.exit(1)
print("Confirmed matching (<1mm) -- safe to proceed.\n")

# ---------------------------------------------------------------------
# Now the actual 1-parameter refit: extra shoulder_pitch offset, against
# the 9 already-collected I=8 points (achieved ticks + real NDI position).
# ---------------------------------------------------------------------
ar_list, actual_list = [], []
for r in rows:
    achieved_ticks = np.array([float(r[f"achieved_tick_{i}"]) for i in range(7)])
    ar_list.append(gvp.ticks_to_radians(achieved_ticks))
    actual_list.append(np.array([float(r["actual_x_mm"]), float(r["actual_y_mm"]), float(r["actual_z_mm"])]))


def predict_with_extra_sp_offset(ar, extra_sp_offset):
    ar2 = ar.copy()
    # Adding directly to the RAW angle before scale/offset is applied would
    # double-count the joint's own scale factor -- instead, patch bp
    # (params.joint_offsets) with the extra correction, matching exactly
    # how gvp.predict() itself applies joint_offsets.
    params = ck.unpack_params(bp)
    joint_offsets2 = params.joint_offsets.copy()
    joint_offsets2[SHOULDER_PITCH] += extra_sp_offset
    bp2 = ck.pack_params(joint_offsets2, params.tool_xyz, params.tool_rpy, params.base_xyz, params.base_rpy)
    return gvp.predict(ar2, bp2, js, tl, oe, osy, owp, fab, cc, gc)


def residuals(x):
    extra_sp_offset = x[0]
    res = []
    for ar, actual in zip(ar_list, actual_list):
        Tp = predict_with_extra_sp_offset(ar, extra_sp_offset)
        res.append(actual - Tp[:3, 3] * 1000.0)
    return np.concatenate(res)


before = residuals(np.array([0.0])).reshape(-1, 3)
before_dev = np.linalg.norm(before, axis=1)
print(f"BEFORE (no extra correction): mean={before_dev.mean():.2f}mm "
      f"rms={np.sqrt((before_dev**2).mean()):.2f}mm max={before_dev.max():.2f}mm")

fit_result = least_squares(
    residuals, x0=np.array([0.0]), method="trf", x_scale="jac",
    bounds=([-np.radians(10.0)], [np.radians(10.0)]),
)
extra_sp_offset = fit_result.x[0]

after = residuals(fit_result.x).reshape(-1, 3)
after_dev = np.linalg.norm(after, axis=1)
print(f"\nAFTER (extra_sp_offset={np.degrees(extra_sp_offset):+.4f} deg = "
      f"{extra_sp_offset * ck.TICKS_PER_RADIAN:+.2f} ticks in the 'ar' convention): "
      f"mean={after_dev.mean():.2f}mm rms={np.sqrt((after_dev**2).mean()):.2f}mm max={after_dev.max():.2f}mm")

sp_scale = js[SHOULDER_PITCH]
zero_tick_delta = -extra_sp_offset * ck.TICKS_PER_RADIAN / sp_scale
print(f"\nrobot_calibration.cpp shoulder_pitch (motor 1) zeroTick should change by "
      f"{zero_tick_delta:+.2f} ticks (zeroTick_new = zeroTick_current - offset_rad*TICKS_PER_RADIAN/scale) "
      f"to fold this correction in directly.")

print("\nPer-point deviation (mm):")
for r, b, a in zip(rows, before_dev, after_dev):
    print(f"  test_id {r['test_id']}: before={b:.2f}mm -> after={a:.2f}mm")
