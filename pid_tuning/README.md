# pid_tuning/

Everything related to the shoulder_pitch I-gain (PID) tuning investigation,
consolidated here 2026-08-19 (previously scattered across `tests/`,
`calibration/current/`, and `build/`). See `CLAUDE.md`'s I-gain sections
for the full story — this is just a map of what lives here.

**Bottom line finding: no I-gain value tested (`4` through `30`) is more
accurate than `I=0` on real position accuracy**, even though non-zero
values do measurably reduce tick-level settling error. `I=0` remains
deployed.

## `tests/` — C++ hardware tools

- `check_servo_model.cpp` — confirmed the arm's real servo models (MX-64/MX-28, not AX-12A — a corrected long-standing assumption).
- `set_i_gain.cpp` — writes an I-gain value and leaves it applied (RAM-only, resets on power-cycle).
- `test_i_gain.cpp` — single-trial before/after I-gain test, always restores the original value.
- `autotune_i_gain.cpp` — automatic tick-error-minimizing sweep across candidate I-values (the tool that originally picked `I=8` — later found to be optimizing the wrong metric, see below).
- `test_compliance_punch.cpp`, `reset_compliance.cpp` — earlier AX-series-style compliance-margin/punch tools, superseded once the MX-series real-PID-gain registers were confirmed, kept for reference.

These build via the root `CMakeLists.txt` same as everything in `tests/` — just with source paths pointing here now.

## `scripts/` — analysis and automation

- `fit_shoulder_pitch_i_gain_8_correction_term.py` — fits a new, physically-integrated correction term for the `I=8` deflection effect on top of the frozen 48-param model. Result: real but modest (~52% reduction on the tested range), and this term was never deployed.
- `diagnose_i_gain_8_regression_offset_refit_attempt.py`, `diagnose_i_gain_8_regression_rotation_fit_attempt.py` — two earlier, superseded attempts at explaining the `I=8` accuracy regression geometrically (one by re-fitting shoulder_pitch's offset, one by fitting a pure rotation to the measured position shift). Both are dead ends — kept for reference, not because they succeeded.
- `run_i_gain_sweep_12pose.ps1` / `_35pose.ps1` / `_37pose.ps1` / `_final_45pose.ps1` — automated sweep scripts, each cycling through I-gain candidates `0,4,8,12,16,20,24,30` fully unattended against its own pose set (torque held enabled the whole time, released only via one Enter press at the very end). `_final_45pose.ps1` (the hand-posed 45-pose set) is the most complete/final version; the `_12pose`/`_35pose`/`_37pose` scripts are earlier, smaller-subset attempts, kept since they're still runnable as-is.

**Run these from anywhere** — they set their own working directory internally (the `.exe` tools live in `../../build/`, referenced via `$PSScriptRoot`-relative paths).

## `data/`

Every file here starts with `i_gain_<value>_...` for a specific I-gain setting, or `i_gain_sweep_...`/`i_gain_final_sweep_...` for the pose sets the sweep scripts move the arm through.

- `i_gain_final_sweep_45pose_hand_posed_dataset.csv` — the 45 hand-posed poses (elbow_yaw locked) behind the final, most complete sweep, recorded via `calibration/collection/record_hand_poses.cpp`.
- `i_gain_final_sweep_45pose_validate_input.csv` — the same 45 poses, reformatted as an input file for `ndi_capture_and_validate.exe --validate`.
- `i_gain_{0,4,8,12,16,20,24,30}_final_sweep_accuracy_results.csv` — the final sweep's real NDI-measured accuracy results for each I-gain candidate, on the 45-pose set above. This is the data behind the "no candidate beats I=0" bottom-line finding.
- `i_gain_sweep_12pose_spread_across_range_input.csv`, `i_gain_sweep_35pose_early_subset_input.csv`, `i_gain_sweep_37pose_early_subset_input.csv` — three earlier, smaller candidate pose sets, each the literal input to its matching `.ps1` script above. Superseded by the 45-pose final sweep but kept since the scripts still run against them as-is.
- `i_gain_0_baseline_accuracy_results.csv`, `i_gain_8_regression_discovery_9pose_results.csv` — the original 9-point same-session A/B test (I=0 vs I=8) that first caught I=8 making real accuracy *worse* (10.25mm → 24.60mm mean deviation) — the finding that triggered this whole investigation.
- `i_gain_8_374pose_dataset_partial_133of374_results.csv` — a 133-of-374-point partial NDI capture from the first attempt to collect I=8 accuracy data on the full 374-pose calibration dataset (run was interrupted before finishing); read by `fit_shoulder_pitch_i_gain_8_correction_term.py`.
- `i_gain_test_10pose_with_predicted_positions.csv` — a `--validate`-format file (tick targets + the calibration model's predicted positions) for 10 test points, used as a reference/cross-check input by the two geometric diagnostic scripts above.

Superseded I-gain-era files (the 12-point sweep's own results, abandoned partial collections, early compliance diagnostics) were left in `build/archive/` rather than duplicated here — see that folder's own documentation.
