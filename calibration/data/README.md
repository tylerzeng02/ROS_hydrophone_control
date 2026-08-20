# calibration/data/

The real, tracked NDI-capture datasets behind the deployed kinematic
calibration model — moved out of `build/` on 2026-08-20 so real data isn't
sitting inside what's supposed to be a regenerable CMake build directory.
See `CLAUDE.md`'s kinematic-calibration history for the full story behind
either file.

- **`deployed_model_training_dataset_374pose.csv`** — the 374-pose,
  elbow_yaw-locked NDI capture the currently-deployed 48-param calibration
  model (in `src/robot_calibration.cpp` / `references/cyton_gamma_1500_trac_ik.urdf`)
  was actually fit on. Blocked-CV RMS 0.78mm — the best validated result in
  the project. Read by `calibration/current/final_deployment_fit.py`,
  `validate_48param_model_blocked_cross_validation.py`,
  `deployed_model_predictions.py`, and
  `pid_tuning/scripts/fit_shoulder_pitch_i_gain_8_correction_term.py`. This
  is the one file you need to reproduce the deployed model.
- **`marker_drift_recheck_dataset_377pose.csv`** — a from-scratch
  recollection of the same 382-pose batch (377 of 382 poses captured),
  done after a suspected NDI fixed-marker drift was found and the physical
  setup was corrected. Confirmed the recollected data fits just as well
  (in-sample 0.633mm / blocked-CV 0.727mm) as the original — i.e. this file
  exists to answer "is the original dataset still trustworthy," and the
  answer was yes. **Not currently read by any fitting script** — the
  deployed model still comes from the original 374-pose file above; this is
  a QA/verification artifact, not a second candidate training set.

`tests/test_five_pose_ndi_capture.cpp`'s `QUICK_TEST_CSV` constant still
writes fresh `--quick-test` output under `marker_drift_recheck_dataset_377pose.csv`'s
plain filename — but since the tool runs from (and writes into) `build/` by
default, a future rerun's output will land in `build/`, not here, and needs
moving into this folder manually afterward if it's meant to be kept (see
`build/README.md`).
