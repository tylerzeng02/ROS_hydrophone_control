# calibration/data/

The real, tracked NDI-capture dataset behind the deployed kinematic
calibration model — moved out of `build/` on 2026-08-20 so real data isn't
sitting inside what's supposed to be a regenerable CMake build directory.
See `CLAUDE.md`'s kinematic-calibration history for the full story.

- **`deployed_model_training_dataset_374pose.csv`** — the 374-pose,
  elbow_yaw-locked NDI capture the currently-deployed 48-param calibration
  model (in `src/robot_calibration.cpp` / `references/cyton_gamma_1500_trac_ik.urdf`)
  was actually fit on. Blocked-CV RMS 0.78mm — the best validated result in
  the project. Read by `calibration/current/final_deployment_fit.py`,
  `validate_48param_model_blocked_cross_validation.py`,
  `deployed_model_predictions.py`, and
  `pid_tuning/scripts/fit_shoulder_pitch_i_gain_8_correction_term.py`. This
  is the one file you need to reproduce the deployed model.

`marker_drift_recheck_dataset_377pose.csv` (a from-scratch recollection of
the same batch, done to verify a suspected NDI marker drift hadn't
corrupted the original data — it hadn't) was archived to
`calibration/archive/` on 2026-08-20: it was never read by any fitting
script, and kept nothing live in `calibration/data/` needing it.

`calibration/collection/ndi_capture_and_validate.cpp`'s `QUICK_TEST_CSV` constant still
writes fresh `--quick-test` output under that same filename — but since the
tool runs from (and writes into) `build/` by default, a future rerun's
output lands in `build/`, not here, and needs moving manually if it's meant
to be kept (see `build/README.md`).
