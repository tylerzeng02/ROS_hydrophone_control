# ros/

Data files tracked at the top level of the ROS 2 workspace, and what each
one is for. All of the NDI-capture-format files below share the same
`ndi_measure` CSV schema (joint angles, MoveIt pose, moving/fixed camera
readings, and the moving-relative-to-fixed result).

- **`ndi_moveit_rotation_calibration_data.csv`** (14 rows, 13 valid) —
  paired MoveIt/NDI position readings, used by
  `calibration/current/refit_moveit_ndi_rotation.py` to fit
  `R_MOVEIT_TO_NDI`, the rotation deployed in
  `ros/src/cyton_accuracy_check/src/move_between_points.cpp` that converts
  a commanded MoveIt-frame delta into the NDI tracker's frame for error
  scoring. One row (exact zero position with identity orientation) is a
  failed `getCurrentPose()` sentinel, not a real reading, and is skipped
  by the fitting script.

- **`skull_probe_accuracy_test_target_points.csv`** (12 rows) — real,
  NDI-measured target points from the skull-probing application's own
  working volume, used as `move_between_points`'s input for the
  real-world round-trip accuracy check. Hardcoded as the default input
  path in `ros/src/cyton_pose_commander/src/replay_ndi_capture.cpp` and
  `replay_ndi_capture_sim.cpp`.

- **`skull_probe_accuracy_test_results.csv`** (12 rows) — the measured
  results from running the target points above through
  `move_between_points`: the real-world round-trip accuracy figure. An
  output artifact, not read by any script.

- **`fus_targeting_session_*.csv`** — auto-generated, timestamped session
  logs, one per `fus_targeting_gui` run, written by `main_window.py`'s
  `_start_csv_log()`. Each row is one accepted target from that session:
  picked mesh point and normal, alignment parameters, computed target
  pose, and plan/execute result. These are a running log of GUI usage,
  not curated datasets.
