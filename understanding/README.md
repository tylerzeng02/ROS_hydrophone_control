# Understanding — Presentation Reference Package

Everything needed to reproduce or explain the figures/tables actually used in
the presentation. Only includes what made it into the deck — not the full
project history of tested-and-rejected ideas (see the main `CLAUDE.md` for
that).

## Core model used

**Optimal deployed model — 48 parameters, no Fourier/coupling/gravity terms.**
Offset (7) + tool frame (6) + base frame (6) + joint scale (7) + axis tilt (7)
+ origin corrections (8, across 3 joints) + backlash compensation (control
code, not a fitted parameter). Fourier/coupling/gravity were tested and found
NOT to improve accuracy on this dataset (0.783mm vs. 0.777mm with them —
effectively identical) — see `data/reduced_model_blocked_cv_data.csv` /
`data/reduced_model_summary.json` for that comparison.

Fit on `data/quick_calibration_test_fixed_elbow_yaw.csv` (374 poses,
`elbow_yaw` physically locked near its calibrated midpoint throughout
collection — that lock is a permanent hardware/deployment decision, not part
of the kinematic model itself).

## Figures / tables and what produced them

| Figure | Script | Data used |
|---|---|---|
| Table 1 — fitted corrections | `scripts/final_model_full_report.py` | `data/final_model_corrections_data.csv`, `data/final_model_pose_dependent_data.csv` |
| Fig. 5 — blocked CV per fold | `scripts/make_fig5_cv_and_table.py` | `data/final_model_blocked_cv_data.csv` |
| Fig. 6 — in-sample vs. blocked CV | `scripts/make_fig5_cv_and_table.py` (same file) | `data/reduced_model_summary.json` (**note:** uses the reduced/48-param model, not the full 60-param one — see script comments) |
| Fig. 4 — real-world round-trip accuracy | `scripts/make_fig4_minimal_light.py` / `make_fig4_fig5style.py` | `data/move_between_points_results_easypoints_rerun.csv` (held-out test set) |
| Fig. 7(a) — repeatability, example clusters | `scripts/make_fig7a_three_panel_example.py` | `data/validation_results_8point_repeatability_ARCHIVED.csv` |
| Fig. 7(c) — repeatability grid, all 8 points | `scripts/make_fig7c_repeatability_grid.py` | same as above |
| Table 3 — repeatability RMS summary | `scripts/make_table3_repeatability_rms.py` | same as above, output also in `data/table3_repeatability_rms_data.csv` |

## Other major test: Kabsch rotation-frame calibration

`scripts/refit_moveit_ndi_rotation.py` fits the rotation matrix that aligns
MoveIt's reported frame with NDI's measured frame
(`v_ndi = R * v_moveit`) — a prerequisite for `move_between_points`'s
accuracy numbers to mean anything, not an accuracy result on its own.

- **Calibration set:** `data/moveit_ndi_accuracy_check_new13_replay_clean.csv`
  (13 valid paired poses; kept separate from the test set on purpose — see
  main conversation history for why reusing the same points for both would
  optimistically bias the result).
- **Test set:** `data/move_between_points_results_easypoints_rerun.csv` (the
  Fig. 4 data) — a genuinely held-out set, never used to fit the rotation.
- Real historical result from an earlier drift-correction instance (not
  reproduced in this folder's data, described here for reference): stale
  rotation → **6.03mm RMS**; refit rotation → **3.44mm RMS** (~43% better).
  That number is an in-sample fit residual on the calibration points
  themselves, not a held-out accuracy test — don't present it as directly
  comparable to Fig. 4's number.

## Known caveats, worth remembering before presenting any of these numbers

- **Fig. 6 / Table 1's blocked-CV number (0.777–0.783mm) is optimistic.**
  It's fit on a narrow-range dataset (`elbow_yaw` locked, every other joint's
  range also much smaller than the arm's full reachable workspace — see
  main conversation history for the exact per-joint tick-range comparison)
  and evaluated via blocked cross-validation on the *same* dataset, not an
  independent physical test.
- **The genuinely trustworthy, independently-measured numbers are the
  physical round-trip tests:** Fig. 4's ~5.2mm RMS (`easypoints_rerun`), and
  historically ~2.86–4.58mm depending on DOF configuration (see main
  conversation history).
- **Baseline comparison:** an uncorrected/minimal (19-param: offsets + tool
  frame + base frame only) model on the larger, full-range 298-pose dataset
  scores ~23mm RMS. The fair "math corrections only" comparison (same
  dataset, same DOF, no dataset-range advantage) is ~23mm → ~5.88mm with
  scale/tilt/origin/coupling/gravity added — not the ~0.78mm headline number,
  which also benefits from the smaller-range, `elbow_yaw`-locked dataset.
