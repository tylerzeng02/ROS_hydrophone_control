"""Check whether wrist_roll (motor 6) diversity is actually narrow in the
current 298-pose dataset, and whether train/test folds share the same
range (2026-07-30) -- prompted by the question: if wrist_roll's range is
similar between train and test, poor held-out error can't be attributed to
extrapolating into unseen wrist_roll territory, since the model was never
asked to extrapolate there in the first place. That's a different question
from "would adding genuinely new wrist_roll values help future accuracy,"
which this dataset's cross-validation literally cannot speak to either way.

No fitting -- pure data inspection of actual_tick_6 / actual_rad_6 already
in the CSV.
"""
import csv
import numpy as np

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

pose_id_all = np.array([int(r["pose_id"]) for r in rows])
wrist_roll_tick = np.array([float(r["actual_tick_6"]) for r in rows])
n = len(wrist_roll_tick)

print(f"Dataset: {n} poses")
print(f"\nWrist_roll (motor 6) tick range across ALL {n} poses:")
print(f"  min={wrist_roll_tick.min():.0f}  max={wrist_roll_tick.max():.0f}  "
      f"range={wrist_roll_tick.max()-wrist_roll_tick.min():.0f}  "
      f"std={wrist_roll_tick.std():.1f}")
print(f"  percentiles: p5={np.percentile(wrist_roll_tick,5):.0f} p25={np.percentile(wrist_roll_tick,25):.0f} "
      f"p50={np.percentile(wrist_roll_tick,50):.0f} p75={np.percentile(wrist_roll_tick,75):.0f} "
      f"p95={np.percentile(wrist_roll_tick,95):.0f}")

bad_mask = (pose_id_all >= 191) & (pose_id_all <= 227)
print(f"\nWrist_roll tick range in the bad region (pose_id 191-227, {bad_mask.sum()} poses):")
print(f"  min={wrist_roll_tick[bad_mask].min():.0f}  max={wrist_roll_tick[bad_mask].max():.0f}  "
      f"std={wrist_roll_tick[bad_mask].std():.1f}")
print(f"Wrist_roll tick range OUTSIDE the bad region ({(~bad_mask).sum()} poses):")
print(f"  min={wrist_roll_tick[~bad_mask].min():.0f}  max={wrist_roll_tick[~bad_mask].max():.0f}  "
      f"std={wrist_roll_tick[~bad_mask].std():.1f}")

# Same 8 contiguous-index folds used throughout this session -- check per-fold
# train (rest) vs test (this fold) wrist_roll range overlap directly.
print("\n=== Per-fold wrist_roll range: test (held-out) vs train (rest) ===")
K = 8
fold_bounds = np.linspace(0, n, K + 1).astype(int)
for k in range(K):
    lo, hi = fold_bounds[k], fold_bounds[k + 1]
    test_mask = np.zeros(n, dtype=bool)
    test_mask[lo:hi] = True
    train_mask = ~test_mask
    test_vals = wrist_roll_tick[test_mask]
    train_vals = wrist_roll_tick[train_mask]
    overlap_lo = max(test_vals.min(), train_vals.min())
    overlap_hi = min(test_vals.max(), train_vals.max())
    test_range = test_vals.max() - test_vals.min()
    frac_test_range_covered_by_train = (
        max(0.0, overlap_hi - overlap_lo) / test_range if test_range > 0 else 1.0
    )
    print(f"  fold {k}: test range=[{test_vals.min():.0f},{test_vals.max():.0f}]  "
          f"train range=[{train_vals.min():.0f},{train_vals.max():.0f}]  "
          f"train covers {frac_test_range_covered_by_train*100:.0f}% of test's own range")
