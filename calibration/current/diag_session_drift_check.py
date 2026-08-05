"""Check whether fold 5's persistent error is a session-length thermal/
voltage drift artifact rather than a kinematic modeling gap (2026-07-30).
This project's OWN comments (QUICK_TEST_POSE_INDICES in test_five_pose_
ndi_capture.cpp) already document that an earlier, superseded 200-pose
dataset suffered from "multi-hour session-length drift" -- a real,
previously-confirmed problem, not a hypothetical one. The current 298-pose
combined dataset is a merge of several collection batches, and the CSV
has a timestamp_ms column that has never been used in any analysis this
session. If fold 5's poses were mostly captured in one continuous,
multi-hour session (rather than being spread across many separate
sessions), and that session had drift, the region's elevated error could
be a data-collection artifact, not something any kinematic model
(coupling, gravity, GP) could ever fix.
"""
import csv
import numpy as np

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_combined_298.csv"

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

pose_id_all = np.array([int(r["pose_id"]) for r in rows])
timestamp_ms_all = np.array([float(r["timestamp_ms"]) for r in rows])
n = len(rows)

print(f"Dataset: {n} poses")
print(f"Timestamp range: {timestamp_ms_all.min():.0f} to {timestamp_ms_all.max():.0f} ms "
      f"({(timestamp_ms_all.max() - timestamp_ms_all.min()) / 3_600_000:.2f} hours total span)\n")

# ---------------------------------------------------------------------------
# Detect session boundaries: a large gap in timestamp between consecutive
# rows (by pose_id/row order, already confirmed as true capture order)
# signals a new capture session started, as opposed to the next pose in
# the same continuous sitting.
# ---------------------------------------------------------------------------
gaps_ms = np.diff(timestamp_ms_all)
print("Inter-pose timestamp gap percentiles (ms):")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(gaps_ms, p):.0f}")

SESSION_GAP_THRESHOLD_MS = 5 * 60 * 1000  # 5 minutes -- a gap this big means a new sitting
session_boundaries = np.where(gaps_ms > SESSION_GAP_THRESHOLD_MS)[0]
session_id = np.zeros(n, dtype=int)
current_session = 0
for i in range(1, n):
    if gaps_ms[i - 1] > SESSION_GAP_THRESHOLD_MS:
        current_session += 1
    session_id[i] = current_session

n_sessions = current_session + 1
print(f"\nDetected {n_sessions} distinct capture sessions "
      f"(gap > {SESSION_GAP_THRESHOLD_MS/60000:.0f} min signals a new sitting)")

for s in range(n_sessions):
    mask = session_id == s
    span_ms = timestamp_ms_all[mask].max() - timestamp_ms_all[mask].min()
    print(f"  session {s}: {mask.sum()} poses, pose_id "
          f"{pose_id_all[mask].min()}-{pose_id_all[mask].max()}, "
          f"span={span_ms/60000:.1f} min")

fold5_mask = (pose_id_all >= 191) & (pose_id_all <= 227)
fold5_sessions = sorted(set(session_id[fold5_mask].tolist()))
print(f"\nFold 5 (pose_id 191-227) sessions: {fold5_sessions}")
for s in fold5_sessions:
    mask = session_id == s
    span_ms = timestamp_ms_all[mask].max() - timestamp_ms_all[mask].min()
    print(f"  session {s} full span: {span_ms/60000:.1f} min, "
          f"{mask.sum()} poses total in this session "
          f"({(mask & fold5_mask).sum()} of them in fold 5)")

print("\n-> If fold 5 sits entirely inside one long session (many minutes/hours "
      "span) while other folds are spread across many short/separate sessions, "
      "that's circumstantial support for a session-drift explanation -- worth "
      "checking against the actual error next.")
