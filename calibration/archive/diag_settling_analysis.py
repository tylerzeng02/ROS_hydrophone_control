import csv
import numpy as np

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/settling_diagnostic.csv"

rows_by_pose = {}
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["valid"]) == 0:
            continue
        pid = int(row["pose_id"])
        rows_by_pose.setdefault(pid, []).append((
            int(row["elapsed_ms"]),
            float(row["tx_mm"]),
            float(row["ty_mm"]),
            float(row["tz_mm"]),
        ))

# Real inter-sample interval
all_gaps = []
for pid, rows in rows_by_pose.items():
    rows.sort()
    times = [r[0] for r in rows]
    all_gaps.extend(np.diff(times))
print(f"Real inter-sample interval: mean={np.mean(all_gaps):.1f}ms, "
      f"median={np.median(all_gaps):.1f}ms (intended was {20}ms)")
print()

print(f"{'pose':>4} {'n':>3} {'first-vs-settled(mm)':>20} {'early-avg-vs-settled(mm)':>24} {'noise-floor-std(mm)':>20}")

first_vs_settled = []
earlyavg_vs_settled = []

for pid in sorted(rows_by_pose.keys()):
    rows = sorted(rows_by_pose[pid])
    pos = np.array([[r[1], r[2], r[3]] for r in rows])
    times = np.array([r[0] for r in rows])

    if len(pos) < 5:
        continue

    # "settled" reference = mean of samples after the first 1000ms
    settled_mask = times >= 1000
    if settled_mask.sum() < 3:
        settled_mask = np.zeros_like(times, dtype=bool)
        settled_mask[-5:] = True
    settled_pos = pos[settled_mask].mean(axis=0)

    # first single sample (what "start averaging immediately" would include)
    first_delta = np.linalg.norm(pos[0] - settled_pos)

    # average of samples within first 750ms (mimics current SETTLING_TIME_MS
    # cutoff -- i.e. what the real capture loop would already have excluded)
    early_mask = times < 750
    if early_mask.sum() > 0:
        early_avg = pos[early_mask].mean(axis=0)
        early_delta = np.linalg.norm(early_avg - settled_pos)
    else:
        early_delta = float("nan")

    # noise floor: std of the settled samples themselves
    noise_std = np.linalg.norm(pos[settled_mask].std(axis=0))

    first_vs_settled.append(first_delta)
    earlyavg_vs_settled.append(early_delta)

    print(f"{pid:>4} {len(pos):>3} {first_delta:>20.3f} {early_delta:>24.3f} {noise_std:>20.3f}")

print()
print(f"Mean |first sample - settled|: {np.mean(first_vs_settled):.3f} mm (max {np.max(first_vs_settled):.3f})")
print(f"Mean |samples<750ms avg - settled|: {np.nanmean(earlyavg_vs_settled):.3f} mm (max {np.nanmax(earlyavg_vs_settled):.3f})")
