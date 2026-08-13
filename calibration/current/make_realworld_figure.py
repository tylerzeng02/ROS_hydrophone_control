"""Real-world move_between_points deviation figure, styled to match the
reference image: clean white background, bold caption below the plot,
minimal gridlines, clear axis labels.

Data source: ros/move_between_points_results.csv (12 target points, real
IK + real execution, current/fixed rotation matrix).
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "C:/Users/ConformalUser/Desktop/cyton_setup/build/presentation_figures/"

point_idx = []
errors = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/ros/move_between_points_results.csv", newline="") as f:
    for row in csv.DictReader(f):
        point_idx.append(int(row["point_index"]))
        errors.append(float(row["error_mm"]))

errors = np.array(errors)
mean_err = errors.mean()
rms_err = np.sqrt(np.mean(errors ** 2))

with open(OUT_DIR + "fig4_realworld_deviation_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["point_index", "deviation_mm"])
    for p, e in zip(point_idx, errors):
        w.writerow([p, f"{e:.3f}"])
    w.writerow(["mean", f"{mean_err:.3f}"])
    w.writerow(["rms", f"{rms_err:.3f}"])

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

fig, ax = plt.subplots(figsize=(8, 5))
labels = [str(p + 1) for p in point_idx]
bars = ax.bar(labels, errors, color="#1565C0", width=0.6, zorder=3)
ax.axhline(mean_err, color="#D32F2F", linestyle="--", linewidth=1.5, zorder=4,
           label=f"Mean = {mean_err:.2f} mm")
for b, e in zip(bars, errors):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{e:.2f}",
            ha="center", va="bottom", fontsize=8.5)
ax.set_xlabel("Target point")
ax.set_ylabel("Position deviation (mm)")
ax.set_ylim(0, max(errors) * 1.18)
ax.legend(loc="upper right", frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02,
          f"Figure 4: Real-world round-trip accuracy across 12 test points "
          f"(mean {mean_err:.2f} mm, RMS {rms_err:.2f} mm).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig4_realworld_deviation.png", dpi=150)
plt.close()

print(f"Mean: {mean_err:.3f}mm  RMS: {rms_err:.3f}mm  Max: {errors.max():.3f}mm")
print("Saved fig4_realworld_deviation.png + data CSV")
