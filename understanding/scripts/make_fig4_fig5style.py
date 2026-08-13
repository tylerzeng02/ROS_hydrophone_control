"""Fig. 4 data (12-point real-world round-trip accuracy), restyled to match
Figure 5's exact convention: RMS-based dashed reference line (not mean),
same y-limit factor (1.25x), same legend/label formatting.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme
import font_roboto

FONT_DIR = os.path.join(os.path.dirname(font_roboto.__file__), "files")
for fname in ["Roboto-Regular.ttf", "Roboto-Bold.ttf"]:
    fm.fontManager.addfont(os.path.join(FONT_DIR, fname))

OUT_DIR = "C:/Users/ConformalUser/Desktop/cyton_setup/build/presentation_figures/"

theme = load_theme("arctic_light")
theme.apply()
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

point_idx = []
errors = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/ros/move_between_points_results_easypoints_rerun.csv", newline="") as f:
    for row in csv.DictReader(f):
        point_idx.append(int(row["point_index"]))
        errors.append(float(row["error_mm"]))
errors = np.array(errors)
rms_err = np.sqrt(np.mean(errors ** 2))

fig, ax = plt.subplots(figsize=(8, 5))
labels = [str(p + 1) for p in point_idx]
bars = ax.bar(labels, errors, color="#2388aa", zorder=3)
ax.axhline(rms_err, color="#D32F2F", linestyle="--", linewidth=1.5, zorder=4,
           label=f"Pooled = {rms_err:.2f} mm")
ax.set_xlabel("Target point", fontweight="bold")
ax.set_ylabel("Position deviation (mm)", fontweight="bold")
ax.set_ylim(0, max(errors) * 1.25)
legend = ax.legend(loc="upper right", frameon=True)
legend.get_frame().set_facecolor(ax.get_facecolor())
legend.get_frame().set_alpha(0.75)
legend.get_frame().set_edgecolor("#333333")
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02,
          f"Figure 4: Real-world round-trip accuracy across 12 test points "
          f"(pooled RMS {rms_err:.2f} mm).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig4_realworld_fig5style.png", dpi=150)
plt.close()
print(f"Saved fig4_realworld_fig5style.png. Pooled RMS = {rms_err:.3f}mm, "
      f"per-point range {errors.min():.3f}-{errors.max():.3f}mm")
