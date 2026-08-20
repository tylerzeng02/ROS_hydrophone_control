"""Figure 4 only, restyled with aquarel's 'arctic_light' theme, real Roboto
font (via the font-roboto PyPI package, actual TTF files registered with
matplotlib -- not just requested by name and silently falling back),
slightly larger tick labels, and bold axis titles.
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "build", "presentation_figures") + "/"

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
with open(os.path.join(REPO_ROOT, "ros", "move_between_points_results_easypoints_rerun.csv"), newline="") as f:
    for row in csv.DictReader(f):
        point_idx.append(int(row["point_index"]))
        errors.append(float(row["error_mm"]))
errors = np.array(errors)
mean_err = errors.mean()
rms_err = np.sqrt(np.mean(errors ** 2))

fig, ax = plt.subplots(figsize=(8, 5))
labels = [str(p + 1) for p in point_idx]
bars = ax.bar(labels, errors, color="#2388aa", zorder=3)
ax.axhline(mean_err, color="#D32F2F", linestyle="--", linewidth=1.5, zorder=4,
           label=f"Mean = {mean_err:.2f} mm")
ax.set_xlabel("Target point", fontweight="bold")
ax.set_ylabel("Position deviation (mm)", fontweight="bold")
ax.set_ylim(0, max(errors) * 1.18)
legend = ax.legend(loc="upper right", frameon=True)
legend.get_frame().set_facecolor(ax.get_facecolor())
legend.get_frame().set_alpha(0.75)
legend.get_frame().set_edgecolor("#333333")
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02,
          f"Figure 4: Real-world round-trip accuracy across 12 test points "
          f"(mean {mean_err:.2f} mm, RMS {rms_err:.2f} mm).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig4_realworld_deviation.png", dpi=150)
plt.close()

theme.apply_transforms()
print("Regenerated fig4 with aquarel 'minimal_light' theme.")
