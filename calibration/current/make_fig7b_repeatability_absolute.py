"""Figure 7b: same 8-point x 4-repeat repeatability data as Figure 7, but
plotted at ABSOLUTE NDI-measured positions instead of recentered offsets --
shows the true workspace scale, so the viewer can see for themselves how
tight each point's 4-repeat cluster is relative to how far apart the 8
target points actually are.
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

rows = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/build/archive/"
          "validation_results_8point_repeatability_ARCHIVED.csv", newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

per_point = {}
for point in range(1, 9):
    ids = [str(point + 8 * k) for k in range(4)]
    pts = [np.array([float(r["actual_x_mm"]), float(r["actual_y_mm"]), float(r["actual_z_mm"])])
           for r in rows if r["test_id"] in ids]
    per_point[point] = pts  # [1st visit, repeat1, repeat2, repeat3], absolute mm

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(8, 8))
for point, pts in per_point.items():
    color = cmap((point - 1) % 10)
    xy = np.array(pts)[:, :2]
    ax.scatter(xy[1:, 0], xy[1:, 1], s=70, color=color, zorder=3, label=f"Point {point}")
    ax.scatter(xy[0, 0], xy[0, 1], s=140, color=color, marker="X",
               edgecolors="black", linewidths=0.8, zorder=4)

ax.set_xlabel("X, absolute NDI-measured position (mm)", fontweight="bold")
ax.set_ylabel("Y, absolute NDI-measured position (mm)", fontweight="bold")
ax.set_aspect("equal", adjustable="datalim")

from matplotlib.lines import Line2D
marker_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markersize=9,
           label="Repeat visit"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="#555555", markeredgecolor="black",
           markersize=11, label="Initial arrival"),
]
point_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True,
                          title="Test point", fontsize=9)
ax.add_artist(point_legend)
legend2 = ax.legend(handles=marker_legend, loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=True)
for lg in (point_legend, legend2):
    lg.get_frame().set_facecolor(ax.get_facecolor())
    lg.get_frame().set_alpha(0.9)
    lg.get_frame().set_edgecolor("#333333")

plt.tight_layout(rect=(0, 0.09, 0.82, 1))
fig.text(0.41, 0.02,
          "Figure 7b: Same repeatability data at true workspace scale -- each cluster of 4 measurements "
          "is a single visible target point.",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig7b_repeatability_absolute.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 7b saved.")
