"""Fig. 7(a): orthogonal XY / XZ / YZ projections of one representative
test point's repeat-visit cluster, offset from its own 3D centroid (mean
of the 3 repeat visits only, matching the RMS table's methodology).
Illustrative example only -- the full 8-point summary is reported via the
separate RMS table, not duplicated here.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from aquarel import load_theme
import font_roboto
from matplotlib.lines import Line2D

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

EXAMPLE_POINTS = [1, 7]  # Point 1: RMS 0.090mm (near median); Point 7: RMS 0.212mm (worst case)

rows = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/build/archive/"
          "validation_results_8point_repeatability_ARCHIVED.csv", newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

AXIS_LIM = 0.3
TICK_STEP = 0.1
color = "#2388aa"

panel_axes = [
    ("X offset (mm)", "Y offset (mm)", 0, 1, "XY plane"),
    ("X offset (mm)", "Z offset (mm)", 0, 2, "XZ plane"),
    ("Y offset (mm)", "Z offset (mm)", 1, 2, "YZ plane"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for row_i, point in enumerate(EXAMPLE_POINTS):
    ids = [str(point + 8 * k) for k in range(4)]
    pts = np.array([[float(r["actual_x_mm"]), float(r["actual_y_mm"]), float(r["actual_z_mm"])]
                    for r in rows if r["test_id"] in ids])
    first, repeats = pts[0], pts[1:]
    centroid = np.mean(repeats, axis=0)  # 3D centroid, repeat visits only

    first_off = first - centroid
    repeats_off = repeats - centroid

    for col_i, (xlabel, ylabel, ia, ib, title) in enumerate(panel_axes):
        ax = axes[row_i, col_i]
        ax.scatter(repeats_off[:, ia], repeats_off[:, ib], s=140, color=color, zorder=3)
        ax.scatter(first_off[ia], first_off[ib], s=260, color=color, marker="X",
                   edgecolors="black", linewidths=1.1, zorder=4)
        ax.set_xlim(-AXIS_LIM, AXIS_LIM)
        ax.set_ylim(-AXIS_LIM, AXIS_LIM)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(TICK_STEP))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(TICK_STEP))
        ax.axhline(0, color="#999999", linewidth=0.7, zorder=1)
        ax.axvline(0, color="#999999", linewidth=0.7, zorder=1)
        ax.set_title(f"Point {point} - {title}", fontweight="bold", fontsize=13)
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=11)
        ax.set_aspect("equal", adjustable="box")

marker_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=14, label="Repeat visit"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor=color, markeredgecolor="black",
           markersize=16, label="Initial arrival"),
]
legend = fig.legend(handles=marker_legend, loc="upper center", ncol=2, frameon=True,
                     bbox_to_anchor=(0.5, 1.03), fontsize=13)
legend.get_frame().set_facecolor(axes[0, 0].get_facecolor())
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_edgecolor("#333333")

plt.subplots_adjust(hspace=0.5, wspace=0.35)
plt.tight_layout(rect=(0.01, 0.03, 1, 0.94))
fig.text(0.5, 0.005,
          "Fig. 7(a). Repeat-visit clusters for Point 1 (near-median repeatability) and "
          "Point 7 (worst-case repeatability), orthogonal projections offset from own 3D centroid.",
          ha="center", fontsize=12.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig7a_three_panel_example.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Fig. 7(a) saved for Points {EXAMPLE_POINTS}.")
