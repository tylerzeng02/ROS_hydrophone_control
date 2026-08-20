"""Figure 7c: small-multiples repeatability grid -- 8 panels, one per test
point, each showing that point's measurements as an OFFSET from its own
steady-state centroid (not absolute position), on a SHARED, fixed axis
scale and tick spacing across all 8 panels. Sharing the scale (rather than
each panel auto-zooming to its own data) is what makes cross-panel
comparison honest: a visually bigger spread on the page always means a
genuinely bigger spread in the data, not just a more-zoomed-in panel.
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
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
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
    pts = [np.array([float(r["actual_x_mm"]), float(r["actual_y_mm"])])
           for r in rows if r["test_id"] in ids]
    centroid = np.mean(pts[1:], axis=0)  # steady-state centroid (repeats only)
    per_point[point] = [p - centroid for p in pts]  # offsets, 1st visit first

# One shared axis range + tick spacing for every panel, sized to comfortably
# fit the single worst offset seen anywhere in the dataset (point 5's 1st
# visit, ~0.43mm), then rounded out a bit further per this session's request
# to zoom out slightly so points read as visibly clustered rather than
# maximally tight.
AXIS_LIM = 0.5
TICK_STEP = 0.25

cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(2, 4, figsize=(18, 10.5))

for point, ax in zip(range(1, 9), axes.flat):
    color = cmap((point - 1) % 10)
    pts = np.array(per_point[point])
    first, repeats = pts[0], pts[1:]

    ax.scatter(repeats[:, 0], repeats[:, 1], s=140, color=color, zorder=3)
    ax.scatter(first[0], first[1], s=260, color=color, marker="X",
               edgecolors="black", linewidths=1.1, zorder=4)

    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(TICK_STEP))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(TICK_STEP))
    ax.axhline(0, color="#999999", linewidth=0.7, zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.7, zorder=1)
    ax.set_title(f"Point {point}", fontweight="bold", fontsize=15)
    ax.set_aspect("equal", adjustable="box")

marker_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markersize=16,
           label="Repeat visit"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="#555555", markeredgecolor="black",
           markersize=18, label="Initial arrival"),
]
legend = fig.legend(handles=marker_legend, loc="upper center", ncol=2, frameon=True,
                     bbox_to_anchor=(0.5, 1.04), fontsize=16)
legend.get_frame().set_facecolor(axes.flat[0].get_facecolor())
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_edgecolor("#333333")

plt.subplots_adjust(hspace=0.45, wspace=0.3)
plt.tight_layout(rect=(0.01, 0.05, 1, 0.92))
fig.text(0.5, 0.01,
          "Figure 7c: Repeatability per test point, all 8 panels on the same shared scale "
          "for direct comparison.",
          ha="center", fontsize=13, fontweight="bold")
plt.savefig(OUT_DIR + "fig7c_repeatability_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 7c saved.")
