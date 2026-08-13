"""Figure 7: recentered repeatability scatter across all 8 test points.
Each point's 4 measurements (1st visit + 3 steady-state repeats) are
re-expressed as offsets from THAT point's own steady-state centroid, then
all 8 points are overlaid on one plot -- since the 8 points are physically
scattered across the workspace, this is the only way to compare their
clustering tightness directly, on one shared set of axes.

Source: build/archive/validation_results_8point_repeatability_ARCHIVED.csv
(8 points x 4 round-robin repeats, re-archived this session after the live
validation_results.csv was overwritten by an unrelated later run).
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
    first, repeats = pts[0], pts[1:]
    centroid = np.mean(repeats, axis=0)
    per_point[point] = {
        "first_rel": first - centroid,
        "repeats_rel": [p - centroid for p in repeats],
    }

with open(OUT_DIR + "fig7_repeatability_all8_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["point", "visit", "dx_mm_from_own_centroid", "dy_mm_from_own_centroid",
                "dz_mm_from_own_centroid"])
    for point, d in per_point.items():
        w.writerow([point, "1st visit", f"{d['first_rel'][0]:.4f}", f"{d['first_rel'][1]:.4f}",
                    f"{d['first_rel'][2]:.4f}"])
        for i, r in enumerate(d["repeats_rel"]):
            w.writerow([point, f"repeat {i + 1}", f"{r[0]:.4f}", f"{r[1]:.4f}", f"{r[2]:.4f}"])

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(7.5, 7.5))
for point, d in per_point.items():
    color = cmap((point - 1) % 10)
    rep_xy = np.array(d["repeats_rel"])[:, :2]
    ax.scatter(rep_xy[:, 0], rep_xy[:, 1], s=70, color=color, zorder=3,
               label=f"Point {point}")
    ax.scatter(d["first_rel"][0], d["first_rel"][1], s=140, color=color, marker="X",
               edgecolors="black", linewidths=0.8, zorder=4)

ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
ax.axvline(0, color="#999999", linewidth=0.8, zorder=1)
ax.set_xlabel("X offset from own repeat-visit centroid (mm)", fontweight="bold")
ax.set_ylabel("Y offset from own repeat-visit centroid (mm)", fontweight="bold")
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
          "Figure 7: Repeatability across all 8 test points, recentered to each point's own "
          "repeat-visit centroid.",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig7_repeatability_all8.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 7 saved.")
