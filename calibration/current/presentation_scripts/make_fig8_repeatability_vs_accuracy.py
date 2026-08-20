"""Figure 8: repeatability vs. accuracy, illustrated with a single real test
point (Point 8) chosen because it has the tightest repeat cluster in the
dataset (~0.07mm spread) while also having the largest miss from its
commanded IK target (~6.6mm) -- the clearest real example that repeatable
does not mean accurate. Two panels side by side, sharing one legend: left is zoomed in on just the
cluster so the 4 repeats read as distinct dots instead of one blob, right is
zoomed out to show both the cluster and the target. No arrows or distance
labels -- the point is made by the two scales alone. Same aquarel
'arctic_light' + Roboto styling as the other figures in this set.
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

FONT_DIR = os.path.join(os.path.dirname(font_roboto.__file__), "files")
for fname in ["Roboto-Regular.ttf", "Roboto-Bold.ttf"]:
    fm.fontManager.addfont(os.path.join(FONT_DIR, fname))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "build", "presentation_figures") + "/"
os.makedirs(OUT_DIR, exist_ok=True)

theme = load_theme("arctic_light")
theme.apply()
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

POINT = 8  # tightest repeatability, largest target miss -- clearest story

rows = []
with open(os.path.join(REPO_ROOT, "understanding", "data",
                        "validation_results_8point_repeatability_ARCHIVED.csv"), newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

ids = [str(POINT + 8 * k) for k in range(4)]
sel = [r for r in rows if r["test_id"] in ids]
actual = np.array([[float(r["actual_x_mm"]), float(r["actual_y_mm"])] for r in sel])
target = np.array([float(sel[0]["predicted_x_mm"]), float(sel[0]["predicted_y_mm"])])
centroid = actual.mean(axis=0)
offset_mm = np.linalg.norm(centroid - target)
spread_mm = np.max(np.linalg.norm(actual - centroid, axis=1))

REPEAT_COLOR = "#2388aa"
TARGET_COLOR = "#D32F2F"

fig, (ax_zoom, ax_wide) = plt.subplots(1, 2, figsize=(13, 6.5))

# --- right panel: zoomed out, cluster + target ------------------------------
ax_wide.scatter(actual[:, 0], actual[:, 1], s=200, color=REPEAT_COLOR,
                 edgecolors="black", linewidths=1.0, zorder=3, label="Repeated arm attempts")
ax_wide.scatter(target[0], target[1], s=200, color=TARGET_COLOR,
                 edgecolors="black", linewidths=1.0, zorder=4, label="Target IK point")

pad = 4
xs = np.concatenate([actual[:, 0], [target[0]]])
ys = np.concatenate([actual[:, 1], [target[1]]])
ax_wide.set_xlim(xs.min() - pad, xs.max() + pad)
ax_wide.set_ylim(ys.min() - pad, ys.max() + pad)
ax_wide.set_box_aspect(1)  # same physical panel size as the right subplot
ax_wide.xaxis.set_major_locator(mticker.LinearLocator(numticks=6))
ax_wide.yaxis.set_major_locator(mticker.LinearLocator(numticks=6))
ax_wide.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax_wide.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax_wide.set_xlabel("X (mm)", fontweight="bold")
ax_wide.set_ylabel("Y (mm)", fontweight="bold")
ax_wide.set_title("Zoomed out", fontweight="bold", fontsize=14)

# --- left panel: zoomed in on just the repeat cluster -----------------------
ax_zoom.scatter(actual[:, 0], actual[:, 1], s=260, color=REPEAT_COLOR,
                 edgecolors="black", linewidths=1.0, zorder=3)

# generous padding relative to the actual spread so the 4 points sit close
# together near the center of the panel, instead of stretched to its edges
zoom_pad = spread_mm * 4.5 + 0.05
ax_zoom.set_xlim(centroid[0] - zoom_pad, centroid[0] + zoom_pad)
ax_zoom.set_ylim(centroid[1] - zoom_pad, centroid[1] + zoom_pad)
ax_zoom.set_box_aspect(1)  # same physical panel size as the left subplot
ax_zoom.xaxis.set_major_locator(mticker.LinearLocator(numticks=6))
ax_zoom.yaxis.set_major_locator(mticker.LinearLocator(numticks=6))
ax_zoom.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax_zoom.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax_zoom.set_xlabel("X (mm)", fontweight="bold")
ax_zoom.set_ylabel("Y (mm)", fontweight="bold")
ax_zoom.set_title("Zoomed in on the repeats", fontweight="bold", fontsize=14)

handles, labels = ax_wide.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
                     bbox_to_anchor=(0.5, 1.02), fontsize=12.5)
legend.get_frame().set_facecolor(ax_wide.get_facecolor())
legend.get_frame().set_alpha(0.85)
legend.get_frame().set_edgecolor("#333333")

plt.tight_layout(rect=(0, 0.06, 1, 0.93))
fig.text(0.5, 0.015, "Figure 8: Repeatability is not accuracy.",
          ha="center", fontsize=12, fontweight="bold")
plt.savefig(OUT_DIR + "fig8_repeatability_vs_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
theme.apply_transforms()
print(f"Figure 8 saved. spread={spread_mm:.3f}mm offset={offset_mm:.3f}mm")