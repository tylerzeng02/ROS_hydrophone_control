"""Fig. 7. Composite repeatability figure: (a) the 8-point small-multiples
grid, (b) the RMS repeatability summary table -- combined into one figure
with a proper scientific caption.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

rows = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/build/archive/"
          "validation_results_8point_repeatability_ARCHIVED.csv", newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

per_point = {}
rms_summary = []
for point in range(1, 9):
    ids = [str(point + 8 * k) for k in range(4)]
    pts = [np.array([float(r["actual_x_mm"]), float(r["actual_y_mm"])]) for r in rows if r["test_id"] in ids]
    centroid = np.mean(pts[1:], axis=0)
    per_point[point] = [p - centroid for p in pts]
    dists = np.array([np.linalg.norm(p - centroid) for p in pts[1:]])
    rms_summary.append(np.sqrt(np.mean(dists ** 2)))
pooled_rms = np.sqrt(np.mean(np.array(rms_summary) ** 2))

AXIS_LIM = 0.5
TICK_STEP = 0.25
cmap = plt.get_cmap("tab10")

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 4, height_ratios=[0.12, 1, 0.85], hspace=0.55, wspace=0.32,
                        figure=fig, top=0.94, bottom=0.05, left=0.05, right=0.98)

# --- legend row ---
legend_ax = fig.add_subplot(gs[0, :])
legend_ax.axis("off")
marker_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markersize=13, label="Repeat visit"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="#555555", markeredgecolor="black",
           markersize=15, label="Initial arrival"),
]
legend = legend_ax.legend(handles=marker_legend, loc="center", ncol=2, frameon=True, fontsize=13)
legend.get_frame().set_facecolor("#E8ECEF")
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_edgecolor("#333333")

# --- (a) 8-panel grid, row 1 ---
grid_axes = [fig.add_subplot(gs[1, c]) for c in range(4)]
gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :])  # placeholder unused

fig2_axes = []
positions = [(1, c) for c in range(4)]
axes_a = []
for point, c in zip(range(1, 5), range(4)):
    ax = fig.add_subplot(gs[1, c])
    axes_a.append(ax)
for point, c in zip(range(5, 9), range(4)):
    pass

# Rebuild properly: need 2 rows for 8 panels -> use nested gridspec inside row 1
gs_top = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, :], hspace=0.55, wspace=0.32)
for point in range(1, 9):
    r, c = divmod(point - 1, 4)
    ax = fig.add_subplot(gs_top[r, c])
    color = cmap((point - 1) % 10)
    pts = np.array(per_point[point])
    first, repeats = pts[0], pts[1:]
    ax.scatter(repeats[:, 0], repeats[:, 1], s=80, color=color, zorder=3)
    ax.scatter(first[0], first[1], s=150, color=color, marker="X", edgecolors="black",
               linewidths=0.9, zorder=4)
    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(TICK_STEP))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(TICK_STEP))
    ax.axhline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.set_title(f"Point {point}", fontweight="bold", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    if r == 1:
        ax.set_xlabel("X offset (mm)", fontsize=9.5)
    if c == 0:
        ax.set_ylabel("Y offset (mm)", fontsize=9.5)

fig.text(0.06, 0.895, "(a)", fontsize=18, fontweight="bold")

# --- (b) summary table, row 2 ---
table_ax = fig.add_subplot(gs[2, :])
table_ax.axis("off")
col_labels = ["Test Point", "Repeatability, RMS (mm)"]
cell_text = [[f"Point {p}", f"{rms_summary[p - 1]:.3f}"] for p in range(1, 9)]
cell_text.append(["Pooled (all 8)", f"{pooled_rms:.3f}"])

table = table_ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
                        bbox=[0.28, 0.0, 0.44, 1.0])
table.auto_set_font_size(False)
table.set_fontsize(11)

HEADER_COLOR = "#2388aa"
n_rows = len(cell_text) + 1
for (row_i, col_i), cell in table.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row_i == 0:
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")
    elif row_i == n_rows - 1:
        cell.set_facecolor("#D9E6EA")
        cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor("#F5F7F8" if row_i % 2 == 0 else "white")

fig.text(0.06, 0.335, "(b)", fontsize=18, fontweight="bold")

fig.text(0.5, 0.005,
          "Fig. 7. Pose repeatability across 8 test points (4 revisits each). (a) Per-point offset from "
          "repeat-visit centroid, all panels on a shared \u00b10.5 mm scale for direct comparison. "
          "(b) Repeatability (RMS) summary per point.",
          ha="center", fontsize=11.5, fontweight="bold", wrap=True)

plt.savefig(OUT_DIR + "fig7_composite.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Fig. 7 composite saved. Pooled RMS = {pooled_rms:.3f}mm")
