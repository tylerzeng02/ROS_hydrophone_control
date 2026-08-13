"""Table 2: pose repeatability summary, per test point -- initial arrival
offset and repeat-visit spread (mean + max), matching ISO 9283-style
pose-repeatability reporting. Same source and per-point numbers as
figures 7/7c.
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

rows = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/build/archive/"
          "validation_results_8point_repeatability_ARCHIVED.csv", newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

summary = []
for point in range(1, 9):
    ids = [str(point + 8 * k) for k in range(4)]
    pts = [np.array([float(r["actual_x_mm"]), float(r["actual_y_mm"]), float(r["actual_z_mm"])])
           for r in rows if r["test_id"] in ids]
    first, repeats = pts[0], pts[1:]
    centroid = np.mean(repeats, axis=0)
    first_offset = np.linalg.norm(first - centroid)
    repeat_dists = [np.linalg.norm(p - centroid) for p in repeats]
    summary.append({
        "point": point,
        "initial_arrival_mm": first_offset,
        "repeat_mean_mm": np.mean(repeat_dists),
        "repeat_max_mm": np.max(repeat_dists),
    })

with open(OUT_DIR + "table2_repeatability_summary_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["point", "initial_arrival_offset_mm", "repeat_visit_mean_mm", "repeat_visit_max_mm"])
    for s in summary:
        w.writerow([s["point"], f"{s['initial_arrival_mm']:.3f}", f"{s['repeat_mean_mm']:.3f}",
                    f"{s['repeat_max_mm']:.3f}"])
    avg_initial = np.mean([s["initial_arrival_mm"] for s in summary])
    avg_repeat_mean = np.mean([s["repeat_mean_mm"] for s in summary])
    avg_repeat_max = np.mean([s["repeat_max_mm"] for s in summary])
    w.writerow(["AVERAGE", f"{avg_initial:.3f}", f"{avg_repeat_mean:.3f}", f"{avg_repeat_max:.3f}"])

col_labels = ["Test Point", "Initial Arrival\nOffset (mm)", "Repeat Visit\nSpread, Mean (mm)",
              "Repeat Visit\nSpread, Max (mm)"]
cell_text = [[f"Point {s['point']}", f"{s['initial_arrival_mm']:.3f}", f"{s['repeat_mean_mm']:.3f}",
              f"{s['repeat_max_mm']:.3f}"] for s in summary]
cell_text.append(["Average (all 8)", f"{avg_initial:.3f}", f"{avg_repeat_mean:.3f}", f"{avg_repeat_max:.3f}"])

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.axis("off")
table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.1)

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

plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02, "Table 2: Pose repeatability summary, all 8 test points (4 revisits each).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "table2_repeatability_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Table 2 saved.")
