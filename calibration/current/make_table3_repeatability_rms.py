"""Table 3: simplified pose repeatability summary -- Test Point + RMS
distance of the 3 repeat visits from their own centroid. RMS chosen for
consistency with every other error figure in this deck (blocked-CV RMS,
real-world RMS), and because it weighs larger deviations more heavily than
a plain mean -- the more conservative, standard choice for a repeatability
statistic.
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
    repeats = pts[1:]
    centroid = np.mean(repeats, axis=0)
    dists = np.array([np.linalg.norm(p - centroid) for p in repeats])
    rms = np.sqrt(np.mean(dists ** 2))
    summary.append({"point": point, "rms_mm": rms})

pooled_rms = np.sqrt(np.mean([s["rms_mm"] ** 2 for s in summary]))

with open(OUT_DIR + "table3_repeatability_rms_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["point", "repeatability_rms_mm"])
    for s in summary:
        w.writerow([s["point"], f"{s['rms_mm']:.3f}"])
    w.writerow(["POOLED (all 8)", f"{pooled_rms:.3f}"])

col_labels = ["Test Point", "Repeatability, RMS (mm)"]
cell_text = [[f"Point {s['point']}", f"{s['rms_mm']:.3f}"] for s in summary]
cell_text.append(["Pooled (all 8)", f"{pooled_rms:.3f}"])

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.axis("off")
table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
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
fig.text(0.5, 0.02, "Table 3: Pose repeatability (RMS), all 8 test points.",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "table3_repeatability_rms.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Table 3 saved. Pooled RMS = {pooled_rms:.3f}mm")
