"""Figure 5 (pooled blocked-CV error per fold, for the final <1mm model)
and the final corrections table -- both styled identically to Figure 4:
aquarel 'arctic_light' theme, real Roboto font, azure (#2388aa) accents,
bold black axis text, boxed opaque legend.
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

# =====================================================================
# FIGURE 5: pooled blocked-CV test RMS per fold (final model, no ablation)
# =====================================================================
folds = []
with open(OUT_DIR + "final_model_blocked_cv_data.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["fold"] == "POOLED":
            pooled_rms = float(row["test_rms_mm"])
            continue
        folds.append((int(row["fold"]), float(row["test_rms_mm"])))

fig, ax = plt.subplots(figsize=(8, 5))
labels = [str(f + 1) for f, _ in folds]
values = [v for _, v in folds]
bars = ax.bar(labels, values, color="#2388aa", zorder=3)
ax.axhline(pooled_rms, color="#D32F2F", linestyle="--", linewidth=1.5, zorder=4,
           label=f"Pooled = {pooled_rms:.2f} mm")
ax.set_xlabel("Fold", fontweight="bold")
ax.set_ylabel("Held-out test RMS (mm)", fontweight="bold")
ax.set_ylim(0, max(values) * 1.25)
legend = ax.legend(loc="upper right", frameon=True)
legend.get_frame().set_facecolor(ax.get_facecolor())
legend.get_frame().set_alpha(0.75)
legend.get_frame().set_edgecolor("#333333")
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02,
          f"Figure 5: Blocked 8-fold cross-validation error, final calibration model "
          f"(pooled RMS {pooled_rms:.2f} mm).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig5_blocked_cv_per_fold.png", dpi=150)
plt.close()
print(f"Figure 5 saved. Pooled RMS = {pooled_rms:.3f}mm, folds range "
      f"{min(values):.3f}-{max(values):.3f}mm")

# =====================================================================
# FINAL CORRECTIONS TABLE -- one single consolidated table (every fitted
# parameter, joint-level AND pose-dependent, in one grid).
# =====================================================================
rows = []
with open(OUT_DIR + "final_model_corrections_data.csv", newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

pose_dep = []
with open(OUT_DIR + "final_model_pose_dependent_data.csv", newline="") as f:
    for row in csv.DictReader(f):
        pose_dep.append(row)

col_labels = ["Joint / Term", "Offset (deg)", "Scale", "Axis Tilt (deg)",
              "Origin \u0394 (mm)", "Gravity (rad/m)", "Pose-Dependent"]
cell_text = []
for r in rows:
    origin = "-"
    if r["origin_dx_mm"]:
        dx, dy, dz = float(r["origin_dx_mm"]), float(r["origin_dy_mm"]), float(r["origin_dz_mm"])
        mag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        origin = f"{mag:.2f}"
    is_locked = r["joint"] == "elbow_yaw"
    cell_text.append([
        r["joint"].replace("_", " "),
        "n/a (locked)" if is_locked else r["offset_deg"],
        "n/a (locked)" if is_locked else r["scale"][:8],
        "n/a (locked)" if is_locked else r["axis_tilt_deg"],
        origin,
        r["gravity_coeff_rad_per_m"],
        "-",
    ])

pose_dep_labels = [
    "Fourier: shoulder_pitch (sin)", "Fourier: shoulder_pitch (cos)",
    "Coupling: shoulder_roll x shoulder_yaw -> shoulder_yaw",
    "Coupling: shoulder_yaw x elbow_yaw -> elbow_yaw",
    "Coupling: shoulder_pitch x elbow_pitch -> elbow_pitch",
]
for label, p in zip(pose_dep_labels, pose_dep):
    cell_text.append([label, "-", "-", "-", "-", "-", f"{float(p['value']):+.4f}"])

fig, ax = plt.subplots(figsize=(12, 6.2))
ax.axis("off")
table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.9)
table.auto_set_column_width(col=list(range(len(col_labels))))

HEADER_COLOR = "#2388aa"
for (row_i, col_i), cell in table.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row_i == 0:
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#F5F7F8" if row_i % 2 == 0 else "white")
        if col_i == 0:
            cell.set_text_props(ha="left")

plt.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.5, 0.02,
          "Table 1: All fitted corrections, optimal model (blocked-CV RMS "
          f"{pooled_rms:.2f} mm).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "table1_final_corrections.png", dpi=150, bbox_inches="tight")
plt.close()
print("Corrections table saved.")

# =====================================================================
# FIGURE 6: in-sample vs. blocked-CV, REDUCED model (no Fourier/coupling/
# gravity -- the 48-param model actually deployed to robot_calibration.cpp
# and the URDF, since pose-dependent terms can't live in a static model).
# =====================================================================
with open(OUT_DIR + "reduced_model_summary.json") as f:
    import json
    summary = json.load(f)

in_sample_rms = summary["full_dataset_rms_mm"]
blocked_cv_rms = summary["pooled_blocked_cv_rms_mm"]

fig, ax = plt.subplots(figsize=(4.2, 5.5))
x = [0, 0.5]
labels = ["In-Sample", "Blocked CV"]
values = [in_sample_rms, blocked_cv_rms]
colors = ["#8FC1D4", "#2388aa"]
bars = ax.bar(x, values, color=colors, width=0.3, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim(-0.3, 0.8)
ax.set_xlabel("Evaluation Method", fontweight="bold")
ax.set_ylabel("RMS Error (mm)", fontweight="bold")
ax.set_ylim(0, max(values) * 1.25)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.2f}",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02, "Figure 6: In-sample vs. blocked-CV accuracy, deployed calibration model "
          "(no pose-dependent terms).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig6_insample_vs_blockedcv.png", dpi=150)
plt.close()
print(f"Figure 6 saved. In-sample={in_sample_rms:.3f}mm, blocked-CV={blocked_cv_rms:.3f}mm")
