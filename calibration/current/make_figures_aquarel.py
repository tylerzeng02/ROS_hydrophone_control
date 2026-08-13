"""Regenerates fig1a, fig1b, fig3, fig4 using the aquarel 'scientific' theme
for consistent, clean styling across all presentation figures. Same source
data as before (see make_presentation_figures.py / make_realworld_figure.py
for provenance notes) -- this script only changes the visual styling.
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from aquarel import load_theme

OUT_DIR = "C:/Users/ConformalUser/Desktop/cyton_setup/build/presentation_figures/"

theme = load_theme("scientific")
theme.apply()

# =====================================================================
# FIGURE 1a: axis tilt per joint
# =====================================================================
JOINT_NAMES = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow_pitch",
               "elbow_yaw", "wrist_pitch", "wrist_roll"]
NOMINAL_AXES = np.array([
    [0, 0, 1], [1, 0, 0], [0, -1, 0], [1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 0, 1],
], dtype=float)
CORRECTED_AXES = np.array([
    [0.013792, 0.014877, 0.999794],
    [0.998997, -0.043573, -0.010357],
    [-0.027678, -0.999437, 0.018973],
    [0.999677, -0.024005, 0.008304],
    [0, -1, 0],
    [0.999044, 0.042957, 0.008105],
    [-0.006451, -0.018572, 0.999807],
], dtype=float)
APPLIED = [True, True, True, True, False, True, True]

tilt_deg = []
for n, c in zip(NOMINAL_AXES, CORRECTED_AXES):
    n = n / np.linalg.norm(n); c = c / np.linalg.norm(c)
    tilt_deg.append(np.degrees(np.arccos(np.clip(np.dot(n, c), -1, 1))))
tilt_deg = np.array(tilt_deg)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#1565C0" if a else "#B0BEC5" for a in APPLIED]
bars = ax.bar(JOINT_NAMES, tilt_deg, color=colors, zorder=3)
ax.set_ylabel("Axis tilt from CAD-nominal (deg)")
ax.set_xticks(range(len(JOINT_NAMES)))
ax.set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
for b, t in zip(bars, tilt_deg):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{t:.2f}",
             ha="center", va="bottom", fontsize=9)
ax.legend(handles=[
    Patch(color="#1565C0", label="Applied to deployed model"),
    Patch(color="#B0BEC5", label="Not applied (elbow_yaw locked)"),
], loc="upper left", frameon=False, fontsize=8.5)
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02, "Figure 1: Fitted joint rotation-axis tilt vs. nominal URDF axis.",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig1a_axis_tilt.png", dpi=150)
plt.close()

# =====================================================================
# FIGURE 1b: origin correction magnitude
# =====================================================================
NOMINAL_ORIGIN = {
    "shoulder_yaw": np.array([-0.0215, -0.0205, 0.1255]),
    "elbow_pitch": np.array([0.018, 0.0206, 0.1158]),
    "wrist_pitch": np.array([0.02626, 0.018, 0.0718]),
}
CORRECTED_ORIGIN = {
    "shoulder_yaw": np.array([-0.02478414, -0.0205, 0.1308452]),
    "elbow_pitch": np.array([0.01656849, 0.02722018, 0.11356304]),
    "wrist_pitch": np.array([0.02765348, 0.01273746, 0.07244612]),
}
origin_rows = []
for j in ["shoulder_yaw", "elbow_pitch", "wrist_pitch"]:
    delta_mm = (CORRECTED_ORIGIN[j] - NOMINAL_ORIGIN[j]) * 1000.0
    origin_rows.append((j, delta_mm))

fig, ax = plt.subplots(figsize=(6, 4.5))
names = [r[0] for r in origin_rows]
mags = [np.linalg.norm(r[1]) for r in origin_rows]
bars = ax.bar(names, mags, color="#1565C0", zorder=3)
ax.set_ylabel("Origin correction magnitude (mm)")
for b, m in zip(bars, mags):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{m:.2f}mm",
             ha="center", va="bottom", fontsize=9)
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02, "Figure 2: Fitted joint-origin position correction vs. CAD-nominal.",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig1b_origin_correction.png", dpi=150)
plt.close()

# =====================================================================
# FIGURE 3: repeatability scatter
# =====================================================================
rows = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/build/archive/"
          "validation_results_8point_repeatability_ARCHIVED.csv") as f:
    for row in csv.DictReader(f):
        rows.append(row)
point1_ids = ["1", "9", "17", "25"]
pts = []
for r in rows:
    if r["test_id"] in point1_ids:
        pts.append((float(r["actual_x_mm"]), float(r["actual_y_mm"]), float(r["actual_z_mm"]), r["test_id"]))

xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(xs[1:], ys[1:], s=130, color="#1565C0", label="Repeat visits (2nd-4th)", zorder=3)
ax.scatter(xs[0], ys[0], s=190, color="#D32F2F", marker="X", label="Initial arrival", zorder=4)
for i, (x, y, _, _) in enumerate(pts):
    ax.annotate(f"{i + 1}", (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9)
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
ax.legend(frameon=False)
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02, "Figure 3: Repeatability across 4 revisits to the same commanded point.",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig3_repeatability.png", dpi=150)
plt.close()

# =====================================================================
# FIGURE 4: real-world deviation
# =====================================================================
point_idx = []
errors = []
with open("C:/Users/ConformalUser/Desktop/cyton_setup/ros/move_between_points_results.csv", newline="") as f:
    for row in csv.DictReader(f):
        point_idx.append(int(row["point_index"]))
        errors.append(float(row["error_mm"]))
errors = np.array(errors)
mean_err = errors.mean(); rms_err = np.sqrt(np.mean(errors ** 2))

fig, ax = plt.subplots(figsize=(8, 5))
labels = [str(p + 1) for p in point_idx]
bars = ax.bar(labels, errors, color="#1565C0", zorder=3)
ax.axhline(mean_err, color="#D32F2F", linestyle="--", linewidth=1.5, zorder=4,
           label=f"Mean = {mean_err:.2f} mm")
for b, e in zip(bars, errors):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{e:.2f}",
            ha="center", va="bottom", fontsize=8.5)
ax.set_xlabel("Target point")
ax.set_ylabel("Position deviation (mm)")
ax.set_ylim(0, max(errors) * 1.18)
ax.legend(loc="upper right", frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(0.5, 0.02,
          f"Figure 4: Real-world round-trip accuracy across 12 test points "
          f"(mean {mean_err:.2f} mm, RMS {rms_err:.2f} mm).",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig4_realworld_deviation.png", dpi=150)
plt.close()

theme.apply_transforms()
print("Regenerated fig1a, fig1b, fig3, fig4 with aquarel 'scientific' theme.")
