"""Generates the three presentation figures requested:
  1. Axis-tilt (+ origin correction) bar charts -- URDF changes
  2. In-sample vs. blocked-CV grouped bar chart -- calibration methodology
  3. Repeatability scatter plot -- first-visit vs. steady-state

All source numbers are pulled directly from the deployed URDF, the
calibration model's nominal geometry, and the project's documented
blocked-CV results -- nothing here is fabricated or eyeballed. See each
section's comment for exact provenance.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

OUT_DIR = "C:/Users/ConformalUser/Desktop/cyton_setup/build/presentation_figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# FIGURE 1a: axis tilt per joint.
# Nominal axes from calibrate_kinematics.py's JOINT_AXES; corrected axes
# read directly from references/cyton_gamma_1500_trac_ik.urdf's <axis>
# tags (2026-08-10). elbow_yaw is deliberately NOT corrected (locked,
# poorly identified -- see robot_calibration.cpp's own comment).
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
    n = n / np.linalg.norm(n)
    c = c / np.linalg.norm(c)
    tilt_deg.append(np.degrees(np.arccos(np.clip(np.dot(n, c), -1, 1))))
tilt_deg = np.array(tilt_deg)

with open(OUT_DIR + "fig1a_axis_tilt_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["joint", "tilt_deg", "applied_to_deployed_model"])
    for name, t, a in zip(JOINT_NAMES, tilt_deg, APPLIED):
        w.writerow([name, f"{t:.3f}", a])

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2E7D32" if a else "#BDBDBD" for a in APPLIED]
bars = ax.bar(JOINT_NAMES, tilt_deg, color=colors)
ax.set_ylabel("Axis tilt from CAD-nominal (deg)")
ax.set_title("Fitted joint rotation-axis tilt vs. nominal URDF axis")
ax.set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
for b, t in zip(bars, tilt_deg):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{t:.2f}",
             ha="center", va="bottom", fontsize=9)
ax.legend(handles=[
    Patch(color="#2E7D32", label="Applied to deployed model"),
    Patch(color="#BDBDBD", label="Not applied (elbow_yaw locked, poorly identified)"),
], loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR + "fig1a_axis_tilt.png", dpi=150)
plt.close()

# =====================================================================
# FIGURE 1b: origin correction magnitude, the 3 joints that got one.
# Nominal origins from calibrate_kinematics.py's JOINT_ORIGINS_M;
# corrected origins from the URDF's <origin xyz> on those same joints.
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

with open(OUT_DIR + "fig1b_origin_correction_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["joint", "dx_mm", "dy_mm", "dz_mm", "magnitude_mm"])
    for j, d in origin_rows:
        w.writerow([j, f"{d[0]:.3f}", f"{d[1]:.3f}", f"{d[2]:.3f}", f"{np.linalg.norm(d):.3f}"])

fig, ax = plt.subplots(figsize=(6, 4))
names = [r[0] for r in origin_rows]
mags = [np.linalg.norm(r[1]) for r in origin_rows]
bars = ax.bar(names, mags, color="#1565C0")
ax.set_ylabel("Origin correction magnitude (mm)")
ax.set_title("Fitted joint-origin position correction vs. CAD-nominal")
for b, m in zip(bars, mags):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{m:.2f}mm",
             ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR + "fig1b_origin_correction.png", dpi=150)
plt.close()

# =====================================================================
# FIGURE 2: in-sample vs blocked-CV RMS across model stages.
# Numbers from CLAUDE.md's documented 298-pose-dataset fitting history
# (joint-coupling and gravity-deflection sections, 2026-07-29).
# =====================================================================
stages = ["Tilt+scale+origin\n+Fourier (38p)", "+Joint coupling\n(53p)",
          "+Gravity/elastostatic\n(60p)"]
in_sample = [8.187, 6.874, 5.876]
blocked_cv = [11.85, 9.87, 8.91]

with open(OUT_DIR + "fig2_blocked_cv_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model_stage", "in_sample_rms_mm", "blocked_cv_rms_mm"])
    for s, i, b in zip(stages, in_sample, blocked_cv):
        w.writerow([s.replace("\n", " "), i, b])
    w.writerow(["+GP residual layer (67p, CV-only, no comparable in-sample number)", "", 7.81])

x = np.arange(len(stages))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width / 2, in_sample, width, label="In-sample (full-dataset fit)", color="#90CAF9")
b2 = ax.bar(x + width / 2, blocked_cv, width, label="Blocked 8-fold CV (honest, held-out)", color="#1565C0")
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel("RMS error (mm)")
ax.set_title("In-sample vs. blocked-CV accuracy across model corrections\n(298-pose dataset)")
ax.legend()
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR + "fig2_blocked_cv.png", dpi=150)
plt.close()

# =====================================================================
# FIGURE 3: repeatability scatter, point 1 (test_id 1, 9, 17, 25).
# Source: build/archive/validation_results_8point_repeatability_ARCHIVED.csv
# (re-archived this session, since the live validation_results.csv has
# since been overwritten by a newer, unrelated --validate run).
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

with open(OUT_DIR + "fig3_repeatability_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["visit", "test_id", "actual_x_mm", "actual_y_mm", "actual_z_mm"])
    for i, (x, y, z, tid) in enumerate(pts):
        label = "1st visit" if i == 0 else f"repeat {i}"
        w.writerow([label, tid, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])

xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(xs[1:], ys[1:], s=120, color="#1565C0", label="Steady-state repeats (2nd-4th visit)", zorder=3)
ax.scatter(xs[0], ys[0], s=180, color="#D32F2F", marker="X", label="1st visit", zorder=4)
for i, (x, y, _, _) in enumerate(pts):
    ax.annotate(f"{i + 1}", (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title("Repeatability: 4 revisits to the same commanded point\n(NDI-measured position, X-Y plane)")
ax.legend()
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig(OUT_DIR + "fig3_repeatability.png", dpi=150)
plt.close()

print("Done. Figures + data tables written to", OUT_DIR)
print()
print("=== Figure 1a: axis tilt (deg) ===")
for name, t, a in zip(JOINT_NAMES, tilt_deg, APPLIED):
    print(f"  {name}: {t:.3f} deg  (applied={a})")
print()
print("=== Figure 1b: origin correction (mm) ===")
for j, d in origin_rows:
    print(f"  {j}: dx={d[0]:.3f} dy={d[1]:.3f} dz={d[2]:.3f}  |mag|={np.linalg.norm(d):.3f}")
print()
print("=== Figure 2: blocked CV ===")
for s, i, b in zip(stages, in_sample, blocked_cv):
    print(f"  {s.replace(chr(10), ' ')}: in-sample={i}mm, blocked-CV={b}mm")
print()
print("=== Figure 3: repeatability (point 1, X-Y-Z mm) ===")
for i, (x, y, z, tid) in enumerate(pts):
    label = "1st visit" if i == 0 else f"repeat {i}"
    print(f"  {label} (test {tid}): ({x:.3f}, {y:.3f}, {z:.3f})")
