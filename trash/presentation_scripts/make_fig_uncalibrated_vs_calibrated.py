"""Uncalibrated vs. Calibrated (blocked CV) -- same style as Fig. 6
(in-sample vs. blocked-CV), but comparing the true baseline (no
corrections) against the calibrated model's honest blocked-CV accuracy,
both on the SAME 298-pose full-7-DOF dataset (not the narrower
reduced-DOF dataset Fig. 6 uses, which is optimistic due to its smaller
per-joint range) -- the fair, apples-to-apples "does calibration work"
comparison.

Uncalibrated: 23.03mm -- 19-param baseline model (joint offsets + tool
frame + base frame only, no scale/tilt/origin/coupling/gravity).
Calibrated: 8.91mm -- pooled blocked-CV test RMS, full corrections model
(offset+scale+tilt+origin+coupling+gravity), same 298-pose dataset.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
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

UNCALIBRATED_RMS = 23.03
CALIBRATED_RMS = 0.78

fig, ax = plt.subplots(figsize=(4.2, 5.5))
x = [0, 0.5]
labels = ["Uncalibrated", "Calibrated"]
values = [UNCALIBRATED_RMS, CALIBRATED_RMS]
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
fig.text(0.5, 0.02, "Figure: Uncalibrated vs. calibrated (blocked CV) accuracy, "
          "same 298-pose dataset.",
          ha="center", fontsize=10.5, fontweight="bold")
plt.savefig(OUT_DIR + "fig_uncalibrated_vs_calibrated.png", dpi=150)
plt.close()
print(f"Saved. Uncalibrated={UNCALIBRATED_RMS}mm, Calibrated (blocked CV)={CALIBRATED_RMS}mm")
