"""Figure 10: the blocked 8-fold CV splitting scheme itself -- which
contiguous slice of the 374-pose, capture-order dataset is held out as test
in each fold, vs. the standard random-shuffle k-fold it's deliberately NOT
using. Complements fig5 (per-fold error) and fig6 (in-sample vs. CV), which
show the CV *results*; this one shows the CV *procedure*. Same fold
boundaries as calibration/current/reduced_model_blocked_cv.py and
final_model_full_report.py (np.linspace over the pose count, 8 folds).
Same aquarel 'arctic_light' + Roboto styling as the other figures in this
set.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme
import font_roboto

FONT_DIR = os.path.join(os.path.dirname(font_roboto.__file__), "files")
for fname in ["Roboto-Regular.ttf", "Roboto-Bold.ttf"]:
    fm.fontManager.addfont(os.path.join(FONT_DIR, fname))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "build", "presentation_figures") + "/"
os.makedirs(OUT_DIR, exist_ok=True)

theme = load_theme("arctic_light")
theme.apply()
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

DATA_CSV = os.path.join(REPO_ROOT, "understanding", "data",
                         "quick_calibration_test_fixed_elbow_yaw.csv")

with open(DATA_CSV, newline="") as f:
    n_all = sum(1 for _ in csv.DictReader(f))

N_FOLDS = 8
fold_bounds = np.linspace(0, n_all, N_FOLDS + 1).astype(int)

TRAIN_COLOR = "#2388aa"
TEST_COLOR = "#D32F2F"

fig, ax = plt.subplots(figsize=(6.2, 6.5))

bar_h = 0.72
for f in range(N_FOLDS):
    lo_i, hi_i = fold_bounds[f], fold_bounds[f + 1]
    y = N_FOLDS - f  # fold 1 at top
    ax.barh(y, n_all, height=bar_h, left=0, color=TRAIN_COLOR, zorder=2)
    ax.barh(y, hi_i - lo_i, height=bar_h, left=lo_i, color=TEST_COLOR, zorder=3)

ax.set_yticks(range(1, N_FOLDS + 1))
ax.set_yticklabels([f"Fold {N_FOLDS - f}" for f in range(N_FOLDS)])
ax.set_xlim(0, n_all)
ax.set_ylim(0.5, N_FOLDS + 0.5)
ax.set_xlabel("Pose index (capture order)", fontweight="bold")

# Fixed, figure-relative margins (not tight_layout+rect) so the legend and
# title -- both placed in FIGURE coordinates below -- can never overflow the
# canvas and force bbox_inches="tight" to silently re-expand/distort it.
fig.subplots_adjust(left=0.16, right=0.97, top=0.80, bottom=0.13)

fig.suptitle("Blocked 8-fold cross-validation split", fontweight="bold", fontsize=14, y=0.97)

handles = [
    mpatches.Patch(color=TRAIN_COLOR, label="Train"),
    mpatches.Patch(color=TEST_COLOR, label="Held-out test"),
]
legend = fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True,
                     bbox_to_anchor=(0.56, 0.92))
legend.get_frame().set_facecolor(ax.get_facecolor())
legend.get_frame().set_alpha(0.85)
legend.get_frame().set_edgecolor("#333333")

fig.text(0.5, 0.02,
          f"Figure 10: Each fold holds out one contiguous block of capture-order poses ({n_all} total) "
          "-- not a random shuffle -- so held-out error reflects genuinely unseen poses.",
          ha="center", fontsize=10, fontweight="bold", wrap=True)
plt.savefig(OUT_DIR + "fig10_blocked_cv_scheme.png", dpi=150)
plt.close()
theme.apply_transforms()
print(f"Figure 10 saved. {n_all} poses, {N_FOLDS} folds, boundaries: {list(fold_bounds)}")