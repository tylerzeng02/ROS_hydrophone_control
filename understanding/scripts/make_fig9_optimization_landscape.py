"""Figure 9: the actual nonlinear least-squares optimization landscape used
by the calibration pipeline (scipy.optimize.least_squares, method='trf',
the reduced 48-param kinematic model from
calibration/current/reduced_model_blocked_cv.py) on the real 374-pose
dataset. Not a stylized illustration -- this fits the real model to the
real data, then evaluates the REAL cost function (position RMS, mm) on a
grid around the fitted optimum along two of its 48 dimensions (the
shoulder_roll and shoulder_pitch joint-scale terms), so the bowl shown is
the actual local shape the trust-region-reflective solver descended.
Rendered as a 3D surface, styled to match the aquarel 'arctic_light' +
Roboto look used across this figure set.
"""

import os
import sys

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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "build", "presentation_figures") + "/"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(REPO_ROOT, "calibration", "current"))
import calibrate_kinematics as ck          # noqa: E402
import reduced_model_blocked_cv as rm      # noqa: E402

theme = load_theme("arctic_light")
theme.apply()
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["xtick.labelsize"] = 10.5
plt.rcParams["ytick.labelsize"] = 10.5
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"

DATA_CSV = os.path.join(REPO_ROOT, "understanding", "data",
                         "quick_calibration_test_fixed_elbow_yaw.csv")
FIT_CACHE = os.path.join(OUT_DIR, "fig9_fit_cache.npy")

angles, pos_mm, quat = ck.load_poses_from_csv(DATA_CSV)

if os.path.exists(FIT_CACHE):
    print(f"Reusing cached fit from {FIT_CACHE} (delete it to refit).")
    x_fit = np.load(FIT_CACHE)
else:
    print("Loading real capture data and fitting the real 48-param model "
          "(scipy least_squares, method='trf') -- this takes ~2 minutes...")
    result = rm.fit(angles, pos_mm, quat)
    x_fit = result.x
    np.save(FIT_CACHE, x_fit)


class _Result:
    pass


result = _Result()
result.x = x_fit
bp0, js0, tl0, oe0, osy0, owp0 = rm.unpack(result.x)
fitted_rms = rm.rms(angles, pos_mm, bp0, js0, tl0, oe0, osy0, owp0)
print(f"Fitted in-sample RMS: {fitted_rms:.3f} mm ({len(angles)} poses)")

# two of the 48 fitted dimensions to slice: the shoulder_roll and
# shoulder_pitch joint-SCALE terms. Chosen because (a) scale terms have a
# natural, physically bounded range (0.90-1.10, the solver's own bounds)
# to grid over, and (b) unlike several other dimensions in this fit, the
# solver actually left them sitting in the bounds' interior rather than
# pinned at an edge -- so the grid captures a genuine local minimum.
IDX_A, IDX_B = rm.OFF_SCALE + 0, rm.OFF_SCALE + 1   # shoulder_roll, shoulder_pitch scale
LABEL_A, LABEL_B = "shoulder_roll scale", "shoulder_pitch scale"
center_a, center_b = result.x[IDX_A], result.x[IDX_B]

N_GRID = 31
grid_a = np.linspace(0.90, 1.10, N_GRID)
grid_b = np.linspace(0.90, 1.10, N_GRID)


def cost_at(a, b):
    x = result.x.copy()
    x[IDX_A], x[IDX_B] = a, b
    bp, js, tl, oe, osy, owp = rm.unpack(x)
    return rm.rms(angles, pos_mm, bp, js, tl, oe, osy, owp)


print(f"Evaluating {N_GRID}x{N_GRID} real cost-function grid around the fitted optimum...")
Z = np.zeros((N_GRID, N_GRID))
for i, a in enumerate(grid_a):
    for j, b in enumerate(grid_b):
        Z[j, i] = cost_at(a, b)
A, B = np.meshgrid(grid_a, grid_b)

fig = plt.figure(figsize=(9, 7.5))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

PANE_COLOR = (1.0, 1.0, 1.0, 1.0)  # white, per this session's request
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.set_pane_color(PANE_COLOR)
    pane._axinfo["grid"]["color"] = (0.75, 0.77, 0.8, 0.6)

surf = ax.plot_surface(A, B, Z, cmap="Blues_r", linewidth=0, antialiased=True,
                        rstride=1, cstride=1, alpha=0.95, zorder=1)
ax.contour(A, B, Z, zdir="z", offset=Z.min() - (Z.max() - Z.min()) * 0.12,
           cmap="Blues_r", linewidths=1.0, levels=10)

fitted_cost = cost_at(center_a, center_b)
ax.scatter([center_a], [center_b], [fitted_cost], color="#D32F2F", s=90,
           edgecolors="black", linewidths=1.0, depthshade=False, zorder=5,
           label="Fitted optimum")

ax.set_xlabel(f"\n{LABEL_A}", fontweight="bold", fontsize=12)
ax.set_ylabel(f"\n{LABEL_B}", fontweight="bold", fontsize=12)
ax.set_zlabel("\nPosition RMS (mm)", fontweight="bold", fontsize=12)
ax.set_zlim(Z.min() - (Z.max() - Z.min()) * 0.12, Z.max())
ax.view_init(elev=27, azim=-125)

legend = ax.legend(loc="upper left", frameon=True, fontsize=11)
legend.get_frame().set_facecolor(PANE_COLOR)
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_edgecolor("#333333")

cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
cbar.set_label("Position RMS (mm)", fontweight="bold", fontsize=10.5)

plt.tight_layout(rect=(0, 0.05, 1, 1))
fig.text(0.5, 0.015,
          f"Figure 9: Real cost-function landscape around the fitted calibration optimum "
          f"({LABEL_A} vs. {LABEL_B}; fitted RMS {fitted_cost:.2f} mm).",
          ha="center", fontsize=11, fontweight="bold")
plt.savefig(OUT_DIR + "fig9_optimization_landscape.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
theme.apply_transforms()
print(f"Figure 9 saved. Grid cost range: {Z.min():.3f}-{Z.max():.3f} mm, "
      f"fitted point: ({center_a:.4f}, {center_b:.4f}) -> {fitted_cost:.3f} mm")