"""Figure 11: a 2D schematic of the Cyton Gamma 1500's kinematic chain,
generated directly from references/cyton_gamma_1500_trac_ik.urdf (the
CORRECTED origins/axes actually deployed, not nominal ones -- copied
verbatim from that file below), with each joint's fitted correction types
highlighted. Which corrections apply to which joint comes straight from
that URDF's own header comment table (offset/scale/tilt/origin, or "tilt
only" / "NONE" for the two degenerate/locked joints).

Joint positions are real forward kinematics (translate-by-origin, then
rotate-by-axis) evaluated at a chosen non-zero "reaching" pose -- not the
all-zero home pose, which happens to leave this arm nearly straight and
uninformative to look at -- projected onto the Y-Z plane (arm's forward
reach vs. height; the small X offsets are dropped for a clean 2D side
view). Styled to match the aquarel 'arctic_light' + Roboto look used
across this figure set.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme
from scipy.spatial.transform import Rotation
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

# ---------------------------------------------------------------------------
# Straight from references/cyton_gamma_1500_trac_ik.urdf -- the CORRECTED
# per-joint <origin xyz> and <axis xyz> actually deployed (see that file's
# top-of-file comment table for which correction produced each change).
# ---------------------------------------------------------------------------
JOINTS = [
    # name,           origin_xyz_m,                          axis_xyz,                              corrections,                     scale
    ("shoulder_roll",  [0, 0, 0.05315],                       [0.013792, 0.014877, 0.999794],        ["tilt"],                        0.988203),
    ("shoulder_pitch", [0.0205, 0, 0.12435],                  [0.998997, -0.043573, -0.010357],      ["offset", "scale", "tilt"],     1.001931),
    ("shoulder_yaw",   [-0.02478414, -0.0205, 0.1308452],     [-0.027678, -0.999437, 0.018973],      ["offset", "scale", "tilt", "origin"], 0.964711),
    ("elbow_pitch",    [0.01656849, 0.02722018, 0.11356304],  [0.999677, -0.024005, 0.008304],       ["offset", "scale", "tilt", "origin"], 1.014467),
    ("elbow_yaw",      [-0.0171, -0.018, 0.09746],            [0, -1, 0],                             ["locked"],                      None),
    ("wrist_pitch",    [0.02765348, 0.01273746, 0.07244612],  [0.999044, 0.042957, 0.008105],        ["offset", "scale", "tilt", "origin"], 1.006796),
    ("wrist_roll",     [-0.026255, 0, 0.051425],              [-0.006451, -0.018572, 0.999807],      ["tilt"],                        1.002933),
]
TOOL_ORIGIN_M = [-0.002316, 0.0079, 0.079425]  # virtual_endeffector_joint

# A non-zero "reaching forward and down" demo pose (degrees) -- shown purely
# to make the schematic legible; NOT a fitted or calibration-relevant value.
DEMO_POSE_DEG = {
    "shoulder_roll": 0, "shoulder_pitch": -55, "shoulder_yaw": 10,
    "elbow_pitch": 70, "elbow_yaw": 0, "wrist_pitch": -20, "wrist_roll": 0,
}

# ---------------------------------------------------------------------------
# forward kinematics -> real 3D joint positions at the demo pose
# ---------------------------------------------------------------------------
T = np.eye(4)
positions_m = [("base", T[:3, 3].copy())]
for name, origin, axis, _corr, _scale in JOINTS:
    origin = np.array(origin, dtype=float)
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    T_origin = np.eye(4)
    T_origin[:3, 3] = origin
    T = T @ T_origin
    positions_m.append((name, T[:3, 3].copy()))
    theta = np.radians(DEMO_POSE_DEG[name])
    R = Rotation.from_rotvec(axis * theta).as_matrix()
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    T = T @ T_rot
T_tool = np.eye(4)
T_tool[:3, 3] = TOOL_ORIGIN_M
T = T @ T_tool
positions_m.append(("tool", T[:3, 3].copy()))

# project to the Y-Z plane (forward reach vs. height), in mm
chain_yz = {name: (p[1] * 1000.0, p[2] * 1000.0) for name, p in positions_m}

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
COLORS = {
    "offset": "#a8690a",
    "scale": "#2388aa",
    "tilt": "#7259b8",
    "origin": "#2f8f5b",
    "locked": "#8a97a1",
    "frame": "#c62f2f",
}
LABELS = {
    "offset": "Zero-offset", "scale": "Scale", "tilt": "Axis tilt",
    "origin": "Origin Δ", "locked": "Locked (no fit)", "frame": "Frame transform (fit)",
}

fig, ax = plt.subplots(figsize=(7.5, 9))

chain_order = ["base", "shoulder_roll", "shoulder_pitch", "shoulder_yaw",
                "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll", "tool"]
xs = [chain_yz[n][0] for n in chain_order]
zs = [chain_yz[n][1] for n in chain_order]
ax.plot(xs, zs, color="#98a4ad", linewidth=9, solid_capstyle="round",
        solid_joinstyle="round", zorder=1, alpha=0.55)

badge_dx, badge_w, badge_gap = 16, 9, 3


def draw_badges(x, y, corrections):
    if corrections == ["locked"]:
        ax.text(x + badge_dx, y, "LOCKED", fontsize=8.5, fontweight="bold",
                 color=COLORS["locked"], va="center", ha="left", zorder=4)
        return
    for i, c in enumerate(corrections):
        bx = x + badge_dx + i * (badge_w + badge_gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, y - badge_w / 2), badge_w, badge_w,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=0, facecolor=COLORS[c], zorder=4))


for name, origin, axis, corrections, scale in JOINTS:
    x, y = chain_yz[name]
    ax.add_patch(mpatches.Circle((x, y), 6.3, facecolor="white",
                                  edgecolor="#8a97a1" if corrections == ["locked"] else "#1e2833",
                                  linewidth=2,
                                  linestyle=(0, (2, 2)) if corrections == ["locked"] else "solid",
                                  zorder=5))
    draw_badges(x, y, corrections)
    label = name.replace("_", " ")
    ax.text(x - 14, y + 1, label, fontsize=10.5, fontweight="bold",
             color="#1e2833", va="center", ha="right", zorder=4)
    if scale is not None:
        ax.text(x - 14, y - 13, f"scale {scale:.3f}", fontsize=8.5,
                 style="italic", color="#5c6b78", va="center", ha="right", zorder=4)

# base and tool: frame transforms, not per-joint corrections
for name, dx, label, sub in [
    ("base", 1, "Base Frame", "tracker -> base_link (fit)"),
    ("tool", 1, "Tool Frame", "wrist -> virtual_endeffector (fit)"),
]:
    x, y = chain_yz[name]
    ax.add_patch(mpatches.Circle((x, y), 6.3, facecolor=COLORS["frame"],
                                  edgecolor="#1e2833", linewidth=2, zorder=5))
    ax.text(x + 14, y + 5, label, fontsize=10.5, fontweight="bold",
             color="#1e2833", va="center", ha="left", zorder=4)
    ax.text(x + 14, y - 8, sub, fontsize=8.5, style="italic",
             color="#5c6b78", va="center", ha="left", zorder=4)

ax.set_xlim(-90, 290)
ax.set_ylim(-30, 660)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Cyton Gamma 1500 — Calibration Parameter Map", fontweight="bold", fontsize=15, pad=10)

legend_handles = [mpatches.Patch(color=COLORS[k], label=LABELS[k])
                   for k in ["offset", "scale", "tilt", "origin", "locked", "frame"]]
legend = ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0),
                    frameon=True, fontsize=9.5, title="Fitted correction", title_fontsize=10,
                    ncol=1, handlelength=1.2, handleheight=1.2)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(0.92)
legend.get_frame().set_edgecolor("#333333")

plt.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.5, 0.015,
          "Figure 11: which of the 48 fitted correction parameters lives at each joint, per "
          "references/cyton_gamma_1500_trac_ik.urdf. Pose shown is illustrative, not a fitted value.",
          ha="center", fontsize=9.5, fontweight="bold", wrap=True)
plt.savefig(OUT_DIR + "fig11_robot_schematic.png", dpi=150)
plt.close()
theme.apply_transforms()
print("Figure 11 saved.")