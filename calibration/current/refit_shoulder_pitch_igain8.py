"""Single-parameter refit: shoulder_pitch's offset, using the 9 points
already collected with I=8 active (build/validation_results_igain8_native.csv).

Context: every other parameter in the deployed 60-param model
(offsets/scales/tilts/origins/tool+base frame/fourier/coupling/gravity)
was fit using data collected under the OLD servo behavior (I=0), where
shoulder_pitch consistently undershot its commanded tick by ~7-9 ticks.
The fitted shoulder_pitch offset silently absorbed/compensated for that
consistent undershoot. Now that I=8 fixes the undershoot at the servo
level (confirmed: mean tick error 8.8 -> 0.6 ticks), the model's stale
compensation is wrong in the same direction it used to correct for,
producing a fresh ~14mm systematic error (confirmed:
build/validation_results_igain0_baseline.csv vs
build/validation_results_igain8_native.csv, same 9 points).

This script holds every other parameter FIXED at its already-fitted value
(hardcoded below, copied from calibration/current/final_deployment_fit.py's
2026-08-13 rerun -- confirmed reproducing the deployed model exactly,
0.66mm in-sample RMS) and fits ONLY an additional shoulder_pitch offset
against the 9 (achieved_tick, actual_position) pairs already collected
with I=8 active -- no new data collection needed.

Run from calibration/current/ (imports calibrate_kinematics as ck, same
requirement as every other script in this family):
    python3 refit_shoulder_pitch_igain8.py
"""

import csv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

SHOULDER_ROLL, SHOULDER_PITCH, SHOULDER_YAW = 0, 1, 2
ELBOW_PITCH, ELBOW_YAW, WRIST_PITCH, WRIST_ROLL = 3, 4, 5, 6
JOINT_NAMES = ['shoulder_roll', 'shoulder_pitch', 'shoulder_yaw', 'elbow_pitch',
               'elbow_yaw', 'wrist_pitch', 'wrist_roll']

# ---------------------------------------------------------------------
# Hardcoded, already-fitted values (2026-08-13 rerun of
# final_deployment_fit.py on build/quick_calibration_test_fixed_elbow_yaw.csv
# -- confirmed matching the deployed model, 0.66mm in-sample RMS). Held
# FIXED here -- only extra_sp_offset below is free.
# ---------------------------------------------------------------------
joint_offsets = np.radians([0.0007, 0.0900, 0.5238, 0.5297, 0.3182, 0.4909, 0.8648])
js = np.array([0.988203, 1.001931, 0.964711, 1.014467, 1.002896, 1.006796, 1.002933])
tl = np.array([
    [0.014880, -0.013795], [-0.010366, 0.043617], [0.018984, 0.027694],
    [0.008307, 0.024013], [-0.000464, -0.006366], [0.008113, -0.042998],
    [-0.018576, 0.006452],
])
oe = np.array([-1.43149464, 6.62017123, -2.2369582]) / 1000.0
osy = np.array([-3.28414373, 5.34520352]) / 1000.0
owp = np.array([1.39345109, -5.26254658, 0.64612426]) / 1000.0
tool_xyz = np.array([26.43914407, 26.20739803, 14.47733691]) / 1000.0
tool_rpy = np.radians([-45.68262033, 1.13374905, 90.85256803])
base_xyz = np.array([-114.93560492, 7.02203115, 41.80225408]) / 1000.0
base_rpy = np.radians([-203.32669092, 45.6842781, -263.50263258])
fab = np.array([-0.002404, 0.012840])
COUPLE_TERMS = [(SHOULDER_ROLL, SHOULDER_YAW, SHOULDER_YAW),
                (SHOULDER_YAW, ELBOW_YAW, ELBOW_YAW),
                (SHOULDER_PITCH, ELBOW_PITCH, ELBOW_PITCH)]
cc = np.array([0.002638, -0.000372, 0.002293])
gc = np.array([0.001381, 0.002937, 0.000380, 0.011437, -0.001249, 0.000190, 0.000000])

PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u); v = np.cross(a, u)
    PERP.append((u, v))

def tilted_axes(tilt_all):
    axes = []
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]; u, v = PERP[i]
        p = a + tilt_all[i, 0] * u + tilt_all[i, 1] * v
        axes.append(p / np.linalg.norm(p))
    return axes

def fk_combined(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes = tilted_axes(tilt_all); T = np.eye(4)
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH: origin = origin + o_wp
        if i == ELBOW_PITCH: origin = origin + o_elbow
        if i == SHOULDER_YAW: origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        R = Rotation.from_rotvec(axes[i] * ja[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, origin)
    return T

def fk_frames(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes = tilted_axes(tilt_all); T = np.eye(4); pl = []; al = []
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH: origin = origin + o_wp
        if i == ELBOW_PITCH: origin = origin + o_elbow
        if i == SHOULDER_YAW: origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        p = T[:3, :3] @ origin + T[:3, 3]; a = T[:3, :3] @ axes[i]
        pl.append(p); al.append(a)
        R = Rotation.from_rotvec(axes[i] * ja[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, origin)
    return T, pl, al

GDIR = np.array([0.0, 0.0, -1.0])
def gterms(ca, tilt_all, o_elbow, o_sy_xz, o_wp):
    T, pl, al = fk_frames(ca, tilt_all, o_elbow, o_sy_xz, o_wp); pee = T[:3, 3]
    g = np.zeros(7)
    for i in range(7):
        lever = pee - pl[i]; g[i] = np.dot(np.cross(lever, GDIR), al[i])
    return g

def predict(ar, extra_sp_offset):
    ca = (ar * js + joint_offsets).copy()
    ca[SHOULDER_PITCH] += extra_sp_offset  # <-- the one new free parameter
    rsp = ar[SHOULDER_PITCH]
    ca[SHOULDER_PITCH] += fab[0] * np.sin(rsp) + fab[1] * np.cos(rsp)
    for k, (i, j, t) in enumerate(COUPLE_TERMS):
        ca[t] += cc[k] * ar[i] * ar[j]
    g = gterms(ca, tl, oe, osy, owp)
    cag = ca + gc * g
    Tfk = fk_combined(cag, tl, oe, osy, owp)
    Ttool = ck.build_tool_transform(tool_xyz, tool_rpy)
    Tbase = ck.build_base_transform(base_xyz, base_rpy)
    return Tbase @ Tfk @ Ttool

# ---------------------------------------------------------------------
# Load the 9 already-collected I=8 points.
# ---------------------------------------------------------------------
DATA_PATH = "../../build/validation_results_igain8_native.csv"
with open(DATA_PATH) as f:
    rows = list(csv.DictReader(f))

ar_list, actual_list = [], []
for r in rows:
    achieved_ticks = np.array([float(r[f"achieved_tick_{i}"]) for i in range(7)])
    ar = (achieved_ticks - np.array(ck.NOMINAL_ZERO_TICKS)) / ck.TICKS_PER_RADIAN
    ar_list.append(ar)
    actual_list.append(np.array([float(r["actual_x_mm"]), float(r["actual_y_mm"]), float(r["actual_z_mm"])]))

print(f"Loaded {len(ar_list)} points from {DATA_PATH}")

def residuals(x):
    extra_sp_offset = x[0]
    res = []
    for ar, actual in zip(ar_list, actual_list):
        Tp = predict(ar, extra_sp_offset)
        res.append(actual - Tp[:3, 3] * 1000.0)
    return np.concatenate(res)

# Before: extra_sp_offset = 0 (i.e. the stale, deployed model as-is)
before = residuals(np.array([0.0]))
before_per_point = before.reshape(-1, 3)
before_dev = np.linalg.norm(before_per_point, axis=1)
print(f"\nBEFORE (deployed model, no extra correction): "
      f"mean={before_dev.mean():.2f}mm rms={np.sqrt((before_dev**2).mean()):.2f}mm max={before_dev.max():.2f}mm")

result = least_squares(residuals, x0=np.array([0.0]), method='trf', x_scale='jac',
                        bounds=([-np.radians(10.0)], [np.radians(10.0)]))
extra_sp_offset = result.x[0]

after = residuals(result.x)
after_per_point = after.reshape(-1, 3)
after_dev = np.linalg.norm(after_per_point, axis=1)
print(f"\nAFTER  (extra_sp_offset={np.degrees(extra_sp_offset):+.4f} deg = "
      f"{extra_sp_offset*ck.TICKS_PER_RADIAN:+.2f} ticks): "
      f"mean={after_dev.mean():.2f}mm rms={np.sqrt((after_dev**2).mean()):.2f}mm max={after_dev.max():.2f}mm")

print(f"\nCurrent robot_calibration.cpp shoulder_pitch (motor 1) zeroTick should be adjusted by "
      f"{-extra_sp_offset*ck.TICKS_PER_RADIAN/js[SHOULDER_PITCH]:+.2f} ticks "
      f"(zeroTick_new = zeroTick_current - offset_rad*TICKS_PER_RADIAN/scale) "
      f"to fold this correction in directly.")

print("\nPer-point deviation (mm):")
for r, b, a in zip(rows, before_dev, after_dev):
    print(f"  test_id {r['test_id']}: before={b:.2f}mm -> after={a:.2f}mm")
