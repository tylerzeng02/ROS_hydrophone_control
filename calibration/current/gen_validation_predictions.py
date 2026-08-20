"""Generates predicted end-effector positions for a list of joint-tick test
points, using the current best-fit 60-param correction model (recovered in
recovered_60param_fit.py -- see CLAUDE.md's kinematic-calibration section for
where that model came from).

Companion to calibration/collection/ndi_capture_and_validate.cpp's --validate mode: that
program moves the real arm to each test point and measures the actual NDI
position; this script supplies the "predicted" side of the comparison. The
C++ program has no FK/calibration model of its own on purpose -- keeping the
model in one place (Python, where every other fit in this project lives)
avoids a second, easily-drifting reimplementation in C++.

Usage:
    python gen_validation_predictions.py
    python gen_validation_predictions.py --fit-csv a.csv,b.csv --points ticks.csv --out predictions.csv

Input points CSV format (no header requirements beyond 7 columns):
    tick_0,tick_1,tick_2,tick_3,tick_4,tick_5,tick_6

Output predictions CSV format (consumed directly by --validate):
    tick_0,tick_1,tick_2,tick_3,tick_4,tick_5,tick_6,predicted_x_mm,predicted_y_mm,predicted_z_mm

IMPORTANT: for a meaningful validation, the tick targets in the points CSV
should be genuinely new configurations, not ones already present in
--fit-csv -- reusing fit poses here would just re-measure training fit
quality, not real generalization to unseen configurations.

Requires: numpy, scipy
"""

import argparse
import csv
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6
COUPLE_TERMS = [
    (SHOULDER_ROLL_IDX, SHOULDER_YAW_IDX, SHOULDER_YAW_IDX),
    (SHOULDER_YAW_IDX, ELBOW_YAW_IDX, ELBOW_YAW_IDX),
    (SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX, ELBOW_PITCH_IDX),
]
N_COUPLE = len(COUPLE_TERMS)
N_GRAVITY = 7
N_TILT, N_SCALE = 14, 7

PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    PERP.append((u, v))


def tilted_axes(tilt_all):
    axes = []
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        u, v = PERP[i]
        p = a + tilt_all[i, 0] * u + tilt_all[i, 1] * v
        axes.append(p / np.linalg.norm(p))
    return axes


def fk_combined(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes = tilted_axes(tilt_all)
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX:
            origin = origin + o_wp
        if i == ELBOW_PITCH_IDX:
            origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        R = Rotation.from_rotvec(axes[i] * ja[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, origin)
    return T


def fk_frames(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes = tilted_axes(tilt_all)
    T = np.eye(4)
    pl = []
    al = []
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX:
            origin = origin + o_wp
        if i == ELBOW_PITCH_IDX:
            origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX:
            origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        p = T[:3, :3] @ origin + T[:3, 3]
        a = T[:3, :3] @ axes[i]
        pl.append(p)
        al.append(a)
        R = Rotation.from_rotvec(axes[i] * ja[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, origin)
    return T, pl, al


GDIR = np.array([0.0, 0.0, -1.0])


def gterms(ca, tilt_all, o_elbow, o_sy_xz, o_wp):
    T, pl, al = fk_frames(ca, tilt_all, o_elbow, o_sy_xz, o_wp)
    pee = T[:3, 3]
    g = np.zeros(7)
    for i in range(7):
        lever = pee - pl[i]
        g[i] = np.dot(np.cross(lever, GDIR), al[i])
    return g


OFF_SCALE = 19
OFF_TILT = OFF_SCALE + N_SCALE
OFF_OE = OFF_TILT + N_TILT
OFF_OSY = OFF_OE + 3
OFF_OWP = OFF_OSY + 2
OFF_F = OFF_OWP + 3
OFF_C = OFF_F + 2
OFF_G = OFF_C + N_COUPLE
TOTAL = OFF_G + N_GRAVITY


def unpack(x):
    bp = x[0:19]
    js = x[OFF_SCALE:OFF_SCALE + N_SCALE]
    tl = x[OFF_TILT:OFF_TILT + N_TILT].reshape(7, 2)
    oe = x[OFF_OE:OFF_OE + 3]
    osy = x[OFF_OSY:OFF_OSY + 2]
    owp = x[OFF_OWP:OFF_OWP + 3]
    fab = x[OFF_F:OFF_F + 2]
    cc = x[OFF_C:OFF_C + N_COUPLE]
    gc = x[OFF_G:OFF_G + N_GRAVITY]
    return bp, js, tl, oe, osy, owp, fab, cc, gc


def predict(ar, bp, js, tl, oe, osy, owp, fab, cc, gc):
    params = ck.unpack_params(bp)
    ca = (ar * js + params.joint_offsets).copy()
    rsp = ar[SHOULDER_PITCH_IDX]
    ca[SHOULDER_PITCH_IDX] += fab[0] * np.sin(rsp) + fab[1] * np.cos(rsp)
    for k, (i, j, t) in enumerate(COUPLE_TERMS):
        ca[t] += cc[k] * ar[i] * ar[j]
    g = gterms(ca, tl, oe, osy, owp)
    cag = ca + gc * g
    Tfk = fk_combined(cag, tl, oe, osy, owp)
    Ttool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    Tbase = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return Tbase @ Tfk @ Ttool


def make_res(angles, pos, quat):
    n = len(angles)
    TW, TS = 5.0, np.radians(8.0)
    SW = 20.0
    FW, FS = 5.0, np.radians(5.0)
    CW, CS = 5.0, 0.1
    GW, GS = 5.0, 0.1

    def res(x):
        bp, js, tl, oe, osy, owp, fab, cc, gc = unpack(x)
        params = ck.unpack_params(bp)
        pr = np.zeros((n, 3))
        orr = np.zeros((n, 3))
        for i in range(n):
            Tp = predict(angles[i], bp, js, tl, oe, osy, owp, fab, cc, gc)
            pr[i] = pos[i] - Tp[:3, 3] * 1000.0
            Rp = Tp[:3, :3]
            Rm = Rotation.from_quat(quat[i]).as_matrix()
            orr[i] = Rotation.from_matrix(Rp.T @ Rm).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)),
            params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)),
            tl.ravel() * (TW / TS),
            (js - 1.0) * SW,
            fab * (FW / FS),
            cc * (CW / CS),
            gc * (GW / GS),
        ])
        return np.concatenate([pr.ravel(), orr.ravel(), reg])

    return res


def bounds():
    CB, GB = 0.3, 0.5
    lo = np.concatenate([
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3),
        [-np.inf] * 3, [-np.inf] * 3, [-np.inf] * 3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT),
        -0.05 * np.ones(3), -0.05 * np.ones(2), -0.05 * np.ones(3),
        -np.radians(15.0) * np.ones(2), -CB * np.ones(N_COUPLE), -GB * np.ones(N_GRAVITY),
    ])
    up = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3),
        [np.inf] * 3, [np.inf] * 3, [np.inf] * 3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
        np.radians(15.0) * np.ones(2), CB * np.ones(N_COUPLE), GB * np.ones(N_GRAVITY),
    ])
    return lo, up


def x0(angles, pos, quat):
    bx, br = ck.initial_base_guess(angles[0], pos[0], quat[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), bx, br),
        np.ones(N_SCALE),
        np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE + N_GRAVITY),
    ])


def fit(angles, pos, quat):
    lo, up = bounds()
    return least_squares(
        make_res(angles, pos, quat), x0(angles, pos, quat),
        method="trf", x_scale="jac", bounds=(lo, up), max_nfev=5000,
    )


def ticks_to_radians(ticks):
    return (np.array(ticks, dtype=float) - np.array(ck.NOMINAL_ZERO_TICKS)) / ck.TICKS_PER_RADIAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-csv",
        default=(
            "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv,"
            "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_batch3.csv"
        ),
        help="Comma-separated list of CSVs to fit the correction model on.",
    )
    parser.add_argument(
        "--points",
        default="C:/Users/ConformalUser/Desktop/cyton_setup/build/validation_ticks.csv",
        help="Input CSV of tick_0..tick_6 test points (no predictions yet).",
    )
    parser.add_argument(
        "--out",
        default="C:/Users/ConformalUser/Desktop/cyton_setup/build/validation_points.csv",
        help="Output CSV with predicted_x/y/z_mm columns added, for --validate.",
    )
    args = parser.parse_args()

    fit_paths = [p.strip() for p in args.fit_csv.split(",") if p.strip()]
    angles_list, pos_list, quat_list = [], [], []
    for path in fit_paths:
        a, p, q = ck.load_poses_from_csv(path)
        angles_list.append(a)
        pos_list.append(p)
        quat_list.append(q)
        print(f"Loaded {len(a)} poses from {path}")
    angles_all = np.concatenate(angles_list)
    pos_mm_all = np.concatenate(pos_list)
    quat_xyzw_all = np.concatenate(quat_list)
    print(f"Fitting {TOTAL}-param model on {len(angles_all)} combined poses...")

    result = fit(angles_all, pos_mm_all, quat_xyzw_all)
    bp, js, tl, oe, osy, owp, fab, cc, gc = unpack(result.x)

    e = np.zeros(len(angles_all))
    for i in range(len(angles_all)):
        Tp = predict(angles_all[i], bp, js, tl, oe, osy, owp, fab, cc, gc)
        e[i] = np.linalg.norm(pos_mm_all[i] - Tp[:3, 3] * 1000.0)
    print(f"Fit in-sample RMS on combined dataset: {np.sqrt(np.mean(e ** 2)):.2f}mm "
          f"(sanity check only -- not a held-out number)")

    ticks_rows = []
    with open(args.points, newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        try:
            [int(v) for v in first[:7]]
            ticks_rows.append(first)
        except ValueError:
            pass  # header row, skip
        for row in reader:
            if not row:
                continue
            ticks_rows.append(row)

    if not ticks_rows:
        print(f"No test points found in {args.points}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tick_0", "tick_1", "tick_2", "tick_3", "tick_4", "tick_5", "tick_6",
            "predicted_x_mm", "predicted_y_mm", "predicted_z_mm",
        ])
        for row in ticks_rows:
            ticks = [int(v) for v in row[:7]]
            angles = ticks_to_radians(ticks)
            Tp = predict(angles, bp, js, tl, oe, osy, owp, fab, cc, gc)
            pred_mm = Tp[:3, 3] * 1000.0
            writer.writerow(ticks + [f"{pred_mm[0]:.4f}", f"{pred_mm[1]:.4f}", f"{pred_mm[2]:.4f}"])
            print(f"  ticks={ticks} -> predicted (x,y,z)mm = "
                  f"({pred_mm[0]:.2f}, {pred_mm[1]:.2f}, {pred_mm[2]:.2f})")

    print(f"\nWrote {len(ticks_rows)} predicted test points to {args.out}")
    print(f"Run on the arm with: ndi_capture_and_validate.exe --validate {args.out}")


if __name__ == "__main__":
    main()
