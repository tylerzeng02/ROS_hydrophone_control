"""Fits the full 60-param model (the deployed 'optimal' model, <1mm blocked-CV)
fresh on the 374-pose elbow-yaw-locked dataset, in ONE internally-consistent
run, then reports:
  1. Every fitted correction (offset, scale, tilt, origin, Fourier, coupling,
     gravity) -- for the "final table of all corrections" request.
  2. Blocked 8-fold CV, per-fold train/test RMS -- no ablation, just the
     final model's own fold-by-fold numbers.

Saved as JSON + CSV so the figure-generation script can consume it without
re-running the (slow) fit.
"""

import csv
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

FIT_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_fixed_elbow_yaw.csv"
OUT_DIR = "C:/Users/ConformalUser/Desktop/cyton_setup/build/presentation_figures/"

JOINT_NAMES = ["shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow_pitch",
               "elbow_yaw", "wrist_pitch", "wrist_roll"]

SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6
COUPLE_TERMS = [(SHOULDER_ROLL_IDX, SHOULDER_YAW_IDX, SHOULDER_YAW_IDX),
                (SHOULDER_YAW_IDX, ELBOW_YAW_IDX, ELBOW_YAW_IDX),
                (SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX, ELBOW_PITCH_IDX)]
COUPLE_LABELS = ["shoulder_roll*shoulder_yaw -> shoulder_yaw",
                  "shoulder_yaw*elbow_yaw -> elbow_yaw",
                  "shoulder_pitch*elbow_pitch -> elbow_pitch"]
N_COUPLE = len(COUPLE_TERMS)
N_GRAVITY = 7
N_TILT, N_SCALE = 14, 7

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


def tilt_angle_deg(tilt_all, i):
    axes = tilted_axes(tilt_all)
    nominal = ck.JOINT_AXES[i] / np.linalg.norm(ck.JOINT_AXES[i])
    corrected = axes[i]
    return np.degrees(np.arccos(np.clip(np.dot(nominal, corrected), -1, 1)))


def fk_combined(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes = tilted_axes(tilt_all); T = np.eye(4)
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX: origin = origin + o_wp
        if i == ELBOW_PITCH_IDX: origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX: origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        R = Rotation.from_rotvec(axes[i] * ja[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, origin)
    return T


def fk_frames(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes = tilted_axes(tilt_all); T = np.eye(4); pl = []; al = []
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX: origin = origin + o_wp
        if i == ELBOW_PITCH_IDX: origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX: origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
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


TW, TS = 5.0, np.radians(8.0)
SW = 20.0
FW, FS = 5.0, np.radians(5.0)
CW, CS, CB = 5.0, 0.1, 0.3
GW, GS, GB = 5.0, 0.1, 0.5


def make_res(angles, pos, quat):
    n = len(angles)

    def res(x):
        bp, js, tl, oe, osy, owp, fab, cc, gc = unpack(x)
        params = ck.unpack_params(bp)
        pr = np.zeros((n, 3)); orr = np.zeros((n, 3))
        for i in range(n):
            Tp = predict(angles[i], bp, js, tl, oe, osy, owp, fab, cc, gc)
            pr[i] = pos[i] - Tp[:3, 3] * 1000.0
            Rp = Tp[:3, :3]; Rm = Rotation.from_quat(quat[i]).as_matrix()
            orr[i] = Rotation.from_matrix(Rp.T @ Rm).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)), params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)), tl.ravel() * (TW / TS), (js - 1.0) * SW,
            fab * (FW / FS), cc * (CW / CS), gc * (GW / GS),
        ])
        return np.concatenate([pr.ravel(), orr.ravel(), reg])
    return res


def bounds():
    lo = np.concatenate([
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf] * 3, [-np.inf] * 3, [-np.inf] * 3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT), -0.05 * np.ones(3),
        -0.05 * np.ones(2), -0.05 * np.ones(3), -np.radians(15.0) * np.ones(2), -CB * np.ones(N_COUPLE),
        -GB * np.ones(N_GRAVITY),
    ])
    up = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf] * 3, [np.inf] * 3, [np.inf] * 3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT), 0.05 * np.ones(3),
        0.05 * np.ones(2), 0.05 * np.ones(3), np.radians(15.0) * np.ones(2), CB * np.ones(N_COUPLE),
        GB * np.ones(N_GRAVITY),
    ])
    return lo, up


def x0(angles, pos, quat):
    bx, br = ck.initial_base_guess(angles[0], pos[0], quat[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), bx, br), np.ones(N_SCALE),
        np.zeros(N_TILT + 3 + 2 + 3 + 2 + N_COUPLE + N_GRAVITY),
    ])


def fit(angles, pos, quat):
    lo, up = bounds()
    return least_squares(make_res(angles, pos, quat), x0(angles, pos, quat), method="trf",
                          x_scale="jac", bounds=(lo, up), max_nfev=8000)


def rms(angles, pos, bp, js, tl, oe, osy, owp, fab, cc, gc):
    e = np.zeros(len(angles))
    for i in range(len(angles)):
        Tp = predict(angles[i], bp, js, tl, oe, osy, owp, fab, cc, gc)
        e[i] = np.linalg.norm(pos[i] - Tp[:3, 3] * 1000.0)
    return np.sqrt(np.mean(e ** 2))


def main():
    print(f"Fitting full {TOTAL}-param model on {FIT_CSV}...")
    angles_all, pos_mm_all, quat_all = ck.load_poses_from_csv(FIT_CSV)
    n_all = len(angles_all)
    print(f"Loaded {n_all} poses")

    result = fit(angles_all, pos_mm_all, quat_all)
    bp, js, tl, oe, osy, owp, fab, cc, gc = unpack(result.x)
    params = ck.unpack_params(bp)
    full_rms = rms(angles_all, pos_mm_all, bp, js, tl, oe, osy, owp, fab, cc, gc)
    print(f"Full-dataset in-sample RMS: {full_rms:.3f}mm (sanity check vs. documented 0.66mm)")

    # ---- Blocked 8-fold CV, final model only, no ablation ----
    N_FOLDS = 8
    fold_bounds = np.linspace(0, n_all, N_FOLDS + 1).astype(int)
    fold_results = []
    print("\n=== Blocked 8-fold CV (final model) ===")
    for f in range(N_FOLDS):
        lo_i, hi_i = fold_bounds[f], fold_bounds[f + 1]
        test_mask = np.zeros(n_all, dtype=bool)
        test_mask[lo_i:hi_i] = True
        train_mask = ~test_mask
        r = fit(angles_all[train_mask], pos_mm_all[train_mask], quat_all[train_mask])
        bp_f, js_f, tl_f, oe_f, osy_f, owp_f, fab_f, cc_f, gc_f = unpack(r.x)
        train_rms = rms(angles_all[train_mask], pos_mm_all[train_mask], bp_f, js_f, tl_f, oe_f, osy_f, owp_f, fab_f, cc_f, gc_f)
        test_rms = rms(angles_all[test_mask], pos_mm_all[test_mask], bp_f, js_f, tl_f, oe_f, osy_f, owp_f, fab_f, cc_f, gc_f)
        n_test = test_mask.sum()
        fold_results.append({"fold": f, "idx_lo": int(lo_i), "idx_hi": int(hi_i),
                              "n_test": int(n_test), "train_rms": float(train_rms), "test_rms": float(test_rms)})
        print(f"  fold {f} (idx {lo_i}-{hi_i}, n_test={n_test}): train={train_rms:.3f}mm test={test_rms:.3f}mm")

    pooled_sq = sum((fr["test_rms"] ** 2) * fr["n_test"] for fr in fold_results)
    pooled_n = sum(fr["n_test"] for fr in fold_results)
    pooled_rms = np.sqrt(pooled_sq / pooled_n)
    print(f"\nPooled blocked-CV test RMS: {pooled_rms:.3f}mm")

    # ---- Save fold results ----
    with open(OUT_DIR + "final_model_blocked_cv_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "pose_idx_range", "n_test", "train_rms_mm", "test_rms_mm"])
        for fr in fold_results:
            w.writerow([fr["fold"], f"{fr['idx_lo']}-{fr['idx_hi']}", fr["n_test"],
                        f"{fr['train_rms']:.3f}", f"{fr['test_rms']:.3f}"])
        w.writerow(["POOLED", "", pooled_n, "", f"{pooled_rms:.3f}"])

    # ---- Save full correction table ----
    tilts_deg = [tilt_angle_deg(tl, i) for i in range(7)]
    origin_corr = {
        "shoulder_yaw": (osy[0] * 1000.0, 0.0, osy[1] * 1000.0),
        "elbow_pitch": tuple(oe * 1000.0),
        "wrist_pitch": tuple(owp * 1000.0),
    }

    with open(OUT_DIR + "final_model_corrections_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "offset_deg", "scale", "axis_tilt_deg", "origin_dx_mm", "origin_dy_mm",
                    "origin_dz_mm", "gravity_coeff_rad_per_m"])
        for i, name in enumerate(JOINT_NAMES):
            ox, oy, oz = origin_corr.get(name, ("", "", ""))
            ox = f"{ox:.3f}" if ox != "" else ""
            oy = f"{oy:.3f}" if oy != "" else ""
            oz = f"{oz:.3f}" if oz != "" else ""
            w.writerow([name, f"{np.degrees(params.joint_offsets[i]):+.4f}", f"{js[i]:.6f}",
                        f"{tilts_deg[i]:.3f}", ox, oy, oz, f"{gc[i]:.6f}"])

    with open(OUT_DIR + "final_model_pose_dependent_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "value"])
        w.writerow(["shoulder_pitch Fourier a (sin coeff, rad)", f"{fab[0]:.6f}"])
        w.writerow(["shoulder_pitch Fourier b (cos coeff, rad)", f"{fab[1]:.6f}"])
        for label, coeff in zip(COUPLE_LABELS, cc):
            w.writerow([f"coupling: {label} (rad^-1)", f"{coeff:.6f}"])

    # ---- Save summary JSON for the figure script ----
    summary = {
        "total_params": TOTAL,
        "n_poses": n_all,
        "full_dataset_rms_mm": float(full_rms),
        "pooled_blocked_cv_rms_mm": float(pooled_rms),
        "folds": fold_results,
    }
    with open(OUT_DIR + "final_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll data saved to", OUT_DIR)


if __name__ == "__main__":
    main()
