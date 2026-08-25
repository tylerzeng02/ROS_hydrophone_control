"""Validates the deployed 48-param model (offset/scale/tilt/origin, no
coupling/gravity/Fourier) via blocked 8-fold cross-validation: folds are
contiguous pose_id ranges, not a random shuffle. TARGET_POSES is a
sequence of near-identical trajectory steps, so a random split puts each
held-out pose's near-twin in the training set and reports an optimistic
test RMS; blocked folds do not have that leak.
"""

import csv
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

FIT_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/calibration/data/deployed_model_training_dataset_374pose.csv"
OUT_DIR = "C:/Users/ConformalUser/Desktop/cyton_setup/build/presentation_figures/"

WRIST_PITCH_IDX, ELBOW_PITCH_IDX, SHOULDER_YAW_IDX = 5, 3, 2
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


OFF_SCALE = 19
OFF_TILT = OFF_SCALE + N_SCALE
OFF_OE = OFF_TILT + N_TILT
OFF_OSY = OFF_OE + 3
OFF_OWP = OFF_OSY + 2
TOTAL = OFF_OWP + 3


def unpack(x):
    bp = x[0:19]
    js = x[OFF_SCALE:OFF_SCALE + N_SCALE]
    tl = x[OFF_TILT:OFF_TILT + N_TILT].reshape(7, 2)
    oe = x[OFF_OE:OFF_OE + 3]
    osy = x[OFF_OSY:OFF_OSY + 2]
    owp = x[OFF_OWP:OFF_OWP + 3]
    return bp, js, tl, oe, osy, owp


def predict(ar, bp, js, tl, oe, osy, owp):
    params = ck.unpack_params(bp)
    ca = ar * js + params.joint_offsets
    Tfk = fk_combined(ca, tl, oe, osy, owp)
    Ttool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    Tbase = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return Tbase @ Tfk @ Ttool


TW, TS = 5.0, np.radians(8.0)
SW = 20.0


def make_res(angles, pos, quat):
    n = len(angles)

    def res(x):
        bp, js, tl, oe, osy, owp = unpack(x)
        params = ck.unpack_params(bp)
        pr = np.zeros((n, 3)); orr = np.zeros((n, 3))
        for i in range(n):
            Tp = predict(angles[i], bp, js, tl, oe, osy, owp)
            pr[i] = pos[i] - Tp[:3, 3] * 1000.0
            Rp = Tp[:3, :3]; Rm = Rotation.from_quat(quat[i]).as_matrix()
            orr[i] = Rotation.from_matrix(Rp.T @ Rm).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)), params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)), tl.ravel() * (TW / TS), (js - 1.0) * SW,
        ])
        return np.concatenate([pr.ravel(), orr.ravel(), reg])
    return res


def bounds():
    lo = np.concatenate([
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf] * 3, [-np.inf] * 3, [-np.inf] * 3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT), -0.05 * np.ones(3),
        -0.05 * np.ones(2), -0.05 * np.ones(3),
    ])
    up = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf] * 3, [np.inf] * 3, [np.inf] * 3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT), 0.05 * np.ones(3),
        0.05 * np.ones(2), 0.05 * np.ones(3),
    ])
    return lo, up


def x0(angles, pos, quat):
    bx, br = ck.initial_base_guess(angles[0], pos[0], quat[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), bx, br), np.ones(N_SCALE),
        np.zeros(N_TILT + 3 + 2 + 3),
    ])


def fit(angles, pos, quat):
    lo, up = bounds()
    return least_squares(make_res(angles, pos, quat), x0(angles, pos, quat), method="trf",
                          x_scale="jac", bounds=(lo, up), max_nfev=8000)


def rms(angles, pos, bp, js, tl, oe, osy, owp):
    e = np.zeros(len(angles))
    for i in range(len(angles)):
        Tp = predict(angles[i], bp, js, tl, oe, osy, owp)
        e[i] = np.linalg.norm(pos[i] - Tp[:3, 3] * 1000.0)
    return np.sqrt(np.mean(e ** 2))


def main():
    print(f"Fitting REDUCED {TOTAL}-param model (no Fourier/coupling/gravity) on {FIT_CSV}...")
    angles_all, pos_mm_all, quat_all = ck.load_poses_from_csv(FIT_CSV)
    n_all = len(angles_all)
    print(f"Loaded {n_all} poses")

    result = fit(angles_all, pos_mm_all, quat_all)
    bp, js, tl, oe, osy, owp = unpack(result.x)
    full_rms = rms(angles_all, pos_mm_all, bp, js, tl, oe, osy, owp)
    print(f"Full-dataset in-sample RMS (reduced model): {full_rms:.3f}mm")

    N_FOLDS = 8
    fold_bounds = np.linspace(0, n_all, N_FOLDS + 1).astype(int)
    fold_results = []
    print("\n=== Blocked 8-fold CV (REDUCED model, same folds as full model) ===")
    for f in range(N_FOLDS):
        lo_i, hi_i = fold_bounds[f], fold_bounds[f + 1]
        test_mask = np.zeros(n_all, dtype=bool)
        test_mask[lo_i:hi_i] = True
        train_mask = ~test_mask
        r = fit(angles_all[train_mask], pos_mm_all[train_mask], quat_all[train_mask])
        bp_f, js_f, tl_f, oe_f, osy_f, owp_f = unpack(r.x)
        train_rms = rms(angles_all[train_mask], pos_mm_all[train_mask], bp_f, js_f, tl_f, oe_f, osy_f, owp_f)
        test_rms = rms(angles_all[test_mask], pos_mm_all[test_mask], bp_f, js_f, tl_f, oe_f, osy_f, owp_f)
        n_test = int(test_mask.sum())
        fold_results.append({"fold": f, "n_test": n_test, "train_rms": float(train_rms), "test_rms": float(test_rms)})
        print(f"  fold {f} (idx {lo_i}-{hi_i}, n_test={n_test}): train={train_rms:.3f}mm test={test_rms:.3f}mm")

    pooled_sq = sum((fr["test_rms"] ** 2) * fr["n_test"] for fr in fold_results)
    pooled_n = sum(fr["n_test"] for fr in fold_results)
    pooled_rms = np.sqrt(pooled_sq / pooled_n)
    print(f"\nPooled blocked-CV test RMS (REDUCED model): {pooled_rms:.3f}mm")
    print(f"(Full model, for comparison, was 0.777mm pooled)")

    with open(OUT_DIR + "validate_48param_model_blocked_cross_validation_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "n_test", "train_rms_mm", "test_rms_mm"])
        for fr in fold_results:
            w.writerow([fr["fold"], fr["n_test"], f"{fr['train_rms']:.3f}", f"{fr['test_rms']:.3f}"])
        w.writerow(["POOLED", pooled_n, "", f"{pooled_rms:.3f}"])

    with open(OUT_DIR + "reduced_model_summary.json", "w") as f:
        json.dump({"full_dataset_rms_mm": float(full_rms), "pooled_blocked_cv_rms_mm": float(pooled_rms),
                    "folds": fold_results}, f, indent=2)


if __name__ == "__main__":
    main()
