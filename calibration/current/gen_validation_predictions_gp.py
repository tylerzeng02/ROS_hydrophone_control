"""Physical model + GP residual layer predictions for validation test
points -- extends gen_validation_predictions.py (physical-only) with the
GP residual correction validated in gp_residual_fixed_elbow_yaw.py (every
fold improved, pooled blocked-CV 0.78mm -> 0.66mm on the 374-pose
elbow-yaw-locked dataset).

Unlike the blocked-CV script, this fits the physical model AND the GP on
the FULL fit dataset (no held-out fold) -- correct for generating
predictions at genuinely new points, since there's no "test fold" here,
just real new poses never in the fit data at all.

Usage:
    python gen_validation_predictions_gp.py
    python gen_validation_predictions_gp.py --fit-csv a.csv --points ticks.csv --out predictions.csv

Requires: numpy, scipy, scikit-learn
"""
import argparse
import numpy as np
import csv
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
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
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u)
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
    pl, al = [], []
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
        pl.append(p); al.append(a)
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
    return least_squares(make_res(angles, pos, quat), x0(angles, pos, quat),
                          method="trf", x_scale="jac", bounds=(lo, up), max_nfev=5000)


def predicted_positions_mm(angles, x):
    bp, js, tl, oe, osy, owp, fab, cc, gc = unpack(x)
    out = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        Tp = predict(angles[i], bp, js, tl, oe, osy, owp, fab, cc, gc)
        out[i] = Tp[:3, 3] * 1000.0
    return out


def ticks_to_radians(ticks):
    return (np.array(ticks, dtype=float) - np.array(ck.NOMINAL_ZERO_TICKS)) / ck.TICKS_PER_RADIAN


GP_KERNEL_TEMPLATE = lambda n_dims: (
    ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3))
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-csv",
        default="C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_fixed_elbow_yaw.csv",
    )
    parser.add_argument(
        "--points",
        default="C:/Users/ConformalUser/Desktop/cyton_setup/build/validation_ticks_fixed_elbow_yaw.csv",
    )
    parser.add_argument(
        "--out",
        default="C:/Users/ConformalUser/Desktop/cyton_setup/build/validation_points_fixed_elbow_yaw_gp.csv",
    )
    args = parser.parse_args()

    angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(args.fit_csv)
    n_all = len(angles_all)
    print(f"Fitting {TOTAL}-param physical model on {n_all} poses...")
    result = fit(angles_all, pos_mm_all, quat_xyzw_all)
    phys_pred_all = predicted_positions_mm(angles_all, result.x)
    phys_e = np.linalg.norm(pos_mm_all - phys_pred_all, axis=1)
    print(f"Physical model in-sample RMS: {np.sqrt(np.mean(phys_e ** 2)):.2f}mm")

    residual_vec = pos_mm_all - phys_pred_all  # (n, 3) mm

    angle_mean = angles_all.mean(axis=0)
    angle_std = angles_all.std(axis=0)
    angle_std[angle_std < 1e-6] = 1.0
    angles_std = (angles_all - angle_mean) / angle_std

    print("Fitting GP residual layer (one per x/y/z dimension) on full dataset...")
    gps = []
    for dim in range(3):
        kernel = GP_KERNEL_TEMPLATE(7)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True, alpha=1e-6, random_state=0)
        gp.fit(angles_std, residual_vec[:, dim])
        gps.append(gp)

    corrected_pred_all = phys_pred_all + np.column_stack([
        gp.predict(angles_std) for gp in gps
    ])
    combined_e = np.linalg.norm(pos_mm_all - corrected_pred_all, axis=1)
    print(f"Physical+GP in-sample RMS: {np.sqrt(np.mean(combined_e ** 2)):.2f}mm (sanity check only)")

    ticks_rows = []
    with open(args.points, newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        try:
            [int(v) for v in first[:7]]
            ticks_rows.append(first)
        except ValueError:
            pass
        for row in reader:
            if row:
                ticks_rows.append(row)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tick_0", "tick_1", "tick_2", "tick_3", "tick_4", "tick_5", "tick_6",
            "predicted_x_mm", "predicted_y_mm", "predicted_z_mm",
        ])
        for row in ticks_rows:
            ticks = [int(v) for v in row[:7]]
            a = ticks_to_radians(ticks)
            Tp_phys = predict(a, *unpack(result.x))
            phys_pos = Tp_phys[:3, 3] * 1000.0
            a_std = (a - angle_mean) / angle_std
            gp_corr = np.array([gp.predict(a_std.reshape(1, -1))[0] for gp in gps])
            final_pos = phys_pos + gp_corr
            writer.writerow(ticks + [f"{final_pos[0]:.4f}", f"{final_pos[1]:.4f}", f"{final_pos[2]:.4f}"])
            print(f"  ticks={ticks} -> physical=({phys_pos[0]:.2f},{phys_pos[1]:.2f},{phys_pos[2]:.2f}) "
                  f"+GP=({final_pos[0]:.2f},{final_pos[1]:.2f},{final_pos[2]:.2f}) "
                  f"[GP correction: {np.linalg.norm(gp_corr):.2f}mm]")

    print(f"\nWrote {len(ticks_rows)} predicted test points to {args.out}")


if __name__ == "__main__":
    main()
