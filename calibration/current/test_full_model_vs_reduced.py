"""Does the full 60-param model (coupling + gravity/elastostatic + the
shoulder_pitch Fourier term -- the pose-dependent corrections that can't be
baked into a static URDF) explain the error remaining in the
move_between_points/refit_moveit_ndi_rotation.py test, beyond what the
reduced (offset/scale/tilt/origin-only, URDF-representable) model already
captures?

Method: fit the full 60-param model on the 374-pose elbow-yaw-locked
reference dataset (quick_calibration_test_fixed_elbow_yaw.csv -- the same
one behind the deployed robot_calibration.cpp/URDF corrections), then
evaluate it on the 12 fresh poses from ros/moveit_ndi_accuracy_check.csv --
poses that were NEVER part of that fit, so this is a genuine held-out
comparison, not circular.

To isolate "does the full model's own internal pose-to-pose consistency
match NDI reality better," independent of any base-frame staleness
confound (the full model has its own fitted base_frame, from a different
session, which could itself be stale the same way the batch2 rotation
was), this refits its OWN best rotation via Kabsch on the SAME 12 poses --
exactly parallel to what refit_moveit_ndi_rotation.py did for the reduced
(MoveIt-reported) model. Comparing the two methods' resulting RMS residual,
both given their own best-fit rotation, isolates the *model* difference
(full vs. reduced) rather than any frame-alignment difference.

Usage:
    uv run --with numpy --with scipy python test_full_model_vs_reduced.py
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck
import csv

FIT_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_fixed_elbow_yaw.csv"
TEST_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/ros/moveit_ndi_accuracy_check.csv"

SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6
COUPLE_TERMS = [(SHOULDER_ROLL_IDX, SHOULDER_YAW_IDX, SHOULDER_YAW_IDX),
                (SHOULDER_YAW_IDX, ELBOW_YAW_IDX, ELBOW_YAW_IDX),
                (SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX, ELBOW_PITCH_IDX)]
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


def predict_full(ar, bp, js, tl, oe, osy, owp, fab, cc, gc):
    """Full 60-param prediction: offset+scale+tilt+origin+Fourier+coupling+gravity."""
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


def predict_reduced(ar, bp, tl, oe, osy, owp):
    """Reduced model: only what's actually baked into robot_calibration.cpp/
    the URDF -- offset+scale+tilt+origin, NO Fourier/coupling/gravity. This
    is the model MoveIt itself is actually using."""
    params = ck.unpack_params(bp)
    ca = ar.copy() + params.joint_offsets
    Tfk = fk_combined(ca, tl, oe, osy, owp)
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
            Tp = predict_full(angles[i], bp, js, tl, oe, osy, owp, fab, cc, gc)
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


def kabsch(P, Q):
    Pc = P - P.mean(axis=0); Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def rms_delta_error(R, P, Q):
    Pc = P - P.mean(axis=0); Qc = Q - Q.mean(axis=0)
    predicted = (R @ Pc.T).T
    err = np.linalg.norm(predicted - Qc, axis=1)
    return np.sqrt(np.mean(err ** 2)), err


def load_test_poses(path):
    joints, ndi_pts = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mx, my, mz = float(row["moveit_pose_x_mm"]), float(row["moveit_pose_y_mm"]), float(row["moveit_pose_z_mm"])
            qw = float(row["moveit_pose_qw"])
            if mx == 0.0 and my == 0.0 and mz == 0.0 and qw == 1.0:
                continue
            j = np.array([
                float(row["shoulder_roll_joint_rad"]), float(row["shoulder_pitch_joint_rad"]),
                float(row["shoulder_yaw_joint_rad"]), float(row["elbow_pitch_joint_rad"]),
                float(row["elbow_yaw_joint_rad"]), float(row["wrist_pitch_joint_rad"]),
                float(row["wrist_roll_joint_rad"]),
            ])
            n = np.array([float(row["moving_relative_fixed_tx_mm"]), float(row["moving_relative_fixed_ty_mm"]),
                          float(row["moving_relative_fixed_tz_mm"])])
            joints.append(j); ndi_pts.append(n)
    return joints, np.array(ndi_pts)


def main():
    print(f"Fitting full {TOTAL}-param model on {FIT_CSV}...")
    angles_all, pos_mm_all, quat_all = ck.load_poses_from_csv(FIT_CSV)
    print(f"Loaded {len(angles_all)} fit poses")
    result = fit(angles_all, pos_mm_all, quat_all)
    bp, js, tl, oe, osy, owp, fab, cc, gc = unpack(result.x)

    e = np.zeros(len(angles_all))
    for i in range(len(angles_all)):
        Tp = predict_full(angles_all[i], bp, js, tl, oe, osy, owp, fab, cc, gc)
        e[i] = np.linalg.norm(pos_mm_all[i] - Tp[:3, 3] * 1000.0)
    print(f"Fit in-sample RMS: {np.sqrt(np.mean(e ** 2)):.2f}mm (sanity check)\n")

    print(f"Loading held-out test poses from {TEST_CSV}...")
    test_joints, test_ndi = load_test_poses(TEST_CSV)
    print(f"Loaded {len(test_joints)} held-out poses (never in the fit above)\n")

    full_pred = np.array([predict_full(j, bp, js, tl, oe, osy, owp, fab, cc, gc)[:3, 3] * 1000.0
                           for j in test_joints])
    reduced_pred = np.array([predict_reduced(j, bp, tl, oe, osy, owp)[:3, 3] * 1000.0
                              for j in test_joints])

    R_full = kabsch(full_pred, test_ndi)
    R_reduced = kabsch(reduced_pred, test_ndi)

    rms_full, err_full = rms_delta_error(R_full, full_pred, test_ndi)
    rms_reduced, err_reduced = rms_delta_error(R_reduced, reduced_pred, test_ndi)

    print("=== Held-out comparison on the 12 move_between_points poses ===")
    print(f"REDUCED model (offset+scale+tilt+origin only -- what's actually in the URDF today):")
    print(f"  RMS: {rms_reduced:.3f}mm  max: {err_reduced.max():.3f}mm")
    print(f"FULL model (+ coupling + gravity + shoulder_pitch Fourier):")
    print(f"  RMS: {rms_full:.3f}mm  max: {err_full.max():.3f}mm")
    print(f"\nImprovement from adding the pose-dependent terms: "
          f"{(1 - rms_full / rms_reduced) * 100:.1f}%")

    print("\n=== Per-point error (mm) ===")
    print(f"{'point':>5} {'reduced':>9} {'full':>9} {'delta':>9}")
    for i in range(len(err_reduced)):
        delta = err_full[i] - err_reduced[i]
        print(f"{i + 1:>5} {err_reduced[i]:>9.3f} {err_full[i]:>9.3f} {delta:>+9.3f}")


if __name__ == "__main__":
    main()
