"""Fits the deployed 48-param model (offset/scale/tilt/origin) plus a
shoulder_pitch Fourier term on the deployed dataset and prints every
parameter. This is the script that produced the values hardcoded into
src/robot_calibration.cpp and references/cyton_gamma_1500_trac_ik.urdf.

Only offset/scale/tilt/origin (the static 48 params) are ever baked into
those files. The Fourier term is pose-dependent and cannot be represented
in a static URDF, so it is printed for reference but not deployed.
"""

import os

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

# Resolved relative to this script's own location, not a fixed machine
# path, so this runs regardless of where the repo is checked out.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATHS = [
    os.path.join(_REPO_ROOT, 'calibration', 'data', 'deployed_model_training_dataset_374pose.csv'),
]
angles_list, pos_list, quat_list = [], [], []
for p in CSV_PATHS:
    a, pos, q = ck.load_poses_from_csv(p)
    angles_list.append(a); pos_list.append(pos); quat_list.append(q)
    print(f'Loaded {len(a)} poses from {p}')
angles_all = np.concatenate(angles_list)
pos_mm_all = np.concatenate(pos_list)
quat_xyzw_all = np.concatenate(quat_list)
n_all = len(angles_all)
print(f'Total combined: {n_all} poses')

SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, WRIST_PITCH_IDX = 3, 5

PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u); v = np.cross(a, u)
    PERP.append((u, v))


def tilted_axes(tilt_all):
    """Applies a small per-joint axis tilt to each nominal joint axis.

    Args:
        tilt_all: (7, 2) tilt components along each joint's own
            perpendicular basis (see PERP).

    Returns:
        List of 7 tilted, unit-normalized joint axes.
    """
    axes = []
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]; u, v = PERP[i]
        p = a + tilt_all[i, 0] * u + tilt_all[i, 1] * v
        axes.append(p / np.linalg.norm(p))
    return axes


def fk_combined(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    """Forward kinematics with tilt and origin corrections applied.

    Args:
        ja: (7,) corrected joint angles, radians.
        tilt_all: (7, 2) axis tilt components; see tilted_axes().
        o_elbow: (3,) elbow_pitch origin correction, meters.
        o_sy_xz: (2,) shoulder_yaw origin x,z correction, meters.
        o_wp: (3,) wrist_pitch origin correction, meters.

    Returns:
        (4, 4) base_link -> virtual_endeffector transform.
    """
    axes = tilted_axes(tilt_all); T = np.eye(4)
    for i in range(ck.N_JOINTS):
        origin = ck.JOINT_ORIGINS_M[i].copy()
        if i == WRIST_PITCH_IDX: origin = origin + o_wp
        if i == ELBOW_PITCH_IDX: origin = origin + o_elbow
        if i == SHOULDER_YAW_IDX: origin = origin + np.array([o_sy_xz[0], 0.0, o_sy_xz[1]])
        R = Rotation.from_rotvec(axes[i] * ja[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, origin)
    return T


N_TILT, N_SCALE = 14, 7
OFF_SCALE = 19; OFF_TILT = OFF_SCALE + N_SCALE; OFF_OE = OFF_TILT + N_TILT
OFF_OSY = OFF_OE + 3; OFF_OWP = OFF_OSY + 2; OFF_F = OFF_OWP + 3
TOTAL = OFF_F + 2


def unpack(x):
    """Splits the flat parameter vector into named groups.

    Args:
        x: (TOTAL,) flat parameter vector.

    Returns:
        Tuple of (base_params (19,), joint_scales (7,), axis_tilts (7,2),
        elbow_pitch_origin (3,), shoulder_yaw_origin_xz (2,),
        wrist_pitch_origin (3,), shoulder_pitch_fourier_ab (2,)).
    """
    bp = x[0:19]; js = x[OFF_SCALE:OFF_SCALE + N_SCALE]; tl = x[OFF_TILT:OFF_TILT + N_TILT].reshape(7, 2)
    oe = x[OFF_OE:OFF_OE + 3]; osy = x[OFF_OSY:OFF_OSY + 2]; owp = x[OFF_OWP:OFF_OWP + 3]
    fab = x[OFF_F:OFF_F + 2]
    return bp, js, tl, oe, osy, owp, fab


def predict(ar, bp, js, tl, oe, osy, owp, fab):
    """Predicts the moving-marker pose for one measured joint-angle set.

    Args:
        ar: (7,) raw measured joint angles.
        bp: (19,) base calibrate_kinematics parameters.
        js: (7,) joint gear-ratio scales.
        tl: (7, 2) axis tilt components.
        oe: (3,) elbow_pitch origin correction.
        osy: (2,) shoulder_yaw origin x,z correction.
        owp: (3,) wrist_pitch origin correction.
        fab: (2,) shoulder_pitch Fourier sin/cos coefficients.

    Returns:
        (4, 4) predicted fixed-marker -> moving-marker transform.
    """
    params = ck.unpack_params(bp)
    ca = (ar * js + params.joint_offsets).copy()
    rsp = ar[SHOULDER_PITCH_IDX]
    ca[SHOULDER_PITCH_IDX] += fab[0] * np.sin(rsp) + fab[1] * np.cos(rsp)
    Tfk = fk_combined(ca, tl, oe, osy, owp)
    Ttool = ck.build_tool_transform(params.tool_xyz, params.tool_rpy)
    Tbase = ck.build_base_transform(params.base_xyz, params.base_rpy)
    return Tbase @ Tfk @ Ttool


TW, TS = 5.0, np.radians(8.0); SW = 20.0; FW, FS = 5.0, np.radians(5.0)


def make_res(angles, pos, quat):
    """Builds the least_squares residual function for one dataset.

    Args:
        angles: (N, 7) measured joint angles.
        pos: (N, 3) measured positions, millimeters.
        quat: (N, 4) measured orientations, scipy order.

    Returns:
        A residual function taking the flat parameter vector and
        returning position, orientation, and regularization residuals.
    """
    n = len(angles)

    def res(x):
        bp, js, tl, oe, osy, owp, fab = unpack(x); params = ck.unpack_params(bp)
        pr = np.zeros((n, 3)); orr = np.zeros((n, 3))
        for i in range(n):
            Tp = predict(angles[i], bp, js, tl, oe, osy, owp, fab)
            pr[i] = pos[i] - Tp[:3, 3] * 1000.0
            Rp = Tp[:3, :3]; Rm = Rotation.from_quat(quat[i]).as_matrix()
            orr[i] = Rotation.from_matrix(Rp.T @ Rm).as_rotvec() * 100.0 * 0.3
        reg = np.concatenate([
            params.joint_offsets * (8.0 / np.radians(8.0)),
            params.tool_xyz * (1.0 / 0.01),
            params.tool_rpy * (1.0 / np.radians(10.0)),
            tl.ravel() * (TW / TS),
            (js - 1.0) * SW,
            fab * (FW / FS),
        ])
        return np.concatenate([pr.ravel(), orr.ravel(), reg])
    return res


def bounds():
    """Builds the (lower, upper) bound arrays for least_squares().

    Bounds are physically motivated per parameter type (e.g. 15deg max
    tilt/joint-offset, 10cm max tool translation), wide enough not to
    clip any value seen in prior fits. tool_rpy, base_xyz, and base_rpy
    are left unbounded: base has no known nominal value, same as in
    calibrate_kinematics.py, and tool_rpy is widened here past
    calibrate_kinematics.py's tighter +-10deg bound, which caused
    bound-pinning on this dataset.

    Returns:
        Tuple of (lower, upper), each (TOTAL,).
    """
    lo = np.concatenate([
        -np.radians(15.0) * np.ones(7), -0.10 * np.ones(3), [-np.inf] * 3, [-np.inf] * 3, [-np.inf] * 3,
        0.90 * np.ones(N_SCALE), -np.radians(15.0) * np.ones(N_TILT),
        -0.05 * np.ones(3), -0.05 * np.ones(2), -0.05 * np.ones(3),
        -np.radians(15.0) * np.ones(2),
    ])
    up = np.concatenate([
        np.radians(15.0) * np.ones(7), 0.10 * np.ones(3), [np.inf] * 3, [np.inf] * 3, [np.inf] * 3,
        1.10 * np.ones(N_SCALE), np.radians(15.0) * np.ones(N_TILT),
        0.05 * np.ones(3), 0.05 * np.ones(2), 0.05 * np.ones(3),
        np.radians(15.0) * np.ones(2),
    ])
    return lo, up


def x0(angles, pos, quat):
    """Builds the initial parameter guess: zero corrections, unit scales,
    and a closed-form base-frame guess from the first pose.

    Args:
        angles: (N, 7) measured joint angles.
        pos: (N, 3) measured positions, millimeters.
        quat: (N, 4) measured orientations, scipy order.

    Returns:
        (TOTAL,) initial parameter vector.
    """
    bx, br = ck.initial_base_guess(angles[0], pos[0], quat[0])
    return np.concatenate([
        ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), bx, br),
        np.ones(N_SCALE),
        np.zeros(N_TILT + 3 + 2 + 3 + 2),
    ])


def fit(angles, pos, quat):
    """Runs the full bounded least-squares fit.

    Args:
        angles: (N, 7) measured joint angles.
        pos: (N, 3) measured positions, millimeters.
        quat: (N, 4) measured orientations, scipy order.

    Returns:
        scipy.optimize.OptimizeResult from least_squares().
    """
    lo, up = bounds()
    return least_squares(
        make_res(angles, pos, quat), x0(angles, pos, quat),
        method='trf', x_scale='jac', bounds=(lo, up), max_nfev=8000,
    )


print(f'Fitting {TOTAL}-param model on {n_all} combined poses...')
result = fit(angles_all, pos_mm_all, quat_xyzw_all)
bp, js, tl, oe, osy, owp, fab = unpack(result.x)
params = ck.unpack_params(bp)

e = np.zeros(n_all)
for i in range(n_all):
    Tp = predict(angles_all[i], bp, js, tl, oe, osy, owp, fab)
    e[i] = np.linalg.norm(pos_mm_all[i] - Tp[:3, 3] * 1000.0)
print(f'Full-dataset in-sample RMS: {np.sqrt(np.mean(e**2)):.2f}mm')
_, s, _ = np.linalg.svd(result.jac)
print(f'Condition number: {s[0]/s[-1]:.1f}')

JOINT_NAMES = ['shoulder_roll', 'shoulder_pitch', 'shoulder_yaw', 'elbow_pitch', 'elbow_yaw', 'wrist_pitch', 'wrist_roll']
print('\n=== JOINT OFFSETS (deg) ===')
for i, name in enumerate(JOINT_NAMES):
    print(f'  {name}: {np.degrees(params.joint_offsets[i]):+.4f} deg')
print('\n=== JOINT SCALES ===')
for i, name in enumerate(JOINT_NAMES):
    print(f'  {name}: {js[i]:.6f}')
print('\n=== JOINT AXIS TILTS (u,v components, rad) ===')
for i, name in enumerate(JOINT_NAMES):
    print(f'  {name}: u={tl[i,0]:+.6f} v={tl[i,1]:+.6f}  (perp basis u={PERP[i][0]}, v={PERP[i][1]})')
print('\n=== ORIGIN CORRECTIONS (mm) ===')
print(f'  elbow_pitch (idx3) xyz: {oe*1000}')
print(f'  shoulder_yaw (idx2) x,z: {osy*1000}')
print(f'  wrist_pitch (idx5) xyz: {owp*1000}')
print('\n=== TOOL FRAME (virtual_endeffector -> moving marker) ===')
print(f'  xyz mm: {params.tool_xyz*1000}')
print(f'  rpy deg: {np.degrees(params.tool_rpy)}')
print('\n=== BASE FRAME (fixed marker -> base_link), calibration-time only, not for deployment ===')
print(f'  xyz mm: {params.base_xyz*1000}')
print(f'  rpy deg: {np.degrees(params.base_rpy)}')
print('\n=== SHOULDER_PITCH FOURIER ===')
print(f'  a={fab[0]:.6f} b={fab[1]:.6f}')
