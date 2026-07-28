"""Quick targeted check: are joint0 (shoulder_roll) and joint6 (wrist_roll)
axis-tilt corrections structurally degenerate with base_rpy / tool_rpy,
the same way their ORIGIN corrections are? These are the two suspects by
direct analogy (first/last joint in the chain, nothing rotational between
them and the free frame). Interior joints (1-5) don't have an analogous
suspect a priori -- those get checked via the full-model Jacobian SVD
instead (see diag_tilt_sweep_fit.py), the same way the elbow/shoulder_yaw
degeneracy was actually found in practice.
"""
import numpy as np
import calibrate_kinematics as ck

rng = np.random.default_rng(0)


def perp_basis(axis):
    axis = axis / np.linalg.norm(axis)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def fk_full(angles_rad, joint_idx=-1, d_uv=None, base_rpy=None, tool_rpy=None):
    T = np.eye(4)
    for i in range(ck.N_JOINTS):
        a = ck.JOINT_AXES[i]
        if i == joint_idx and d_uv is not None:
            u, v = perp_basis(a)
            perturbed = a + d_uv[0] * u + d_uv[1] * v
            a = perturbed / np.linalg.norm(perturbed)
        R = ck.Rotation.from_rotvec(a * angles_rad[i]).as_matrix()
        T = T @ ck.homogeneous_transform(R, ck.JOINT_ORIGINS_M[i])
    T_base = ck.build_base_transform(np.zeros(3), base_rpy if base_rpy is not None else np.zeros(3))
    T_tool = ck.build_tool_transform(np.zeros(3), tool_rpy if tool_rpy is not None else np.zeros(3))
    return T_base @ T @ T_tool


def col_tilt(angles, joint_idx, comp, h=1e-6):
    d = np.zeros(2); d[comp] = h
    Tp = fk_full(angles, joint_idx, d)
    Tm = fk_full(angles, joint_idx, -d)
    dp = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
    dR = ck.Rotation.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * h)
    return np.concatenate([dp, dR])


def col_rpy(angles, which, comp, h=1e-6):
    d = np.zeros(3); d[comp] = h
    kwargs = {"base_rpy": d} if which == "base" else {"tool_rpy": d}
    kwargs_m = {"base_rpy": -d} if which == "base" else {"tool_rpy": -d}
    Tp = fk_full(angles, **kwargs)
    Tm = fk_full(angles, **kwargs_m)
    dp = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
    dR = ck.Rotation.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * h)
    return np.concatenate([dp, dR])


def random_angles():
    angles = np.zeros(7)
    for j in range(7):
        lo, hi = ck.JOINT_TICK_RANGES[j]
        tick = rng.uniform(lo + 100, hi - 100)
        angles[j] = (tick - ck.NOMINAL_ZERO_TICKS[j]) / ck.TICKS_PER_RADIAN
    return angles


def check_pair(joint_idx, joint_name, rpy_which):
    print(f"--- {joint_name} (joint {joint_idx}) tilt vs {rpy_which}_rpy ---")
    # base_rpy/tool_rpy have 3 components; tilt has 2 (perp-plane only).
    # Build the 3x3 map from tilt-space(2)+angle(1, trivially unused) isn't
    # needed -- just check whether the 2 tilt columns lie in the SAME
    # subspace spanned by the 3 rpy columns (rank should stay <=3 if fully
    # degenerate, or grow if tilt adds new information).
    for _ in range(3):
        angles = random_angles()
        rpy_cols = np.stack([col_rpy(angles, rpy_which, c) for c in range(3)], axis=1)  # 6x3
        tilt_cols = np.stack([col_tilt(angles, joint_idx, c) for c in range(2)], axis=1)  # 6x2
        combined = np.concatenate([rpy_cols, tilt_cols], axis=1)  # 6x5
        rank_rpy_only = np.linalg.matrix_rank(rpy_cols, tol=1e-6)
        rank_combined = np.linalg.matrix_rank(combined, tol=1e-6)
        print(f"  rank(rpy only)={rank_rpy_only}  rank(rpy+tilt)={rank_combined}  "
              f"-> tilt adds {rank_combined - rank_rpy_only} new independent direction(s)")


check_pair(0, "shoulder_roll", "base")
check_pair(6, "wrist_roll", "tool")
