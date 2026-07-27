import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)


def unbounded_fit(idx):
    a, p, q = angles[idx], pos_mm[idx], quat_xyzw[idx]

    def residual(x):
        return ck.residual_function(x, a, p, q)

    base_xyz0, base_rpy0 = ck.initial_base_guess(a[0], p[0], q[0])
    x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
    result = least_squares(residual, x0, method="lm", max_nfev=100000)
    return result


for seed in [0, 1, 2, 3, 4]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(n * 0.2)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    result = unbounded_fit(train_idx)
    train_rms, _ = ck.rms_position_error_mm(result.x, angles[train_idx], pos_mm[train_idx])
    test_rms, _ = ck.rms_position_error_mm(result.x, angles[test_idx], pos_mm[test_idx])

    print(f"seed={seed}: n_train={len(train_idx)} n_test={len(test_idx)}  "
          f"train RMS={train_rms:.2f} mm  held-out RMS={test_rms:.2f} mm  "
          f"gap={test_rms - train_rms:+.2f} mm")
