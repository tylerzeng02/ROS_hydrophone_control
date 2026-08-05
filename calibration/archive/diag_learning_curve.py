import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_all, pos_mm_all, quat_xyzw_all = ck.load_poses_from_csv(CSV_PATH)
n_all = len(angles_all)

# Fixed held-out set (last 30 poses, never used for training in this test),
# so every training-size point is judged against the exact same unseen data.
rng = np.random.default_rng(7)
perm = rng.permutation(n_all)
test_idx = perm[:30]
pool_idx = perm[30:]  # 160 poses available to draw increasing training sets from

angles_te, pos_te, quat_te = angles_all[test_idx], pos_mm_all[test_idx], quat_xyzw_all[test_idx]


def fit_and_eval(n_train):
    idx = pool_idx[:n_train]
    a, p, q = angles_all[idx], pos_mm_all[idx], quat_xyzw_all[idx]

    def residual(x):
        return ck.residual_function(x, a, p, q)

    base_xyz0, base_rpy0 = ck.initial_base_guess(a[0], p[0], q[0])
    x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
    result = least_squares(residual, x0, method="lm", max_nfev=100000)

    train_rms, _ = ck.rms_position_error_mm(result.x, a, p)
    test_rms, _ = ck.rms_position_error_mm(result.x, angles_te, pos_te)
    return train_rms, test_rms


sizes = [10, 20, 30, 40, 50, 70, 90, 110, 130, 160]
print(f"Fixed held-out set: 30 poses (never trained on)\n")
print(f"{'n_train':>8} {'train RMS':>10} {'held-out RMS':>13}")

prev_test = None
for n_train in sizes:
    train_rms, test_rms = fit_and_eval(n_train)
    delta = f"({test_rms - prev_test:+.2f})" if prev_test is not None else ""
    print(f"{n_train:>8} {train_rms:>10.2f} {test_rms:>13.2f} {delta}")
    prev_test = test_rms
