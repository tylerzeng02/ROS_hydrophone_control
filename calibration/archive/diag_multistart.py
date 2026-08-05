import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])

rng = np.random.default_rng(42)
n_trials = 8

print(f"{'trial':>5} {'RMS(mm)':>9} {'tool_xyz(mm)':>28} {'tool_rpy(deg)':>28}")
results = []
for trial in range(n_trials):
    if trial == 0:
        x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
    else:
        joint_perturb = np.radians(rng.uniform(-15, 15, size=7))
        tool_xyz_perturb = rng.uniform(-0.05, 0.05, size=3)
        tool_rpy_perturb = np.radians(rng.uniform(-90, 90, size=3))
        base_xyz_perturb = base_xyz0 + rng.uniform(-0.1, 0.1, size=3)
        base_rpy_perturb = base_rpy0 + np.radians(rng.uniform(-30, 30, size=3))
        x0 = ck.pack_params(joint_perturb, tool_xyz_perturb, tool_rpy_perturb,
                             base_xyz_perturb, base_rpy_perturb)

    result = least_squares(residual, x0, method="lm", max_nfev=100000)
    rms, _ = ck.rms_position_error_mm(result.x, angles, pos_mm)
    params = ck.unpack_params(result.x)
    results.append((rms, params.tool_xyz * 1000.0, np.degrees(params.tool_rpy)))
    print(f"{trial:>5} {rms:>9.2f}   {str(np.round(params.tool_xyz*1000,1)):>26}   {str(np.round(np.degrees(params.tool_rpy),1)):>26}")

best = min(results, key=lambda r: r[0])
print(f"\nBest RMS across {n_trials} restarts: {best[0]:.2f} mm")
print(f"  tool_xyz: {best[1]}")
print(f"  tool_rpy: {best[2]}")
