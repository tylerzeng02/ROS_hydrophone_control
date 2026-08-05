import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)
n = len(angles)


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)
rms, errs = ck.rms_position_error_mm(result.x, angles, pos_mm)
print(f"Baseline fit RMS: {rms:.2f} mm\n")

print("Correlation of per-pose position error against pairwise joint-angle PRODUCTS")
print("(checks for coupling/cross-talk between joints that no per-joint-independent")
print("parameter could capture):\n")

results = []
for i in range(ck.N_JOINTS):
    for j in range(i + 1, ck.N_JOINTS):
        product = angles[:, i] * angles[:, j]
        corr = np.corrcoef(product, errs)[0, 1]
        results.append((abs(corr), corr, ck.JOINT_NAMES[i], ck.JOINT_NAMES[j]))

results.sort(reverse=True)
print(f"{'joint pair':50s} {'correlation':>12s}")
for abscorr, corr, name_i, name_j in results:
    print(f"{name_i + ' * ' + name_j:50s} {corr:12.3f}")

# Also check correlation against individual joint angles again for reference
print("\nFor reference, correlation of error against individual joint angles:")
for i in range(ck.N_JOINTS):
    corr = np.corrcoef(angles[:, i], errs)[0, 1]
    print(f"  {ck.JOINT_NAMES[i]:24s} {corr:8.3f}")
