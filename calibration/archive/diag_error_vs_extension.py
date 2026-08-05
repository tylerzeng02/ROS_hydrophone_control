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
print(f"Baseline (19-param) fit RMS: {rms:.2f} mm\n")

# "Extension" proxy: distance from shoulder_roll's origin (first joint) to
# the nominal FK end-effector position, for each pose -- how far reached-out
# the arm is, independent of direction.
extension_mm = np.zeros(n)
for i in range(n):
    T_fk = ck.forward_kinematics(angles[i])
    ee_pos = T_fk[:3, 3]
    extension_mm[i] = np.linalg.norm(ee_pos) * 1000.0

corr = np.corrcoef(extension_mm, errs)[0, 1]
print(f"Correlation of position error with arm extension/reach: {corr:.3f}")
print(f"Extension range: {extension_mm.min():.1f} - {extension_mm.max():.1f} mm\n")

# Bin by extension quartiles and show mean error per bin
order = np.argsort(extension_mm)
quartile_size = n // 4
print("Extension quartile -> mean error (mm):")
for q in range(4):
    start = q * quartile_size
    end = (q + 1) * quartile_size if q < 3 else n
    idx = order[start:end]
    print(f"  Q{q+1} (ext {extension_mm[idx].min():.0f}-{extension_mm[idx].max():.0f}mm): "
          f"mean error = {errs[idx].mean():.2f} mm, n={len(idx)}")
