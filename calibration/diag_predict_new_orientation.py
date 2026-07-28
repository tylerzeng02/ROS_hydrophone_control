import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

# Fit the baseline (19-param) model on the ORIGINAL 287 poses only
TRAIN_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_287pose.csv"
angles_tr, pos_tr, quat_tr = ck.load_poses_from_csv(TRAIN_CSV)

def residual(x):
    return ck.residual_function(x, angles_tr, pos_tr, quat_tr)

base_xyz0, base_rpy0 = ck.initial_base_guess(angles_tr[0], pos_tr[0], quat_tr[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)
train_rms, _ = ck.rms_position_error_mm(result.x, angles_tr, pos_tr)
print(f"Trained on 287 poses (in-distribution): RMS = {train_rms:.2f} mm\n")

# Now evaluate this SAME fitted model on the NEW orientation-diverse poses
# it has never seen -- a real out-of-distribution test.
NEW_CSV = "C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test.csv"
angles_new, pos_new, quat_new = ck.load_poses_from_csv(NEW_CSV)
print(f"Evaluating on {len(angles_new)} NEW orientation-diverse poses (never used in fitting)\n")

new_rms, new_errs = ck.rms_position_error_mm(result.x, angles_new, pos_new)
print(f"Out-of-distribution RMS on new orientation poses: {new_rms:.2f} mm")
print(f"  individual errors: {np.round(new_errs, 2)}")
print(f"  median: {np.median(new_errs):.2f}  max: {new_errs.max():.2f}")

print(f"\nComparison:")
print(f"  In-distribution (287-pose training):  {train_rms:.2f} mm")
print(f"  Out-of-distribution (new orientations): {new_rms:.2f} mm")
print(f"  Ratio: {new_rms/train_rms:.2f}x")
