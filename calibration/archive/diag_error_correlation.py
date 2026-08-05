import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

def residual(x):
    joint_offsets = x[0:7]
    tool_xyz = x[7:10]
    tool_rpy = x[10:13]
    base_xyz = x[13:16]
    base_rpy = x[16:19]
    params = ck.CalibParams(joint_offsets, tool_xyz, tool_rpy, base_xyz, base_rpy)
    res = np.zeros((len(angles), 3))
    for i in range(len(angles)):
        T = ck.predict_relative_pose(angles[i], params)
        res[i] = pos_mm[i] - T[:3, 3] * 1000.0
    return res.ravel()

base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = np.concatenate([np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0])
result = least_squares(residual, x0, method="lm", verbose=0, max_nfev=50000)
final_res = residual(result.x).reshape(-1, 3)
errs = np.linalg.norm(final_res, axis=1)

print("Correlation of per-pose position error magnitude with each joint's angle:")
for j in range(7):
    c = np.corrcoef(np.degrees(angles[:, j]), errs)[0, 1]
    c_abs = np.corrcoef(np.abs(np.degrees(angles[:, j])), errs)[0, 1]
    print(f"  joint {j}: corr(angle, err)={c:+.3f}  corr(|angle|, err)={c_abs:+.3f}")

# Also check correlation with each joint's absolute deviation from pose[0] (large swings)
print("\nWorst 10 poses by error:")
worst_idx = np.argsort(errs)[-10:]
for i in worst_idx:
    print(f"  pose idx {i}: err={errs[i]:.1f}mm, angles(deg)={np.round(np.degrees(angles[i]),1)}")

print("\nBest 10 poses by error:")
best_idx = np.argsort(errs)[:10]
for i in best_idx:
    print(f"  pose idx {i}: err={errs[i]:.1f}mm, angles(deg)={np.round(np.degrees(angles[i]),1)}")
