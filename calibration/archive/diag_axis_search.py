import itertools
import numpy as np
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)

original_axes = ck.JOINT_AXES.copy()

def rms_for_sign_combo(signs):
    ck.JOINT_AXES = original_axes * np.array(signs).reshape(-1, 1)
    base_xyz, base_rpy = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
    params = ck.CalibParams(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz, base_rpy)
    errs = []
    for i in range(len(angles)):
        T = ck.predict_relative_pose(angles[i], params)
        errs.append(np.linalg.norm(T[:3, 3] * 1000.0 - pos_mm[i]))
    return np.sqrt(np.mean(np.array(errs) ** 2))

best = None
results = []
for signs in itertools.product([1, -1], repeat=7):
    rms = rms_for_sign_combo(signs)
    results.append((rms, signs))

results.sort(key=lambda x: x[0])
print("Top 10 sign combinations by RMS error (mm):")
for rms, signs in results[:10]:
    print(f"  {rms:8.2f} mm  signs={signs}")

ck.JOINT_AXES = original_axes
print("\nBaseline (all +1, current code):", rms_for_sign_combo((1,1,1,1,1,1,1)))
