import numpy as np
import calibrate_kinematics as ck

angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(
    "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
)
print("n poses:", len(angles))
print("angles[0] (rad):", angles[0])
print("angles[0] (deg):", np.degrees(angles[0]))
print("pos_mm[0]:", pos_mm[0])
print("quat_xyzw[0]:", quat_xyzw[0])
print("quat norm[0]:", np.linalg.norm(quat_xyzw[0]))

# Nominal (zero-correction) prediction for pose 0, using identity base/tool
zero_params = ck.CalibParams(
    np.zeros(7), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
)
T_fk = ck.forward_kinematics(angles[0])
print("\nNominal FK (base_link -> virtual_endeffector) for pose 0:")
print("Position (mm):", T_fk[:3, 3] * 1000.0)
print("Rotation matrix:\n", T_fk[:3, :3])

# distance range across all poses (sanity check measured data spread)
dists = np.linalg.norm(pos_mm, axis=1)
print("\nmeasured moving-relative-fixed distance stats (mm): min=%.1f max=%.1f mean=%.1f" %
      (dists.min(), dists.max(), dists.mean()))

# initial base guess + resulting prediction error for pose 0 itself (should be ~0 by construction)
base_xyz, base_rpy = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
guess_params = ck.CalibParams(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz, base_rpy)
T_pred0 = ck.predict_relative_pose(angles[0], guess_params)
print("\nPose 0 self-consistency check (should match exactly):")
print("predicted pos (mm):", T_pred0[:3,3]*1000.0)
print("measured pos (mm): ", pos_mm[0])

# now check RMS error across ALL poses using ONLY this single pose's base guess + zero corrections
errs = []
for i in range(len(angles)):
    T = ck.predict_relative_pose(angles[i], guess_params)
    errs.append(np.linalg.norm(T[:3,3]*1000.0 - pos_mm[i]))
errs = np.array(errs)
print("\nRMS error across all poses using pose-0-derived base + zero joint/tool corrections: %.2f mm" % np.sqrt(np.mean(errs**2)))
print("min/max/mean error:", errs.min(), errs.max(), errs.mean())
