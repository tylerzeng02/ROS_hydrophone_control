import numpy as np
from scipy.optimize import least_squares
import calibrate_kinematics as ck

CSV_PATH = "C:/Users/ConformalUser/Desktop/cyton_setup/build/five_pose_ndi_capture.csv"
angles, pos_mm, quat_xyzw = ck.load_poses_from_csv(CSV_PATH)


def residual(x):
    return ck.residual_function(x, angles, pos_mm, quat_xyzw)


base_xyz0, base_rpy0 = ck.initial_base_guess(angles[0], pos_mm[0], quat_xyzw[0])
x0 = ck.pack_params(np.zeros(7), np.zeros(3), np.zeros(3), base_xyz0, base_rpy0)
result = least_squares(residual, x0, method="lm", max_nfev=100000)

params = ck.unpack_params(result.x)
n = len(angles)
residual_vecs = np.zeros((n, 3))
mags = np.zeros(n)

for i in range(n):
    T_pred = ck.predict_relative_pose(angles[i], params)
    pred_pos = T_pred[:3, 3] * 1000.0
    residual_vecs[i] = pos_mm[i] - pred_pos
    mags[i] = np.linalg.norm(residual_vecs[i])

print("Confirming 'error' metric: straight-line (Euclidean) distance per pose.")
print(f"RMS of straight-line distances: {np.sqrt(np.mean(mags**2)):.2f} mm")
print(f"Mean straight-line distance: {mags.mean():.2f} mm, median: {np.median(mags):.2f} mm\n")

mean_vec = residual_vecs.mean(axis=0)
print(f"Mean residual VECTOR (dx,dy,dz) across all poses: {mean_vec} mm")
print(f"Magnitude of the mean vector: {np.linalg.norm(mean_vec):.2f} mm")
print("(if this is small relative to the ~15mm RMS, error is NOT one consistent direction)\n")

sum_of_mags = mags.sum()
mag_of_sum = np.linalg.norm(residual_vecs.sum(axis=0))
print(f"Sum of |residual| across poses: {sum_of_mags:.1f} mm")
print(f"Magnitude of summed vector: {mag_of_sum:.1f} mm")
print(f"Ratio (1.0 = all same direction, ~0 = random/cancelling directions): {mag_of_sum / sum_of_mags:.3f}\n")

print("Per-axis stats of residual vectors (mm):")
print(f"  dx: mean={residual_vecs[:,0].mean():+.2f} std={residual_vecs[:,0].std():.2f} min={residual_vecs[:,0].min():.2f} max={residual_vecs[:,0].max():.2f}")
print(f"  dy: mean={residual_vecs[:,1].mean():+.2f} std={residual_vecs[:,1].std():.2f} min={residual_vecs[:,1].min():.2f} max={residual_vecs[:,1].max():.2f}")
print(f"  dz: mean={residual_vecs[:,2].mean():+.2f} std={residual_vecs[:,2].std():.2f} min={residual_vecs[:,2].min():.2f} max={residual_vecs[:,2].max():.2f}")

# unit-direction spread: average pairwise dot product of normalized residual vectors
unit_vecs = residual_vecs / mags[:, None]
dots = unit_vecs @ unit_vecs.T
upper = dots[np.triu_indices(n, k=1)]
print(f"\nMean cosine similarity between all pairs of residual directions: {upper.mean():.3f}")
print("(1.0 = all point the same way, 0 = perpendicular/random, negative = opposing)")
