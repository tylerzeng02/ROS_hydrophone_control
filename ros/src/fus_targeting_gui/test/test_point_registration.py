"""Tests for point_registration.py -- pure numpy, no ROS/Qt needed."""

import math

import numpy as np
import pytest

from fus_targeting_gui.point_registration import fit_rigid_transform


def _rotation_matrix_z(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_needs_at_least_3_point_pairs():
    with pytest.raises(ValueError):
        fit_rigid_transform([(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (1, 0, 0)])


def test_mismatched_point_counts_raise():
    with pytest.raises(ValueError):
        fit_rigid_transform([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 0), (1, 0, 0)])


def test_identity_transform_recovered_exactly():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, size=(5, 3))
    R, t, rmse = fit_rigid_transform(points, points)
    assert np.allclose(R, np.eye(3), atol=1e-9)
    assert np.allclose(t, np.zeros(3), atol=1e-9)
    assert rmse < 1e-9


def test_known_rotation_and_translation_recovered():
    rng = np.random.default_rng(1)
    source = rng.uniform(-0.1, 0.1, size=(8, 3))  # meters-scale, like real mesh points
    true_R = _rotation_matrix_z(math.radians(37.0))
    true_t = np.array([0.5, -0.2, 0.1])
    target = (true_R @ source.T).T + true_t

    fitted_R, fitted_t, rmse = fit_rigid_transform(source, target)

    assert np.allclose(fitted_R, true_R, atol=1e-9)
    assert np.allclose(fitted_t, true_t, atol=1e-9)
    assert rmse < 1e-9


def test_fitted_rotation_is_a_proper_rotation_not_a_reflection():
    rng = np.random.default_rng(2)
    source = rng.uniform(-1, 1, size=(6, 3))
    true_R = _rotation_matrix_z(math.radians(-113.0))
    target = (true_R @ source.T).T + np.array([1.0, 2.0, 3.0])

    R, _t, _rmse = fit_rigid_transform(source, target)
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_noisy_points_give_nonzero_but_small_rmse():
    rng = np.random.default_rng(3)
    source = rng.uniform(-0.1, 0.1, size=(10, 3))
    true_R = _rotation_matrix_z(math.radians(20.0))
    true_t = np.array([0.05, 0.0, -0.03])
    noise = rng.normal(scale=0.0005, size=source.shape)  # 0.5mm-scale noise
    target = (true_R @ source.T).T + true_t + noise

    _R, _t, rmse = fit_rigid_transform(source, target)
    assert 0.0 < rmse < 0.002  # should stay in the same ballpark as the injected noise


def test_inconsistent_pairs_give_large_rmse():
    # Deliberately mismatched pairs (not a real rigid transform relating
    # them) should NOT silently produce a near-zero rmse.
    source = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    target = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (5.0, 5.0, 5.0)]
    _R, _t, rmse = fit_rigid_transform(source, target)
    assert rmse > 0.5
