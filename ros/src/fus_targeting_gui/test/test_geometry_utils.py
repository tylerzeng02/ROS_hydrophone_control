"""Pure-math tests for geometry_utils.py -- no ROS/Qt/pyvista needed."""

import math

import numpy as np

from fus_targeting_gui.geometry_utils import look_at_basis, orthonormal_basis, tilt_direction


def test_orthonormal_basis_is_right_handed_and_unit():
    for axis in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.3, 0.7, 0.2)]:
        u, v = orthonormal_basis(axis)
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        assert math.isclose(np.linalg.norm(u), 1.0, abs_tol=1e-9)
        assert math.isclose(np.linalg.norm(v), 1.0, abs_tol=1e-9)
        assert math.isclose(np.dot(u, a), 0.0, abs_tol=1e-9)
        assert math.isclose(np.dot(v, a), 0.0, abs_tol=1e-9)
        assert math.isclose(np.dot(u, v), 0.0, abs_tol=1e-9)
        assert np.allclose(np.cross(u, v), a, atol=1e-9)


def test_tilt_zero_returns_base_direction_regardless_of_azimuth():
    base = (0.0, 0.0, 1.0)
    for az in [0.0, 90.0, 217.0, 359.0]:
        result = tilt_direction(base, 0.0, az)
        assert np.allclose(result, (0.0, 0.0, 1.0), atol=1e-9)


def test_tilt_90_is_perpendicular_to_base():
    base = (0.0, 0.0, 1.0)
    result = tilt_direction(base, 90.0, 30.0)
    assert math.isclose(np.dot(result, base), 0.0, abs_tol=1e-9)
    assert math.isclose(np.linalg.norm(result), 1.0, abs_tol=1e-9)


def test_tilt_direction_always_unit_length_and_correct_angle():
    base = (1.0, 2.0, 3.0)  # deliberately non-unit input
    base_unit = np.asarray(base) / np.linalg.norm(base)
    for tilt in [10.0, 45.0, 89.0]:
        result = tilt_direction(base, tilt, 123.0)
        assert math.isclose(np.linalg.norm(result), 1.0, abs_tol=1e-9)
        cos_angle = np.dot(result, base_unit)
        assert math.isclose(cos_angle, math.cos(math.radians(tilt)), abs_tol=1e-9)


def test_azimuth_rotates_around_base_axis():
    base = (0.0, 0.0, 1.0)
    d0 = tilt_direction(base, 45.0, 0.0)
    d90 = tilt_direction(base, 45.0, 90.0)
    d180 = tilt_direction(base, 45.0, 180.0)
    # Same tilt angle off the axis at every azimuth...
    for d in (d0, d90, d180):
        assert math.isclose(np.dot(d, base), math.cos(math.radians(45.0)), abs_tol=1e-9)
    # ...but genuinely different directions, and opposite azimuths land on
    # opposite sides of the base axis.
    assert not np.allclose(d0, d90, atol=1e-6)
    assert np.allclose(d0 - np.dot(d0, base) * np.asarray(base), -(d180 - np.dot(d180, base) * np.asarray(base)), atol=1e-9)


def test_look_at_basis_is_orthonormal_and_z_matches_direction():
    for direction in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 1.0, 1.0)]:
        x, y, z = look_at_basis(direction)
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        assert np.allclose(z, d, atol=1e-9)
        assert math.isclose(np.linalg.norm(x), 1.0, abs_tol=1e-9)
        assert math.isclose(np.linalg.norm(y), 1.0, abs_tol=1e-9)
        assert math.isclose(np.dot(x, y), 0.0, abs_tol=1e-9)
        assert math.isclose(np.dot(x, z), 0.0, abs_tol=1e-9)
        assert np.allclose(np.cross(x, y), z, atol=1e-9)


def test_look_at_basis_roll_preserves_z_and_rotates_x_y():
    direction = (0.0, 0.0, 1.0)
    x0, y0, z0 = look_at_basis(direction, roll_deg=0.0)
    x90, y90, z90 = look_at_basis(direction, roll_deg=90.0)
    assert np.allclose(z0, z90, atol=1e-9)  # roll never touches the approach axis
    assert np.allclose(x90, y0, atol=1e-6)  # +90 deg roll: x -> old y
    assert np.allclose(y90, -x0, atol=1e-6)


def test_look_at_basis_roll_360_is_identity():
    direction = (0.3, -0.5, 0.8)
    x0, y0, z0 = look_at_basis(direction, roll_deg=0.0)
    x360, y360, z360 = look_at_basis(direction, roll_deg=360.0)
    assert np.allclose(x0, x360, atol=1e-6)
    assert np.allclose(y0, y360, atol=1e-6)
    assert np.allclose(z0, z360, atol=1e-6)
