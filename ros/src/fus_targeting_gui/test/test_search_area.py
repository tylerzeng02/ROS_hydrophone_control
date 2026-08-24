"""Pure-math tests for search_area.py -- no ROS/Qt/pyvista needed."""

import math

import numpy as np
import pytest

from fus_targeting_gui.search_area import _point_in_polygon_2d, generate_scan_grid


def test_point_in_polygon_basic_square():
    square = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    assert _point_in_polygon_2d((0, 0), square) is True
    assert _point_in_polygon_2d((2, 2), square) is False
    assert _point_in_polygon_2d((0.99, 0.99), square) is True


def test_needs_at_least_3_boundary_points():
    with pytest.raises(ValueError):
        generate_scan_grid([(0, 0, 0), (1, 0, 0)], [(0, 0, 1), (0, 0, 1)], spacing_mm=5.0)


def test_rejects_non_positive_spacing():
    boundary = [(-0.01, -0.01, 0.0), (0.01, -0.01, 0.0), (0.01, 0.01, 0.0), (-0.01, 0.01, 0.0)]
    normals = [(0.0, 0.0, 1.0)] * 4
    with pytest.raises(ValueError):
        generate_scan_grid(boundary, normals, spacing_mm=0.0)
    with pytest.raises(ValueError):
        generate_scan_grid(boundary, normals, spacing_mm=-1.0)


def test_grid_stays_within_flat_square_boundary_and_on_plane():
    # 20mm x 20mm square in the XY plane (normal +Z), 5mm spacing.
    boundary = [(-0.010, -0.010, 0.0), (0.010, -0.010, 0.0), (0.010, 0.010, 0.0), (-0.010, 0.010, 0.0)]
    normals = [(0.0, 0.0, 1.0)] * 4
    grid, plane_normal = generate_scan_grid(boundary, normals, spacing_mm=5.0)

    assert math.isclose(plane_normal[2], 1.0, abs_tol=1e-9)
    assert len(grid) > 0
    for p in grid:
        assert -0.0101 <= p[0] <= 0.0101
        assert -0.0101 <= p[1] <= 0.0101
        assert abs(p[2]) < 1e-9


def test_smaller_spacing_yields_more_points():
    boundary = [(-0.010, -0.010, 0.0), (0.010, -0.010, 0.0), (0.010, 0.010, 0.0), (-0.010, 0.010, 0.0)]
    normals = [(0.0, 0.0, 1.0)] * 4
    coarse, _ = generate_scan_grid(boundary, normals, spacing_mm=10.0)
    fine, _ = generate_scan_grid(boundary, normals, spacing_mm=2.0)
    assert len(fine) > len(coarse)


def test_cancelling_normals_raise_instead_of_producing_a_bogus_plane():
    boundary = [(-0.01, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)]
    normals = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)]  # mean == (0,0,0)
    with pytest.raises(ValueError):
        generate_scan_grid(boundary, normals, spacing_mm=5.0)
