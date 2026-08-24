"""Tests for predefined_points.py -- pure csv/dataclasses, no ROS/Qt needed."""

import math
import tempfile
from pathlib import Path

from fus_targeting_gui.predefined_points import load_predefined_points_csv


def _write_csv(tmp_path, text):
    path = tmp_path / "points.csv"
    path.write_text(text)
    return str(path)


def test_loads_points_and_applies_scale(tmp_path):
    csv_text = "label,x,y,z\ntarget_A,10.0,20.0,30.0\ntarget_B,-5.0,0.0,15.0\n"
    path = _write_csv(tmp_path, csv_text)
    points = load_predefined_points_csv(path, scale=0.001)

    assert len(points) == 2
    assert points[0].label == "target_A"
    assert math.isclose(points[0].point_local[0], 0.010, abs_tol=1e-9)
    assert math.isclose(points[0].point_local[1], 0.020, abs_tol=1e-9)
    assert math.isclose(points[0].point_local[2], 0.030, abs_tol=1e-9)
    assert points[0].normal_local is None


def test_optional_normal_columns_are_parsed_when_present(tmp_path):
    csv_text = (
        "label,x,y,z,normal_x,normal_y,normal_z\n"
        "with_normal,0,0,0,0.0,0.0,1.0\n"
        "without_normal,1,1,1,,,\n"
    )
    path = _write_csv(tmp_path, csv_text)
    points = load_predefined_points_csv(path, scale=1.0)

    assert points[0].normal_local == (0.0, 0.0, 1.0)
    assert points[1].normal_local is None


def test_missing_label_defaults_to_empty_string(tmp_path):
    csv_text = "label,x,y,z\n,1.0,2.0,3.0\n"
    path = _write_csv(tmp_path, csv_text)
    points = load_predefined_points_csv(path, scale=1.0)
    assert points[0].label == ""


def test_scale_of_one_leaves_coordinates_unchanged(tmp_path):
    csv_text = "label,x,y,z\np,1.5,2.5,3.5\n"
    path = _write_csv(tmp_path, csv_text)
    points = load_predefined_points_csv(path, scale=1.0)
    assert points[0].point_local == (1.5, 2.5, 3.5)
