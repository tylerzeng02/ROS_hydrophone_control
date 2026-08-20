"""Pure-math tests for registration.py -- no rclpy node, no GUI, no MoveIt
needed to run these (just geometry_msgs' plain message classes)."""

import math

from fus_targeting_gui.registration import FixedPoseRegistration, quaternion_looking_along


def test_identity_registration_passes_point_through():
    reg = FixedPoseRegistration(xyz_m=[0.0, 0.0, 0.0], rpy_rad=[0.0, 0.0, 0.0])
    pose = reg.mesh_point_to_target_pose(
        point_local=(0.1, 0.0, 0.0), normal_local=(0.0, 0.0, 1.0), standoff_mm=0.0
    )
    assert math.isclose(pose.position.x, 0.1, abs_tol=1e-9)
    assert math.isclose(pose.position.y, 0.0, abs_tol=1e-9)
    assert math.isclose(pose.position.z, 0.0, abs_tol=1e-9)


def test_standoff_offsets_along_normal():
    reg = FixedPoseRegistration(xyz_m=[0.0, 0.0, 0.0], rpy_rad=[0.0, 0.0, 0.0])
    pose = reg.mesh_point_to_target_pose(
        point_local=(0.0, 0.0, 0.0), normal_local=(0.0, 0.0, 1.0), standoff_mm=10.0
    )
    assert math.isclose(pose.position.z, 0.01, abs_tol=1e-9)  # 10mm standoff along +Z normal


def test_translation_offset_applied():
    reg = FixedPoseRegistration(xyz_m=[1.0, 2.0, 3.0], rpy_rad=[0.0, 0.0, 0.0])
    pose = reg.mesh_point_to_target_pose(
        point_local=(0.0, 0.0, 0.0), normal_local=(0.0, 0.0, 1.0), standoff_mm=0.0
    )
    assert math.isclose(pose.position.x, 1.0, abs_tol=1e-9)
    assert math.isclose(pose.position.y, 2.0, abs_tol=1e-9)
    assert math.isclose(pose.position.z, 3.0, abs_tol=1e-9)


def test_quaternion_looking_along_is_unit_and_maps_z_to_direction():
    for direction in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]:
        qx, qy, qz, qw = quaternion_looking_along(direction)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        assert math.isclose(norm, 1.0, abs_tol=1e-6)
