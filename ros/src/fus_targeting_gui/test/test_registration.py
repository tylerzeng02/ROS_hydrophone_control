"""Pure-math tests for registration.py. Running these needs no rclpy
node, no GUI, and no MoveIt, only geometry_msgs' plain message classes."""

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


def test_zero_tilt_reproduces_straight_on_normal_behavior():
    reg = FixedPoseRegistration(xyz_m=[0.0, 0.0, 0.0], rpy_rad=[0.0, 0.0, 0.0])
    pose = reg.mesh_point_to_target_pose(
        point_local=(0.1, 0.0, 0.0), normal_local=(0.0, 0.0, 1.0), standoff_mm=10.0,
        tilt_deg=0.0, azimuth_deg=123.0, roll_deg=0.0,  # azimuth/roll must be no-ops at tilt=0
    )
    assert math.isclose(pose.position.x, 0.1, abs_tol=1e-9)
    assert math.isclose(pose.position.y, 0.0, abs_tol=1e-9)
    assert math.isclose(pose.position.z, 0.01, abs_tol=1e-9)


def test_standoff_distance_from_target_is_exact_at_any_tilt():
    reg = FixedPoseRegistration(xyz_m=[0.0, 0.0, 0.0], rpy_rad=[0.0, 0.0, 0.0])
    point_local = (0.02, -0.01, 0.05)
    for tilt_deg in (0.0, 15.0, 45.0, 80.0):
        for azimuth_deg in (0.0, 90.0, 200.0):
            pose = reg.mesh_point_to_target_pose(
                point_local=point_local, normal_local=(0.0, 0.0, 1.0), standoff_mm=12.0,
                tilt_deg=tilt_deg, azimuth_deg=azimuth_deg, roll_deg=0.0,
            )
            target = (pose.position.x, pose.position.y, pose.position.z)
            distance_mm = math.dist(target, point_local) * 1000.0
            assert math.isclose(distance_mm, 12.0, abs_tol=1e-6), (tilt_deg, azimuth_deg, distance_mm)


def test_tilt_and_roll_change_orientation_but_not_default_alignment():
    reg = FixedPoseRegistration(xyz_m=[0.0, 0.0, 0.0], rpy_rad=[0.0, 0.0, 0.0])
    straight_on = reg.mesh_point_to_target_pose(
        point_local=(0.0, 0.0, 0.0), normal_local=(0.0, 0.0, 1.0), standoff_mm=10.0,
    )
    tilted = reg.mesh_point_to_target_pose(
        point_local=(0.0, 0.0, 0.0), normal_local=(0.0, 0.0, 1.0), standoff_mm=10.0,
        tilt_deg=30.0, azimuth_deg=0.0,
    )
    rolled = reg.mesh_point_to_target_pose(
        point_local=(0.0, 0.0, 0.0), normal_local=(0.0, 0.0, 1.0), standoff_mm=10.0,
        roll_deg=45.0,
    )
    assert not math.isclose(straight_on.orientation.x, tilted.orientation.x, abs_tol=1e-6) \
        or not math.isclose(straight_on.orientation.y, tilted.orientation.y, abs_tol=1e-6)
    assert not math.isclose(straight_on.position.x, tilted.position.x, abs_tol=1e-9) \
        or not math.isclose(straight_on.position.y, tilted.position.y, abs_tol=1e-9)
    assert straight_on.orientation != rolled.orientation
