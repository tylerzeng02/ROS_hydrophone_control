"""Converts a picked mesh-local surface point and normal into a target
geometry_msgs/Pose in the robot's planning frame.

This is deliberately isolated behind the Registration base class so a
future NDI-fiducial-based, or any other, registration scheme can replace
FixedPoseRegistration without touching mesh_view.py or moveit_bridge.py.
Neither of those files imports anything from this module except a
Registration instance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import tf_transformations
from geometry_msgs.msg import Pose


@dataclass
class TargetingConfig:
    standoff_mm: float
    planning_time_s: float
    num_planning_attempts: int


def _euler_to_matrix(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return tf_transformations.euler_matrix(roll, pitch, yaw)


def quaternion_looking_along(direction, up_hint=(0.0, 0.0, 1.0)):
    """Computes the rotation that maps the end effector's local +Z axis
    onto `direction`, with both vectors expressed in the same frame. The
    remaining rotation about the approach axis is resolved using `up_hint`,
    a look-at style construction. This does not claim to be the physically
    correct tool roll.

    Important: which local axis of `end_effector_frame` should point along
    the approach direction is a property of how the probe is physically
    mounted, and this could not be verified from the repository alone.
    Only the arm's own virtual_endeffector frame is defined here, not the
    probe tool's own frame convention. The +Z assumption below should be
    confirmed against the real mount before trusting a computed
    orientation, for example by checking a planned or executed pose in
    RViz against the physical probe.
    """
    z = np.array(direction, dtype=float)
    norm = np.linalg.norm(z)
    if norm < 1e-9:
        raise ValueError("Zero-length approach direction.")
    z /= norm

    up = np.array(up_hint, dtype=float)
    if abs(np.dot(z, up)) > 0.999:
        # The direction is parallel or antiparallel to the up hint.
        # Pick a different reference so cross() below does not degenerate.
        up = np.array([1.0, 0.0, 0.0])

    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    rot = np.eye(4)
    rot[0:3, 0] = x
    rot[0:3, 1] = y
    rot[0:3, 2] = z
    return tf_transformations.quaternion_from_matrix(rot)


class Registration(ABC):
    @abstractmethod
    def mesh_point_to_target_pose(self, point_local, normal_local, standoff_mm):
        """point_local and normal_local are (x, y, z) tuples in the mesh's
        own local, pre-registration coordinate frame, as returned by
        mesh_view's picker. Returns a geometry_msgs/Pose in the robot's
        base_frame, offset standoff_mm back along the transformed outward
        normal from the picked surface point."""
        raise NotImplementedError


class FixedPoseRegistration(Registration):
    """Registers the mesh to base_frame using one fixed, hand-specified
    pose. This is exactly the transform publish_skull_marker.py already
    uses to place this same mesh in RViz. Keep this and that script's
    xyz and rpy arguments in sync if the mesh is ever re-registered
    against the real setup.
    """

    def __init__(self, xyz_m, rpy_rad):
        self._matrix = _euler_to_matrix(rpy_rad)
        self._matrix[0:3, 3] = xyz_m

    def mesh_point_to_target_pose(self, point_local, normal_local, standoff_mm):
        point_h = np.array([*point_local, 1.0])
        point_robot = (self._matrix @ point_h)[0:3]

        normal_robot = self._matrix[0:3, 0:3] @ np.array(normal_local, dtype=float)
        normal_robot /= np.linalg.norm(normal_robot)

        standoff_m = standoff_mm / 1000.0
        approach_dir = -normal_robot  # travel INTO the surface
        target_position = point_robot + normal_robot * standoff_m

        quat = quaternion_looking_along(approach_dir)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target_position.tolist()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose
