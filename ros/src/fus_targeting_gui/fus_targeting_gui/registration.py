"""Converts a picked mesh-local surface point (+ normal) into a target
geometry_msgs/Pose in the robot's planning frame.

Deliberately isolated behind the Registration base class so a future
NDI-fiducial-based (or any other) registration scheme can replace
FixedPoseRegistration without touching mesh_view.py or moveit_bridge.py --
neither of those imports anything from this module except a Registration
instance.
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
    """Rotation that maps the end effector's local +Z axis onto `direction`
    (both in the same frame), with the remaining rotation-about-the-approach
    -axis degree of freedom resolved via `up_hint` (a "look-at" style
    construction, not a claim about the physically-correct tool roll).

    IMPORTANT: which local axis of `end_effector_frame` should point along
    the approach direction is a property of how the probe is physically
    mounted, and could not be verified from this repo alone (only the
    arm's own virtual_endeffector frame is defined here, not the probe
    tool's own frame convention). +Z is the assumption below -- confirm
    against the real mount before trusting a computed orientation (e.g. by
    checking a planned/executed pose in RViz against the physical probe).
    """
    z = np.array(direction, dtype=float)
    norm = np.linalg.norm(z)
    if norm < 1e-9:
        raise ValueError("Zero-length approach direction.")
    z /= norm

    up = np.array(up_hint, dtype=float)
    if abs(np.dot(z, up)) > 0.999:
        # direction is (anti)parallel to the up hint -- pick a different
        # reference so cross() below doesn't degenerate.
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
        """point_local/normal_local: (x, y, z) in the mesh's own (local,
        pre-registration) coordinate frame, as returned by mesh_view's
        picker. Returns a geometry_msgs/Pose in the robot's base_frame,
        offset `standoff_mm` back along the (transformed) outward normal
        from the picked surface point."""
        raise NotImplementedError


class FixedPoseRegistration(Registration):
    """Registers the mesh to base_frame via one fixed, hand-specified pose
    -- exactly the transform publish_skull_marker.py already uses to place
    this same mesh in RViz. Keep this and that script's --xyz/--rpy in
    sync if the mesh is ever re-registered against the real setup.
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
