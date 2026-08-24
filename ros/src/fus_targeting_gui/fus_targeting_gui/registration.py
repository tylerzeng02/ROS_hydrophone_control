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

from .geometry_utils import look_at_basis, tilt_direction


@dataclass
class TargetingConfig:
    standoff_mm: float
    planning_time_s: float
    num_planning_attempts: int
    default_search_area_spacing_mm: float = 5.0


def _euler_to_matrix(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return tf_transformations.euler_matrix(roll, pitch, yaw)


def rotation_matrix_to_rpy(R):
    """Inverse of _euler_to_matrix()'s rotation part -- given a 3x3
    rotation matrix (e.g. from point_registration.fit_rigid_transform()),
    returns (roll, pitch, yaw) radians in the exact same convention, so a
    fitted registration round-trips correctly into config.yaml's
    registration.rpy_rad field. Deliberately goes through
    tf_transformations.euler_from_matrix() (the real inverse of
    euler_matrix()) rather than a hand-derived formula -- guarantees
    convention consistency instead of risking a subtly-wrong axis order."""
    matrix4 = np.eye(4)
    matrix4[0:3, 0:3] = R
    roll, pitch, yaw = tf_transformations.euler_from_matrix(matrix4)
    return roll, pitch, yaw


def quaternion_looking_along(direction, up_hint=(0.0, 0.0, 1.0), roll_deg=0.0):
    """Quaternion (x, y, z, w) whose local +Z axis points along `direction`.
    `up_hint` only resolves the otherwise-free rotation about that axis;
    `roll_deg` is the knob that actually controls hydrophone roll, rotating
    further about the approach axis on top of whatever `up_hint` picked --
    up_hint's own zero point isn't physically meaningful, just consistent.

    IMPORTANT: which local axis of `end_effector_frame` should point along
    the approach direction is a property of how the probe is physically
    mounted, and could not be verified from this repo alone. +Z is the
    assumption below -- confirm against the real mount before trusting a
    computed orientation on real hardware.
    """
    x, y, z = look_at_basis(direction, up_hint=up_hint, roll_deg=roll_deg)
    rot = np.eye(4)
    rot[0:3, 0] = x
    rot[0:3, 1] = y
    rot[0:3, 2] = z
    return tf_transformations.quaternion_from_matrix(rot)


class Registration(ABC):
    @abstractmethod
    def mesh_point_to_target_pose(
        self, point_local, normal_local, standoff_mm,
        tilt_deg: float = 0.0, azimuth_deg: float = 0.0, roll_deg: float = 0.0,
    ):
        """point_local/normal_local: (x, y, z) in the mesh's own (local,
        pre-registration) coordinate frame, as returned by mesh_view's
        picker. Returns a geometry_msgs/Pose in the robot's base_frame.

        The approach axis starts at the outward surface normal and can be
        tilted away from it by `tilt_deg` (0 = straight-on, toward 90 =
        grazing), in the tangent-plane direction selected by `azimuth_deg`
        (0-360, rotating around the normal). `roll_deg` additionally
        rotates the probe about its own final approach axis. The probe
        stops `standoff_mm` short of the picked point, measured back along
        that same (possibly tilted) approach axis -- not simply along the
        surface normal -- so it always ends up exactly standoff_mm from the
        target point regardless of tilt/azimuth.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_points_to_base_frame(self, points_local):
        """Batch version of the position half of mesh_point_to_target_pose()
        -- transforms an (N, 3) array of mesh-local points into base_frame,
        with no standoff/tilt/orientation math. Used to move a whole mesh
        (e.g. for a MoveIt collision object, see
        MoveItBridge.set_skull_collision_object()) into base_frame at once,
        rather than one point at a time."""
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

    def transform_points_to_base_frame(self, points_local):
        points = np.asarray(points_local, dtype=float)
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        return (self._matrix @ homogeneous.T).T[:, 0:3]

    def mesh_point_to_target_pose(
        self, point_local, normal_local, standoff_mm,
        tilt_deg: float = 0.0, azimuth_deg: float = 0.0, roll_deg: float = 0.0,
    ):
        point_h = np.array([*point_local, 1.0])
        point_robot = (self._matrix @ point_h)[0:3]

        normal_robot = self._matrix[0:3, 0:3] @ np.array(normal_local, dtype=float)
        normal_robot /= np.linalg.norm(normal_robot)

        # Tilt is applied here in the robot frame (post-registration).
        # mesh_view's live preview applies the same tilt/azimuth in the
        # mesh's local frame instead, for simplicity, which can pick a
        # differently-oriented (but equally arbitrary) azimuth=0 reference
        # -- harmless, since azimuth's zero point isn't physically
        # meaningful either way, but means the preview's exact rotation
        # won't always match this actual computed pose bit-for-bit.
        approach_dir = tilt_direction(-normal_robot, tilt_deg, azimuth_deg)

        standoff_m = standoff_mm / 1000.0
        target_position = point_robot - approach_dir * standoff_m

        quat = quaternion_looking_along(approach_dir, roll_deg=roll_deg)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target_position.tolist()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose
