"""Loads config.yaml into small, typed structures. This is the one file
every other module reads robot, mesh, registration, and targeting
settings through. Nothing else in this package should read config.yaml
directly or hardcode a frame or group name."""

import os
from dataclasses import dataclass

import yaml

from .registration import FixedPoseRegistration, TargetingConfig


def _resolve_path(path: str, config_dir: str) -> str:
    """A relative mesh.default_path/predefined_points_path is resolved
    against the config file's own directory, not the process's current
    working directory. This is what makes it possible to ship a real
    default mesh with this package (meshes/, installed into the same
    share/fus_targeting_gui/ directory as config/) instead of requiring
    every machine to have some absolute, machine-specific mesh path
    (e.g. /home/<user>/...) configured before the GUI can even start.
    An absolute path is returned unchanged."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(config_dir, path)


@dataclass
class RobotConfig:
    planning_group: str
    base_frame: str
    end_effector_frame: str
    joint_names: list
    controller_action_name: str = "arm_controller/follow_joint_trajectory"
    # Joint-space "reset" configuration, in the same order as joint_names.
    # Defaults to all-zero (this robot's own SRDF "home" group_state and
    # ros2_control initial_positions.yaml both use all-zero) if omitted.
    home_joint_positions: list = None


@dataclass
class MeshConfig:
    default_path: str
    scale: float
    predefined_points_path: str = ""  # empty = none configured
    # Max triangle count for the published MoveIt collision object. 0 or
    # None publishes the full, undecimated mesh. FCL collision-checks this
    # geometry on every planner sample, so a very high triangle count can
    # slow planning down; lower this (e.g. 2000) if that becomes a problem.
    collision_max_triangles: int = 0


@dataclass
class AppConfig:
    robot: RobotConfig
    mesh: MeshConfig
    targeting: TargetingConfig
    registration: FixedPoseRegistration


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(path))

    joint_names = list(raw["robot"]["joint_names"])
    home_joint_positions = raw["robot"].get("home_joint_positions")
    robot = RobotConfig(
        planning_group=raw["robot"]["planning_group"],
        base_frame=raw["robot"]["base_frame"],
        end_effector_frame=raw["robot"]["end_effector_frame"],
        joint_names=joint_names,
        controller_action_name=raw["robot"].get(
            "controller_action_name", "arm_controller/follow_joint_trajectory"
        ),
        home_joint_positions=(
            [float(v) for v in home_joint_positions] if home_joint_positions
            else [0.0] * len(joint_names)
        ),
    )
    mesh = MeshConfig(
        default_path=_resolve_path(raw["mesh"]["default_path"], config_dir),
        scale=float(raw["mesh"]["scale"]),
        predefined_points_path=_resolve_path(
            raw["mesh"].get("predefined_points_path") or "", config_dir
        ),
        collision_max_triangles=int(raw["mesh"].get("collision_max_triangles", 0) or 0),
    )
    targeting = TargetingConfig(
        standoff_mm=float(raw["targeting"]["standoff_mm"]),
        planning_time_s=float(raw["targeting"]["planning_time_s"]),
        num_planning_attempts=int(raw["targeting"]["num_planning_attempts"]),
        default_search_area_spacing_mm=float(
            raw["targeting"].get("default_search_area_spacing_mm", 5.0)
        ),
        enforce_orientation=bool(raw["targeting"].get("enforce_orientation", False)),
    )
    registration = FixedPoseRegistration(
        xyz_m=raw["registration"]["xyz_m"],
        rpy_rad=raw["registration"]["rpy_rad"],
    )

    return AppConfig(robot=robot, mesh=mesh, targeting=targeting, registration=registration)
