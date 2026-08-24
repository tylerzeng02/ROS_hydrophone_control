"""Loads config.yaml into small, typed structures. This is the one file
every other module reads robot/mesh/registration/targeting settings
through -- nothing else in this package should read config.yaml directly
or hardcode a frame/group name."""

from dataclasses import dataclass

import yaml

from .registration import FixedPoseRegistration, TargetingConfig


@dataclass
class RobotConfig:
    planning_group: str
    base_frame: str
    end_effector_frame: str
    joint_names: list


@dataclass
class MeshConfig:
    default_path: str
    scale: float
    predefined_points_path: str = ""  # empty = none configured


@dataclass
class AppConfig:
    robot: RobotConfig
    mesh: MeshConfig
    targeting: TargetingConfig
    registration: FixedPoseRegistration


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    robot = RobotConfig(
        planning_group=raw["robot"]["planning_group"],
        base_frame=raw["robot"]["base_frame"],
        end_effector_frame=raw["robot"]["end_effector_frame"],
        joint_names=list(raw["robot"]["joint_names"]),
    )
    mesh = MeshConfig(
        default_path=raw["mesh"]["default_path"],
        scale=float(raw["mesh"]["scale"]),
        predefined_points_path=raw["mesh"].get("predefined_points_path") or "",
    )
    targeting = TargetingConfig(
        standoff_mm=float(raw["targeting"]["standoff_mm"]),
        planning_time_s=float(raw["targeting"]["planning_time_s"]),
        num_planning_attempts=int(raw["targeting"]["num_planning_attempts"]),
        default_search_area_spacing_mm=float(
            raw["targeting"].get("default_search_area_spacing_mm", 5.0)
        ),
    )
    registration = FixedPoseRegistration(
        xyz_m=raw["registration"]["xyz_m"],
        rpy_rad=raw["registration"]["rpy_rad"],
    )

    return AppConfig(robot=robot, mesh=mesh, targeting=targeting, registration=registration)
