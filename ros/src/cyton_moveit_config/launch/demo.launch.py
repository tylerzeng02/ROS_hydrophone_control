"""Full MoveIt demo: robot_state_publisher + ros2_control_node + controller
spawners + move_group + RViz (MotionPlanning plugin). Modeled directly on
moveit_resources_panda_moveit_config's own demo.launch.py (installed at
/opt/ros/jazzy/share/moveit_resources_panda_moveit_config/launch/
demo.launch.py), trimmed to this robot's single "arm" group (no gripper
group -- motor 7 is excluded from the URDF's IK chain) and one controller.

Usage:
    ros2 launch cyton_moveit_config demo.launch.py
    ros2 launch cyton_moveit_config demo.launch.py hardware_type:=real serial_port:=/dev/ttyUSB0

hardware_type defaults to "mock_components" (ros2_control's built-in
mock_components/GenericSystem -- no real servo involved, safe for
exercising the whole pipeline). Only pass hardware_type:=real with the
physical arm connected, powered, and clear to move.

urdf_variant (added 2026-08-12) defaults to "calibrated" (the real,
deployed cyton_gamma_1500.urdf.xacro). Pass urdf_variant:=uncalibrated to
instead load cyton_gamma_1500_uncalibrated.urdf.xacro -- same robot, same
ros2_control/hardware plugin, but MoveIt plans against the ORIGINAL,
uncorrected joint geometry (see that file's own header) instead of this
project's fitted kinematic calibration. Built specifically to A/B compare
real-world positioning accuracy with vs. without the geometry correction,
using the exact same accuracy-check tools either way. Does NOT affect
robot_calibration.cpp's separate real-servo tick<->radian calibration --
only what MoveIt's planner believes the robot's geometry is.

urdf_variant:=sim_7dof (added 2026-08-24) loads
cyton_gamma_1500_sim7dof.urdf.xacro -- identical to the calibrated variant
except elbow_yaw_joint's <limit> is widened back to its full mechanical
range instead of the ~4-degree production lock, restoring genuine 7-DOF
motion for planning/simulation. ONLY valid with hardware_type:=mock_components
-- combining it with hardware_type:=real raises an error (see
launch_setup() below and cyton_gamma_1500_robot_sim7dof.xacro's own header
for why).

compensate_backlash (added 2026-08-13) defaults to "false". Pass
compensate_backlash:=true (only meaningful with hardware_type:=real) to
enable cyton_hardware's new streaming-compatible backlash compensator --
see CytonSystemHardware's own header comment for exactly what this does
and its (not yet real-hardware-validated) status. Unrelated to
dynamixel_motor.cpp's separate, already-validated blocking-move fix, which
this pipeline never uses.

Implementation note: which URDF file to load has to be resolved to a plain
Python string BEFORE MoveItConfigsBuilder runs (moveit_configs_utils joins
file_path with a pathlib.Path internally, which can't accept a launch
Substitution object) -- so, unlike every other argument here, urdf_variant
can't just be threaded through as a LaunchConfiguration passed to
.robot_description(). Everything that depends on it is built inside an
OpaqueFunction instead, which runs at launch time with a real LaunchContext
that LaunchConfiguration.perform(context) can resolve to an actual string.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    urdf_variant = LaunchConfiguration("urdf_variant").perform(context)
    if urdf_variant not in ("calibrated", "uncalibrated", "sim_7dof"):
        raise ValueError(
            "urdf_variant must be 'calibrated', 'uncalibrated', or 'sim_7dof', "
            f"got '{urdf_variant}'"
        )
    hardware_type = LaunchConfiguration("hardware_type").perform(context)
    if urdf_variant == "sim_7dof" and hardware_type == "real":
        # sim_7dof widens elbow_yaw_joint's <limit> back to the full
        # mechanical range instead of the production ~4-degree lock -- see
        # cyton_gamma_1500_robot_sim7dof.xacro's own header for why that's
        # only safe on mock_components (which never invokes
        # robot_calibration.cpp's hardware-level safety clamp at all).
        # Real hardware must never plan against this widened range.
        raise ValueError(
            "urdf_variant=sim_7dof cannot be combined with hardware_type=real -- "
            "that widened elbow_yaw range is only safe in simulation. See "
            "cyton_gamma_1500_robot_sim7dof.xacro's header comment."
        )
    urdf_filename = {
        "uncalibrated": "cyton_gamma_1500_uncalibrated.urdf.xacro",
        "sim_7dof": "cyton_gamma_1500_sim7dof.urdf.xacro",
    }.get(urdf_variant, "cyton_gamma_1500.urdf.xacro")

    ik_solver = LaunchConfiguration("ik_solver").perform(context)
    if ik_solver not in ("kdl", "trac_ik"):
        raise ValueError(f"ik_solver must be 'kdl' or 'trac_ik', got '{ik_solver}'")
    kinematics_filename = "config/kinematics_trac_ik.yaml" if ik_solver == "trac_ik" else "config/kinematics.yaml"

    moveit_config = (
        MoveItConfigsBuilder("cyton_gamma_1500", package_name="cyton_moveit_config")
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory("cyton_description"),
                "urdf",
                urdf_filename,
            ),
            mappings={
                "hardware_type": LaunchConfiguration("hardware_type"),
                "serial_port": LaunchConfiguration("serial_port"),
                "baud_rate": LaunchConfiguration("baud_rate"),
                "compensate_backlash": LaunchConfiguration("compensate_backlash"),
            },
        )
        .robot_description_semantic(file_path="config/cyton_gamma_1500.srdf")
        .robot_description_kinematics(file_path=kinematics_filename)
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("cyton_moveit_config"), "launch", LaunchConfiguration("rviz_config")]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_footprint"],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("cyton_moveit_config"), "config", "ros2_controllers.yaml"
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description, ros2_controllers_path],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "-c", "/controller_manager"],
    )

    return [
        static_tf_node,
        robot_state_publisher_node,
        move_group_node,
        rviz_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
    ]


def generate_launch_description():
    hardware_type_arg = DeclareLaunchArgument(
        "hardware_type",
        default_value="mock_components",
        description="ros2_control hardware plugin to load: mock_components (safe, no real "
        "hardware) or real (cyton_hardware/CytonSystemHardware, talks to the physical arm)",
    )
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyUSB0",
        description="Serial device for the Dynamixel bus (only used when hardware_type:=real)",
    )
    baud_rate_arg = DeclareLaunchArgument(
        "baud_rate", default_value="1000000", description="Dynamixel bus baud rate"
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config", default_value="moveit.rviz", description="RViz configuration file"
    )
    urdf_variant_arg = DeclareLaunchArgument(
        "urdf_variant",
        default_value="calibrated",
        description="Which top-level URDF xacro MoveIt plans against: 'calibrated' (real, "
        "deployed kinematic corrections -- default), 'uncalibrated' (original, uncorrected "
        "joint geometry -- see cyton_gamma_1500_robot_uncalibrated.xacro's header), or "
        "'sim_7dof' (calibrated geometry but elbow_yaw_joint's full mechanical range instead "
        "of the production lock -- mock_components only, rejected with hardware_type=real). "
        "Does not affect robot_calibration.cpp's real-servo calibration either way.",
    )
    compensate_backlash_arg = DeclareLaunchArgument(
        "compensate_backlash",
        default_value="false",
        description="Enable cyton_hardware's streaming backlash compensator (only meaningful "
        "with hardware_type:=real). Default false. Not yet validated against real hardware. "
        "See CytonSystemHardware's own header comment.",
    )
    ik_solver_arg = DeclareLaunchArgument(
        "ik_solver",
        default_value="kdl",
        description="'kdl' (default, production-proven) or 'trac_ik' (cyton_trac_ik_kinematics_"
        "plugin, new and not yet hardware-validated, see that plugin's own header comment). "
        "Selects config/kinematics.yaml versus config/kinematics_trac_ik.yaml.",
    )

    return LaunchDescription(
        [
            hardware_type_arg,
            serial_port_arg,
            baud_rate_arg,
            rviz_config_arg,
            urdf_variant_arg,
            compensate_backlash_arg,
            ik_solver_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
