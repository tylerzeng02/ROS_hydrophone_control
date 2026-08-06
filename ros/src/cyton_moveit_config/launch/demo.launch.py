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
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


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

    moveit_config = (
        MoveItConfigsBuilder("cyton_gamma_1500", package_name="cyton_moveit_config")
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory("cyton_description"),
                "urdf",
                "cyton_gamma_1500.urdf.xacro",
            ),
            mappings={
                "hardware_type": LaunchConfiguration("hardware_type"),
                "serial_port": LaunchConfiguration("serial_port"),
                "baud_rate": LaunchConfiguration("baud_rate"),
            },
        )
        .robot_description_semantic(file_path="config/cyton_gamma_1500.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
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

    return LaunchDescription(
        [
            hardware_type_arg,
            serial_port_arg,
            baud_rate_arg,
            rviz_config_arg,
            static_tf_node,
            robot_state_publisher_node,
            move_group_node,
            rviz_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
        ]
    )
