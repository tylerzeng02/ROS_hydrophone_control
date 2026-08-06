"""Standalone visualization: robot_state_publisher + joint_state_publisher_gui
+ RViz, no ros2_control/MoveIt involved. Useful as the very first sanity
check that the xacro/meshes resolve and the kinematic chain looks right,
before bringing controller_manager or move_group into the picture.

Usage:
    ros2 launch cyton_description display.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("cyton_description"), "rviz", "urdf.rviz"]
        ),
        description="RViz config file to load",
    )

    robot_description_content = ParameterValue(
        Command(
            [
                "xacro ",
                PathJoinSubstitution(
                    [
                        FindPackageShare("cyton_description"),
                        "urdf",
                        "cyton_gamma_1500.urdf.xacro",
                    ]
                ),
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_description_content}],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", LaunchConfiguration("rviz_config")],
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            robot_state_publisher_node,
            joint_state_publisher_gui_node,
            rviz_node,
        ]
    )
