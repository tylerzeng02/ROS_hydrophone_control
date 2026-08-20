"""Launches the fus_targeting_gui node on its own. Run this ALONGSIDE
cyton_bringup's bringup.launch.py (which must already be up -- move_group
has to be running for the GUI's MoveIt bridge to connect to), not instead
of it:

    # terminal 1
    ros2 launch cyton_bringup bringup.launch.py
    # terminal 2, once the above is fully up
    ros2 launch fus_targeting_gui targeting_gui.launch.py

Pass config:=/path/to/config.yaml to override the default config bundled
with this package (config/default_config.yaml).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        "config",
        default_value="",
        description="Path to a config.yaml overriding this package's bundled default. "
        "Empty (default) uses fus_targeting_gui's own installed default_config.yaml.",
    )

    targeting_gui_node = Node(
        package="fus_targeting_gui",
        executable="targeting_gui",
        name="fus_targeting_gui",
        output="screen",
        arguments=[
            "--config", LaunchConfiguration("config"),
        ],
    )

    return LaunchDescription([
        config_arg,
        targeting_gui_node,
    ])
