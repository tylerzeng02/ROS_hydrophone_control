"""Single top-level entry point for this workspace. Thin wrapper around
cyton_moveit_config/launch/demo.launch.py -- see that file for what
actually gets started (robot_state_publisher, ros2_control_node, MoveIt's
move_group, RViz, controller spawners).

Safe, hardware-free default:
    ros2 launch cyton_bringup bringup.launch.py

Real hardware (arm connected, powered, clear to move):
    ros2 launch cyton_bringup bringup.launch.py hardware_type:=real serial_port:=/dev/ttyUSB0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hardware_type_arg = DeclareLaunchArgument(
        "hardware_type",
        default_value="mock_components",
        description="ros2_control hardware plugin: mock_components (safe default) or real "
        "(cyton_hardware/CytonSystemHardware -- only with the physical arm connected, "
        "powered, and clear to move)",
    )
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyUSB0",
        description="Serial device for the Dynamixel bus (only used when hardware_type:=real)",
    )
    baud_rate_arg = DeclareLaunchArgument(
        "baud_rate", default_value="1000000", description="Dynamixel bus baud rate"
    )

    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("cyton_moveit_config"), "launch", "demo.launch.py"]
            )
        ),
        launch_arguments={
            "hardware_type": LaunchConfiguration("hardware_type"),
            "serial_port": LaunchConfiguration("serial_port"),
            "baud_rate": LaunchConfiguration("baud_rate"),
        }.items(),
    )

    return LaunchDescription(
        [
            hardware_type_arg,
            serial_port_arg,
            baud_rate_arg,
            demo_launch,
        ]
    )
