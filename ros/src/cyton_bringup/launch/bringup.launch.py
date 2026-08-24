"""Single top-level entry point for this workspace. Thin wrapper around
cyton_moveit_config/launch/demo.launch.py -- see that file for what
actually gets started (robot_state_publisher, ros2_control_node, MoveIt's
move_group, RViz, controller spawners).

Safe, hardware-free default:
    ros2 launch cyton_bringup bringup.launch.py

Real hardware (arm connected, powered, clear to move):
    ros2 launch cyton_bringup bringup.launch.py hardware_type:=real serial_port:=/dev/ttyUSB0

Uncalibrated-comparison A/B test (2026-08-12): pass urdf_variant:=uncalibrated
to have MoveIt plan against the original, uncorrected joint geometry instead
of this project's fitted kinematic calibration -- e.g.
    ros2 launch cyton_bringup bringup.launch.py hardware_type:=real serial_port:=/dev/ttyUSB0 urdf_variant:=uncalibrated
See cyton_moveit_config/launch/demo.launch.py's own docstring for exactly
what this does and doesn't change.

Full 7-DOF simulation (2026-08-24): pass urdf_variant:=sim_7dof (mock_components
only -- rejected with hardware_type:=real) to restore elbow_yaw_joint's full
mechanical range for planning/simulation instead of the production ~4-degree
lock -- e.g.
    ros2 launch cyton_bringup bringup.launch.py urdf_variant:=sim_7dof
See cyton_gamma_1500_robot_sim7dof.xacro's own header for why this is safe
only in simulation.

Streaming backlash compensation (2026-08-13): pass compensate_backlash:=true
(only meaningful with hardware_type:=real) to enable cyton_hardware's new
reversal-triggered hold-point compensator -- NOT yet validated against real
hardware. See CytonSystemHardware's own header comment.

Pose-dependent correction (2026-08-13): pass compensate_pose_dependent:=true
(only meaningful with hardware_type:=real) to enable the ported joint-
coupling/gravity-deflection/shoulder_pitch-Fourier correction
(src/pose_dependent_correction.cpp) -- numerically verified against the
Python model it ports, but NOT yet validated as live control against real
hardware. See CytonSystemHardware's own header comment.

TRAC-IK (2026-08-13): pass ik_solver:=trac_ik to use
cyton_trac_ik_kinematics_plugin instead of KDL -- brand new, not yet
hardware-validated. See that plugin's own header comment.
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
    urdf_variant_arg = DeclareLaunchArgument(
        "urdf_variant",
        default_value="calibrated",
        description="'calibrated' (default), 'uncalibrated', or 'sim_7dof' (full elbow_yaw "
        "range, mock_components only) -- see demo.launch.py's docstring for what each does.",
    )
    compensate_backlash_arg = DeclareLaunchArgument(
        "compensate_backlash",
        default_value="false",
        description="Enable cyton_hardware's streaming backlash compensator (only meaningful "
        "with hardware_type:=real). Default false, not yet validated against real hardware.",
    )
    compensate_pose_dependent_arg = DeclareLaunchArgument(
        "compensate_pose_dependent",
        default_value="false",
        description="Enable the ported joint-coupling/gravity-deflection/shoulder_pitch-Fourier "
        "correction (only meaningful with hardware_type:=real). Default false, not yet "
        "validated as live control against real hardware.",
    )
    ik_solver_arg = DeclareLaunchArgument(
        "ik_solver",
        default_value="kdl",
        description="'kdl' (default, production-proven) or 'trac_ik' (brand new, not yet "
        "hardware-validated) -- see demo.launch.py's docstring.",
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
            "urdf_variant": LaunchConfiguration("urdf_variant"),
            "compensate_backlash": LaunchConfiguration("compensate_backlash"),
            "compensate_pose_dependent": LaunchConfiguration("compensate_pose_dependent"),
            "ik_solver": LaunchConfiguration("ik_solver"),
        }.items(),
    )

    return LaunchDescription(
        [
            hardware_type_arg,
            serial_port_arg,
            baud_rate_arg,
            urdf_variant_arg,
            compensate_backlash_arg,
            compensate_pose_dependent_arg,
            ik_solver_arg,
            demo_launch,
        ]
    )
