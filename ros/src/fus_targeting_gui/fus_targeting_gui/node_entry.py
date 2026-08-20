"""ros2 run fus_targeting_gui targeting_gui -- entry point. Initializes
rclpy, loads config.yaml, builds the MoveIt bridge, and runs the Qt app."""

import argparse
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from PySide6.QtWidgets import QApplication

from .config import load_config
from .main_window import MainWindow
from .moveit_bridge import MoveItBridge


def _resolve_default_config_path():
    try:
        return get_package_share_directory("fus_targeting_gui") + "/config/default_config.yaml"
    except Exception:
        # Not installed via colcon (e.g. running straight from the source
        # tree during development) -- fall back to the source-tree path.
        return "config/default_config.yaml"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: this package's installed default_config.yaml)."
    )
    args, ros_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    # Empty string (e.g. from targeting_gui.launch.py's unset `config` arg
    # default) means "use the bundled default", same as omitting --config.
    config_path = args.config if args.config else _resolve_default_config_path()

    config = load_config(config_path)

    rclpy.init(args=ros_args)
    node = rclpy.create_node("fus_targeting_gui")

    app = QApplication(sys.argv)
    bridge = MoveItBridge(node, config.robot, config.targeting)
    window = MainWindow(config, bridge)
    window.resize(1400, 900)
    window.show()

    exit_code = app.exec()

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
