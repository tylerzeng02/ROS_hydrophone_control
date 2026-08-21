"""Thin wrapper around pymoveit2's MoveIt2 class, built entirely from
RobotConfig. No Cyton-specific names appear anywhere in this file.
Plan and execute are separated into two calls, rather than using
pymoveit2's combined move_to_pose, so the GUI can show a preview and wait
for operator confirmation before anything actually moves. This matches
the plan-preview-then-confirm UX already established in
cyton_pose_commander/src/pose_commander.cpp for this project's other
MoveIt tools.

This is written against pymoveit2's documented plan/execute split
(https://github.com/AndrejOrsula/pymoveit2). Smoke-test it against
whatever pymoveit2 version actually gets installed (see requirements.txt)
before trusting it. Minor API differences across versions are possible
and this has not been run yet.
"""

from PySide6.QtCore import QObject, QThread, Signal
from geometry_msgs.msg import Pose
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from .config import RobotConfig, TargetingConfig


class _ExecutorThread(QThread):
    """Spins the rclpy executor in the background. This is the same
    requirement, and the same fix, as this project's C++ MoveGroupInterface
    tools use. pymoveit2's service and action clients need this node's
    callbacks actively serviced, and nothing does that unless something
    spins the executor."""

    def __init__(self, executor: SingleThreadedExecutor, parent=None):
        super().__init__(parent)
        self._executor = executor

    def run(self):
        self._executor.spin()

    def stop(self):
        self._executor.shutdown()


class MoveItBridge(QObject):
    status = Signal(str)
    plan_ready = Signal(object)     # emits the planned trajectory (or None on failure)
    execute_finished = Signal(bool)  # True on success

    def __init__(self, node: Node, robot: RobotConfig, targeting: TargetingConfig, parent=None):
        super().__init__(parent)
        self._node = node
        self._targeting = targeting

        callback_group = ReentrantCallbackGroup()
        self._moveit2 = MoveIt2(
            node=node,
            joint_names=robot.joint_names,
            base_link_name=robot.base_frame,
            end_effector_name=robot.end_effector_frame,
            group_name=robot.planning_group,
            callback_group=callback_group,
        )
        self._moveit2.planner_id = "RRTConnectkConfigDefault"

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._executor_thread = _ExecutorThread(self._executor)
        self._executor_thread.start()

    def shutdown(self):
        self._executor_thread.stop()
        self._executor_thread.wait()

    def plan_to_pose(self, pose: Pose):
        """This call blocks and must be called from a worker thread, not
        the GUI thread (see main_window.py). Returns a planned trajectory,
        or None if planning failed."""
        self.status.emit("Planning...")
        try:
            trajectory = self._moveit2.plan(
                position=[pose.position.x, pose.position.y, pose.position.z],
                quat_xyzw=[
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w,
                ],
            )
        except Exception as e:  # noqa: BLE001, reported to the GUI instead of crashing it
            self.status.emit(f"Planning FAILED: {e}")
            self.plan_ready.emit(None)
            return None

        if trajectory is None:
            self.status.emit("Planning FAILED: no trajectory returned.")
        else:
            self.status.emit("Plan ready. Review it before executing.")
        self.plan_ready.emit(trajectory)
        return trajectory

    def execute(self, trajectory):
        """This call blocks and must be called from a worker thread. It
        executes an already-planned trajectory and waits for completion."""
        self.status.emit("Executing...")
        try:
            self._moveit2.execute(trajectory)
            self._moveit2.wait_until_executed()
            success = True
        except Exception as e:  # noqa: BLE001
            self.status.emit(f"Execution FAILED: {e}")
            success = False

        if success:
            self.status.emit("Execution succeeded.")
        self.execute_finished.emit(success)
        return success
