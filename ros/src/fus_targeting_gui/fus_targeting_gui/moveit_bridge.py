"""Thin wrapper around pymoveit2's MoveIt2 class, built entirely from
RobotConfig -- no Cyton-specific names anywhere in this file. Separates
plan and execute into two calls (rather than pymoveit2's combined
move_to_pose) specifically so the GUI can show a preview and wait for
operator confirmation before anything actually moves, matching the
plan-preview-then-confirm UX cyton_pose_commander/src/pose_commander.cpp
already established for this project's other MoveIt tools.

NOTE: written against pymoveit2's documented plan()/execute() split
(https://github.com/AndrejOrsula/pymoveit2) -- smoke-test this against
whatever pymoveit2 version actually gets built (it's a colcon-built ROS 2
package, not a pip package -- see README.md's Setup section) before
trusting it; minor API differences across versions are possible and this
hasn't been run yet.
"""

import threading

import rclpy.time
import tf2_ros
from PySide6.QtCore import QObject, QThread, Signal
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle

from .config import RobotConfig, TargetingConfig

_SKULL_COLLISION_OBJECT_ID = "skull_mesh"


class _ExecutorThread(QThread):
    """Spins the rclpy executor in the background -- same requirement (and
    same fix) as this project's C++ MoveGroupInterface tools: pymoveit2's
    service/action clients need this node's callbacks actively serviced,
    which nothing does unless something spins it."""

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
        self._base_frame = robot.base_frame
        self._end_effector_frame = robot.end_effector_frame

        # For get_current_end_effector_pose() (the Registration panel's
        # "Add Point" workflow) -- a plain TF lookup rather than guessing
        # at a pymoveit2-specific FK method name, since robot_state_publisher
        # already publishes exactly this transform continuously.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        callback_group = ReentrantCallbackGroup()
        self._moveit2 = MoveIt2(
            node=node,
            joint_names=robot.joint_names,
            base_link_name=robot.base_frame,
            end_effector_name=robot.end_effector_frame,
            group_name=robot.planning_group,
            callback_group=callback_group,
        )
        # Without these, MoveIt's 5s/1-attempt defaults are nowhere near
        # enough to randomly sample a valid goal state through elbow_yaw's
        # locked ~4-degree window -- RRTConnect fails with "Unable to
        # sample any valid states for goal tree" almost every time, not
        # because the pose is unreachable but because the search budget is
        # too small. Same fix cyton_pose_commander/src/pose_commander.cpp
        # already needed for this arm (see config/default_config.yaml's
        # own comment on these two values).
        #
        # NOTE: the real property is `allowed_planning_time`, not
        # `planning_time` -- an earlier version of this line used the wrong
        # name, which silently created an unused attribute instead of
        # raising, leaving the real value stuck at pymoveit2's internal
        # 0.5s default. Confirmed directly against the installed pymoveit2
        # (planning always failed in ~0.5s regardless of this setting)
        # before/after fixing it -- don't reintroduce the typo'd name.
        self._moveit2.allowed_planning_time = targeting.planning_time_s
        self._moveit2.num_planning_attempts = targeting.num_planning_attempts
        self._moveit2.planner_id = "RRTConnectkConfigDefault"

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._executor_thread = _ExecutorThread(self._executor)
        self._executor_thread.start()

        # For set_skull_collision_object() -- registering the skull as a
        # real obstacle the planner avoids, not just something drawn in
        # the GUI's own 3D view (that gap was the whole point of adding
        # this).
        self._apply_planning_scene_client = node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )

    def shutdown(self):
        self._executor_thread.stop()
        self._executor_thread.wait()

    def get_current_end_effector_pose(self):
        """Blocking (call from a worker thread) -- looks up the end
        effector's CURRENT live position in base_frame via TF, in meters.
        Used by the Registration panel: physically position the arm's
        probe against a real landmark (e.g. via RViz), then call this to
        record where it actually is, paired with the mesh point you
        clicked for the same landmark. Returns (x, y, z), or None if the
        transform isn't available yet (e.g. robot_state_publisher not up)
        -- reported via the `status` signal either way, never raises."""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, self._end_effector_frame, rclpy.time.Time()
            )
        except tf2_ros.TransformException as e:  # noqa: BLE001
            self.status.emit(f"Could not look up current end effector pose: {e}")
            return None
        t = transform.transform.translation
        return (t.x, t.y, t.z)

    def plan_to_pose(self, pose: Pose):
        """Blocking (called from a worker thread, not the GUI thread -- see
        main_window.py) -- returns a planned trajectory, or None if
        planning failed."""
        self.status.emit("Planning...")
        try:
            trajectory = self._moveit2.plan(
                position=[pose.position.x, pose.position.y, pose.position.z],
                quat_xyzw=[
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w,
                ],
            )
        except Exception as e:  # noqa: BLE001 -- report to the GUI, don't crash it
            self.status.emit(f"Planning FAILED: {e}")
            self.plan_ready.emit(None)
            return None

        if trajectory is None:
            self.status.emit("Planning FAILED: no trajectory returned.")
        else:
            self.status.emit("Plan ready -- review before executing.")
        self.plan_ready.emit(trajectory)
        return trajectory

    def execute(self, trajectory):
        """Blocking (called from a worker thread) -- executes an
        already-planned trajectory and waits for completion."""
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

    def set_skull_collision_object(self, vertices_base_frame, triangles):
        """Blocking (call from a worker thread) -- publishes (ADD, which
        also replaces any existing object with the same id) a real MoveIt
        collision object named "skull_mesh" via /apply_planning_scene, so
        the planner genuinely avoids it -- not just something drawn in the
        GUI's own PyVista view, which move_group has no knowledge of at
        all otherwise (confirmed directly: this codebase never touched
        PlanningScene/CollisionObject before this method existed).

        vertices_base_frame: (N, 3) already in base_frame (see
        Registration.transform_points_to_base_frame()). triangles: (M, 3)
        int indices into vertices. Returns True on success.

        Uses call_async() + a threading.Event, not
        rclpy.spin_until_future_complete() -- this node's executor is
        already spinning on _ExecutorThread (see __init__), and spinning
        the same node from a second thread concurrently would be unsafe;
        the already-running executor services this call's response, this
        thread just waits for it."""
        if not self._apply_planning_scene_client.wait_for_service(timeout_sec=5.0):
            self.status.emit("Could not reach /apply_planning_scene service.")
            return False

        mesh = Mesh()
        mesh.vertices = [
            Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in vertices_base_frame
        ]
        mesh.triangles = [
            MeshTriangle(vertex_indices=[int(t[0]), int(t[1]), int(t[2])]) for t in triangles
        ]

        identity_pose = Pose()
        identity_pose.orientation.w = 1.0  # vertices already in base_frame, no extra offset

        obj = CollisionObject()
        obj.header.frame_id = self._base_frame
        obj.id = _SKULL_COLLISION_OBJECT_ID
        obj.meshes = [mesh]
        obj.mesh_poses = [identity_pose]
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        done_event = threading.Event()
        result_holder = {}

        def _on_done(future):
            result_holder["response"] = future.result()
            done_event.set()

        future = self._apply_planning_scene_client.call_async(request)
        future.add_done_callback(_on_done)
        if not done_event.wait(timeout=10.0):
            self.status.emit("Timed out waiting for /apply_planning_scene response.")
            return False

        response = result_holder.get("response")
        success = bool(response is not None and response.success)
        self.status.emit(
            f"Skull collision object published ({len(triangles)} triangles)."
            if success else "Failed to publish skull collision object."
        )
        return success

    def clear_skull_collision_object(self):
        """Removes the skull collision object (e.g. before publishing an
        updated one for a newly-loaded mesh -- ADD already replaces by id,
        so this is mainly for an explicit "no skull" state)."""
        if not self._apply_planning_scene_client.wait_for_service(timeout_sec=5.0):
            return False

        obj = CollisionObject()
        obj.header.frame_id = self._base_frame
        obj.id = _SKULL_COLLISION_OBJECT_ID
        obj.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        done_event = threading.Event()
        future = self._apply_planning_scene_client.call_async(request)
        future.add_done_callback(lambda _f: done_event.set())
        done_event.wait(timeout=10.0)
        return True
