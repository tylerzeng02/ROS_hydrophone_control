"""Thin wrapper around pymoveit2's MoveIt2 class, built entirely from
RobotConfig. No Cyton-specific names appear anywhere in this file.
Plan and execute are separated into two calls, rather than using
pymoveit2's combined move_to_pose, so the GUI can show a preview and wait
for operator confirmation before anything actually moves. This matches
the plan-preview-then-confirm UX already established in
cyton_pose_commander/src/pose_commander.cpp for this project's other
MoveIt tools.

Written against pymoveit2's documented plan()/execute() split
(https://github.com/AndrejOrsula/pymoveit2), built via colcon per this
package's own README.md "Setup" section (pymoveit2 is not a pip package
-- see requirements.txt for why). Confirmed working against a live
move_group (2026-08-24/25): real plan()/execute() calls, position-only
and orientation-constrained targets, and the bounds-recovery mechanism
below have all been run and verified, not just written.
"""

import threading
import xml.etree.ElementTree as ET

import rclpy.time
import tf2_ros
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from PySide6.QtCore import QObject, QThread, Signal
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from .config import RobotConfig, TargetingConfig

_SKULL_COLLISION_OBJECT_ID = "skull_mesh"

# Safety margin subtracted from each joint's URDF limit before treating the
# current state as "in bounds" -- matches cyton_pose_commander/src/
# pose_commander.cpp's RECOVERY_BUFFER_RAD exactly. Without this, a joint
# sitting precisely on its limit (not past it) can still trip MoveIt's
# CheckStartStateBounds and get an instant, generic FAILURE rejection
# before planning even starts -- the bug this whole mechanism exists to
# work around (see ensure_current_state_within_bounds()'s docstring).
_RECOVERY_BUFFER_RAD = 0.01


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
        self._base_frame = robot.base_frame
        self._end_effector_frame = robot.end_effector_frame
        self._joint_names = list(robot.joint_names)

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

        # For ensure_current_state_within_bounds() -- reads each joint's
        # real <limit lower/upper> from the currently-loaded robot_description
        # (not hardcoded, so this stays correct across URDF variants, e.g.
        # elbow_yaw's range differs a lot between the production-locked and
        # sim_7dof variants) and sends a corrective trajectory directly to
        # the controller when needed, bypassing MoveIt's planning pipeline
        # (which refuses to plan from an out-of-bounds start state, so it
        # can't fix this itself) -- same mechanism as
        # cyton_pose_commander/src/pose_commander.cpp's
        # ensureCurrentStateWithinBounds()/sendCorrectiveTrajectory(), ported
        # to a native rclpy action client instead of shelling out to `ros2
        # action send_goal`.
        self._joint_limits = self._fetch_joint_limits()
        self._trajectory_action_client = ActionClient(
            node, FollowJointTrajectory, robot.controller_action_name
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

    def _fetch_joint_limits(self):
        """Blocking (call only from __init__, before the executor thread's
        spin is doing anything else useful yet) -- reads one message from
        the transient-local /robot_description topic and parses each of
        self._joint_names' <limit lower="..." upper="..."/> out of the URDF
        XML. Returns {name: (lower, upper)}; a joint missing a <limit> (a
        continuous/fixed joint, shouldn't happen for this arm's revolute
        joints) is simply omitted, and ensure_current_state_within_bounds()
        skips checking any joint not in this dict."""
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        done_event = threading.Event()
        result_holder = {}

        def _on_msg(msg):
            if "xml" not in result_holder:
                result_holder["xml"] = msg.data
                done_event.set()

        sub = self._node.create_subscription(String, "/robot_description", _on_msg, qos)
        if not done_event.wait(timeout=10.0):
            self.status.emit(
                "Could not read /robot_description to determine joint limits -- "
                "the out-of-bounds recovery check will be skipped."
            )
            self._node.destroy_subscription(sub)
            return {}
        self._node.destroy_subscription(sub)

        limits = {}
        try:
            root = ET.fromstring(result_holder["xml"])
            for joint_el in root.findall("joint"):
                name = joint_el.get("name")
                if name not in self._joint_names:
                    continue
                limit_el = joint_el.find("limit")
                if limit_el is None:
                    continue
                lower = limit_el.get("lower")
                upper = limit_el.get("upper")
                if lower is None or upper is None:
                    continue
                limits[name] = (float(lower), float(upper))
        except ET.ParseError as e:  # noqa: BLE001
            self.status.emit(f"Could not parse /robot_description XML: {e}")
            return {}

        missing = [n for n in self._joint_names if n not in limits]
        if missing:
            self.status.emit(
                f"No <limit> found for joint(s) {missing} -- these will not be "
                "checked by the out-of-bounds recovery."
            )
        return limits

    def ensure_current_state_within_bounds(self) -> bool:
        """Blocking (call from a worker thread, before every plan attempt)
        -- reads the arm's current joint state and, if any joint is at or
        beyond (limit +/- _RECOVERY_BUFFER_RAD), sends a corrective
        trajectory to pull it back inside before returning.

        Why this exists: MoveIt's CheckStartStateBounds rejects planning
        outright (an immediate, generic FAILURE error code, not a real
        search timeout) if the CURRENT state has any joint sitting exactly
        on or past its limit -- confirmed directly (2026-08-25): after a
        sequence of moves, elbow_yaw_joint was found parked at exactly
        1.790156 rad, bit-for-bit its own sim_7dof URDF upper limit, and
        every subsequent plan attempt failed in under ~1 second (not the
        full ~30s planning budget) until this was manually corrected. Same
        bug class already fixed once in
        cyton_pose_commander/src/pose_commander.cpp; this ports the same
        fix here so "Simulate All Targets" can recover on its own instead
        of just failing.

        Returns True if the state was already fine, or was successfully
        corrected. Returns False if a correction was needed but failed (the
        caller should skip this target rather than plan from a
        possibly-still-invalid state)."""
        joint_state = self._moveit2.joint_state
        if joint_state is None:
            self.status.emit(
                "Could not read current joint state to check bounds -- "
                "proceeding without this check."
            )
            return True

        current = dict(zip(joint_state.name, joint_state.position))
        corrected = []
        any_clamped = False
        for name in self._joint_names:
            value = current.get(name)
            if value is None or name not in self._joint_limits:
                # Can't check this joint (not reported, or no <limit> found)
                # -- pass its current value through unchanged if we have
                # one, matching the "proceed without this check" fallback.
                corrected.append(value if value is not None else 0.0)
                continue
            lower, upper = self._joint_limits[name]
            safe_lower = lower + _RECOVERY_BUFFER_RAD
            safe_upper = upper - _RECOVERY_BUFFER_RAD
            if value < safe_lower:
                self.status.emit(
                    f"{name} is currently {value:.4f} rad, below safe bound "
                    f"{safe_lower:.4f} -- clamping."
                )
                value = safe_lower
                any_clamped = True
            elif value > safe_upper:
                self.status.emit(
                    f"{name} is currently {value:.4f} rad, above safe bound "
                    f"{safe_upper:.4f} -- clamping."
                )
                value = safe_upper
                any_clamped = True
            corrected.append(value)

        if not any_clamped:
            return True

        self.status.emit(
            "Current state has a joint out of its safe bound (most likely "
            "left over from a prior move) -- sending a corrective trajectory "
            "directly to the controller (bypassing MoveIt, since it won't "
            "plan from an invalid start state)..."
        )
        return self._send_corrective_trajectory(corrected)

    def _send_corrective_trajectory(self, positions) -> bool:
        """Blocking -- sends a single-point trajectory with ALL joints'
        positions (not just the corrected one(s)), bypassing MoveIt's
        planning pipeline entirely via a direct FollowJointTrajectory action
        goal to the real-time controller. Same call_async()+threading.Event
        pattern as set_skull_collision_object() -- this node's executor is
        already spinning on _ExecutorThread, so this thread just waits for
        the response rather than spinning itself."""
        if not self._trajectory_action_client.wait_for_server(timeout_sec=5.0):
            self.status.emit("Could not reach the trajectory controller action server.")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self._joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=2, nanosec=0)
        goal.trajectory.points = [point]

        accepted_event = threading.Event()
        result_event = threading.Event()
        result_holder = {}

        def _on_result(future):
            result_holder["result"] = future.result()
            result_event.set()

        def _on_goal_response(future):
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                result_holder["rejected"] = True
                result_event.set()
                accepted_event.set()
                return
            accepted_event.set()
            goal_handle.get_result_async().add_done_callback(_on_result)

        send_future = self._trajectory_action_client.send_goal_async(goal)
        send_future.add_done_callback(_on_goal_response)

        if not accepted_event.wait(timeout=10.0):
            self.status.emit("Timed out waiting for the corrective trajectory to be accepted.")
            return False
        if result_holder.get("rejected"):
            self.status.emit("Corrective trajectory goal was rejected by the controller.")
            return False
        if not result_event.wait(timeout=10.0):
            self.status.emit("Timed out waiting for the corrective trajectory to finish.")
            return False

        result = result_holder.get("result")
        success = bool(result is not None and result.result.error_code == 0)
        self.status.emit(
            "Corrective trajectory succeeded."
            if success else f"Corrective trajectory FAILED (error_code={getattr(result.result, 'error_code', '?') if result else '?'})."
        )
        return success

    def plan_to_pose(self, pose: Pose):
        """Blocking (called from a worker thread, not the GUI thread -- see
        main_window.py) -- returns a planned trajectory, or None if
        planning failed.

        Calls ensure_current_state_within_bounds() proactively before
        planning (see that method's docstring), and once more if the first
        plan attempt fails, in case the state only became invalid partway
        through (mirrors pose_commander.cpp's retry-once behavior)."""
        self.ensure_current_state_within_bounds()

        self.status.emit("Planning...")
        plan_kwargs = {"position": [pose.position.x, pose.position.y, pose.position.z]}
        if self._targeting.enforce_orientation:
            plan_kwargs["quat_xyzw"] = [
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w,
            ]
        try:
            trajectory = self._moveit2.plan(**plan_kwargs)
        except Exception as e:  # noqa: BLE001 -- report to the GUI, don't crash it
            self.status.emit(f"Planning FAILED: {e}")
            self.plan_ready.emit(None)
            return None

        if trajectory is None:
            self.status.emit(
                "Planning failed -- checking for an out-of-bounds current "
                "state and retrying once..."
            )
            if self.ensure_current_state_within_bounds():
                try:
                    trajectory = self._moveit2.plan(**plan_kwargs)
                except Exception as e:  # noqa: BLE001
                    self.status.emit(f"Planning FAILED: {e}")
                    trajectory = None

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
