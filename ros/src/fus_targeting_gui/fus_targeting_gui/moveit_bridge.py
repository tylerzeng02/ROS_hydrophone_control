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
import time
import xml.etree.ElementTree as ET

import rclpy.time
import tf2_ros
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from PySide6.QtCore import QObject, QThread, Signal
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.utils import enum_to_str
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


def _is_identity_pose(pose):
    """set_skull_collision_object() always publishes with an identity
    pose (the registration transform is baked into the vertices instead),
    so a non-identity pose on this object can only have come from
    something else moving it afterward, e.g. RViz's Scene Objects panel.
    Shared by get_skull_collision_pose() and the automatic
    /monitored_planning_scene watcher below."""
    p, q = pose.position, pose.orientation
    return (
        abs(p.x) < 1e-6 and abs(p.y) < 1e-6 and abs(p.z) < 1e-6
        and abs(q.x) < 1e-6 and abs(q.y) < 1e-6 and abs(q.z) < 1e-6
        and abs(q.w - 1.0) < 1e-6
    )

# Each single plan_async() call already runs MoveIt's own internal
# num_planning_attempts (15, see config/default_config.yaml) random-seeded
# tries within a shared allowed_planning_time (30s) budget, not 30s per
# attempt. An easy target fails fast (observed well under a second) since
# RRTConnect gives up quickly when a request is structurally invalid, but
# a genuinely hard or borderline-unreachable target can exhaust the full
# 30s budget on every single attempt (confirmed directly: consecutive
# attempts landing ~30s apart), making this count's real-world cost highly
# target-dependent. Raised back to 5 (2026-08-26, user choice) once real
# usage showed attempts are quick in practice; worst case for a genuinely
# hard target is around 150s. A START_STATE_IN_COLLISION failure is
# handled separately, through _retreat_to_last_safe_state(), since
# retrying the identical request without moving first fails identically
# every time regardless of this count.
_MAX_PLAN_ATTEMPTS = 5

# How many times to retry the original target after a successful nudge
# (see _attempt_nudge_recovery()), rather than just once: an identical
# request can still succeed on a later attempt with no other change, the
# same reasoning as _MAX_PLAN_ATTEMPTS above, and this is functionally
# the same as a person clicking Try Again by hand that many times.
_NUDGE_RETRY_ATTEMPTS = 5

# Last-resort recovery when every normal attempt above has failed: try a
# small move in a different direction, then retry the original target once
# more. Sometimes the current arm configuration itself, not the target, is
# what makes a plan hard to find; shifting slightly can put the arm into a
# configuration the planner has an easier time working from. Ten unit
# directions are a simple, deterministic stand-in for "any available
# direction": the 6 axis-aligned face directions plus 4 tetrahedral cube-
# corner diagonals (alternating corners, so the 4 are spread evenly through
# 3D space rather than clustered), each normalized to length 1 so every
# candidate nudge is exactly _NUDGE_MAGNITUDE_M regardless of how many axes
# it moves along. Each candidate is planned and collision-checked through
# MoveIt's own normal pipeline, so a nudge is only ever executed if it is
# itself a fully verified, safe move, not a raw bypass like
# _send_corrective_trajectory()'s. If every direction fails,
# _attempt_nudge_recovery() returns False and plan_to_pose() falls through
# to its normal failure path unchanged, which main_window.py already turns
# into the Try Again/Cancel dialog. 1cm is small enough not to be a
# meaningful departure from the requested target's own position.
_NUDGE_MAGNITUDE_M = 0.01
_NUDGE_DIAGONAL = 1.0 / (3.0 ** 0.5)
_NUDGE_DIRECTIONS = [
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (_NUDGE_DIAGONAL, _NUDGE_DIAGONAL, _NUDGE_DIAGONAL),
    (_NUDGE_DIAGONAL, -_NUDGE_DIAGONAL, -_NUDGE_DIAGONAL),
    (-_NUDGE_DIAGONAL, _NUDGE_DIAGONAL, -_NUDGE_DIAGONAL),
    (-_NUDGE_DIAGONAL, -_NUDGE_DIAGONAL, _NUDGE_DIAGONAL),
]
# Allowed_planning_time used only for the nudge directions above, not the
# configured targeting.planning_time_s (30s default): a 1cm move is simple
# enough that a much shorter budget is normally plenty, and capping it
# keeps the worst case for all 10 directions combined well under a minute
# instead of up to 10 * 30s.
_NUDGE_PLANNING_TIME_S = 3.0

# Safety margin subtracted from each joint's URDF limit before treating the
# current state as "in bounds". Matches cyton_pose_commander/src/
# pose_commander.cpp's RECOVERY_BUFFER_RAD exactly. Without this, a joint
# sitting precisely on its limit, not past it, can still trip MoveIt's
# CheckStartStateBounds and get an instant, generic FAILURE rejection
# before planning even starts. This is the bug this whole mechanism exists
# to work around (see ensure_current_state_within_bounds()'s docstring).
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
    # Emits (xyz, quat_xyzw) whenever /monitored_planning_scene reports the
    # skull collision object at a non-identity pose, i.e. something moved
    # it externally (RViz's Scene Objects panel drag + Publish). Lets
    # main_window.py auto-sync without the user needing to click Sync From
    # Scene Move -- see _on_monitored_planning_scene()'s own docstring for
    # why this can't replace that button entirely.
    scene_skull_moved = Signal(object)

    def __init__(self, node: Node, robot: RobotConfig, targeting: TargetingConfig, parent=None):
        super().__init__(parent)
        self._node = node
        self._targeting = targeting
        self._base_frame = robot.base_frame
        self._end_effector_frame = robot.end_effector_frame
        self._joint_names = list(robot.joint_names)
        self._planning_group = robot.planning_group
        self._home_joint_positions = list(robot.home_joint_positions)

        # For get_current_end_effector_pose() (the Registration panel's
        # "Add Point" workflow): a plain TF lookup rather than guessing at
        # a pymoveit2-specific FK method name, since robot_state_publisher
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
        # locked ~4-degree window. RRTConnect fails with "Unable to sample
        # any valid states for goal tree" almost every time, not because
        # the pose is unreachable but because the search budget is too
        # small. Same fix cyton_pose_commander/src/pose_commander.cpp
        # already needed for this arm (see config/default_config.yaml's
        # own comment on these two values).
        #
        # NOTE: the property is `allowed_planning_time`, not
        # `planning_time`. Python does not raise on assigning an unknown
        # attribute, so a typo here silently no-ops instead of erroring,
        # leaving planning stuck at pymoveit2's internal 0.5s default.
        self._moveit2.allowed_planning_time = targeting.planning_time_s
        self._moveit2.num_planning_attempts = targeting.num_planning_attempts
        self._moveit2.planner_id = "RRTConnectkConfigDefault"

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._executor_thread = _ExecutorThread(self._executor)
        self._executor_thread.start()

        # For set_skull_collision_object(): registers the skull with
        # MoveIt's planning scene, so the planner avoids it. Distinct from
        # rendering it in the GUI's own 3D view, which the planner has no
        # knowledge of.
        self._apply_planning_scene_client = node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )

        # For ensure_current_state_not_in_collision(): the raw
        # /plan_kinematic_path service pymoveit2.plan_async() calls only
        # ever returns the generic MoveItErrorCodes.FAILURE when a
        # PlanningRequestAdapter (e.g. CheckStartStateCollision) aborts
        # the pipeline, never the specific START_STATE_IN_COLLISION value
        # (confirmed directly: that specific code only surfaces through
        # the separate /move_action interface, which this bridge does not
        # use). So collision cannot be reliably detected by inspecting the
        # plan response's error code; this queries collision state
        # directly instead, the same way ensure_current_state_within_bounds()
        # checks bounds directly rather than waiting for a plan to fail.
        self._state_validity_client = node.create_client(
            GetStateValidity, "/check_state_validity"
        )

        # For get_skull_collision_pose(): reads back the skull collision
        # object's current pose from the live planning scene. Needed
        # because RViz's Scene Objects panel can move that same object
        # (drag + its own Publish action) independently of this GUI, and
        # that pose offset otherwise has no way to reach the GUI's own
        # registration transform or already-picked target poses.
        self._get_planning_scene_client = node.create_client(
            GetPlanningScene, "/get_planning_scene"
        )

        # For automatic scene-move detection (scene_skull_moved): MoveIt
        # echoes every planning-scene change, including RViz's Scene
        # Objects panel Publish action, on this topic -- subscribing here
        # means main_window.py finds out as soon as it happens, instead of
        # only when the user remembers to click Sync From Scene Move.
        self._planning_scene_sub = node.create_subscription(
            PlanningScene, "/monitored_planning_scene",
            self._on_monitored_planning_scene, 10,
        )

        # For ensure_current_state_within_bounds(): reads each joint's
        # <limit lower/upper> from the currently-loaded robot_description,
        # not hardcoded, so this stays correct across URDF variants (e.g.
        # elbow_yaw's range differs a lot between the production-locked
        # and sim_7dof variants), and sends a corrective trajectory
        # directly to the controller when needed, bypassing MoveIt's
        # planning pipeline, which refuses to plan from an out-of-bounds
        # start state and so cannot fix this itself. Same mechanism as
        # cyton_pose_commander/src/pose_commander.cpp's
        # ensureCurrentStateWithinBounds()/sendCorrectiveTrajectory(),
        # ported to a native rclpy action client instead of shelling out
        # to `ros2 action send_goal`.
        self._joint_limits = self._fetch_joint_limits()
        self._trajectory_action_client = ActionClient(
            node, FollowJointTrajectory, robot.controller_action_name
        )

        # The joint state a plan was last successfully computed from. Since
        # MoveIt only returns a trajectory when CheckStartStateCollision
        # passed for that state, this is by definition a collision-free
        # configuration to retreat to if a later plan attempt reports
        # START_STATE_IN_COLLISION. None until the first successful plan.
        self._last_safe_joint_state = None

    def shutdown(self):
        self._executor_thread.stop()
        self._executor_thread.wait()

    def get_current_end_effector_pose(self):
        """Blocking (call from a worker thread). Looks up the end
        effector's current position in base_frame via TF, in meters. Used
        by the Registration panel: physically position the arm's probe
        against a landmark (e.g. via RViz), then call this to record
        where it is, paired with the mesh point clicked for the same
        landmark.

        Returns:
            (x, y, z) tuple, or None if the transform isn't available yet
            (e.g. robot_state_publisher not up). Reported via the
            `status` signal either way; never raises.
        """
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
        spin does anything else useful). Reads one message from the
        transient-local /robot_description topic and parses each of
        self._joint_names' <limit lower="..." upper="..."/> out of the
        URDF XML.

        Returns:
            {name: (lower, upper)}. A joint missing a <limit> (a
            continuous/fixed joint, not expected for this arm's revolute
            joints) is omitted; ensure_current_state_within_bounds() skips
            checking any joint not in this dict.
        """
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
        """Blocking (call from a worker thread, before every plan attempt).
        Reads the arm's current joint state and, if any joint is at or
        beyond (limit +/- _RECOVERY_BUFFER_RAD), sends a corrective
        trajectory to pull it back inside before returning.

        Why this exists: MoveIt's CheckStartStateBounds rejects planning
        outright, with an immediate, generic FAILURE error code rather
        than a search timeout, if the current state has any joint sitting
        exactly on or past its limit. A joint can end up exactly on its
        limit after a sequence of moves (e.g. elbow_yaw_joint parked at
        its own sim_7dof URDF upper limit), and every subsequent plan
        attempt then fails fast until the joint is nudged back inside.
        Same mechanism as
        cyton_pose_commander/src/pose_commander.cpp's
        ensureCurrentStateWithinBounds(), so "Simulate All Targets" can
        recover on its own instead of failing outright.

        Returns:
            True if the state was already within bounds, or was corrected.
            False if a correction was needed but failed; the caller should
            skip this target rather than plan from a possibly still
            invalid state.
        """
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
                # Can't check this joint (not reported, or no <limit>
                # found). Pass its current value through unchanged if we
                # have one, matching the "proceed without this check"
                # fallback.
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

    def ensure_current_state_not_in_collision(self) -> bool:
        """Blocking (call from a worker thread, before every plan attempt).
        Queries /check_state_validity directly for the arm's current
        joint state and, if it is in collision, retreats to
        _last_safe_joint_state (see _retreat_to_last_safe_state()) before
        returning.

        Why this exists: unlike ensure_current_state_within_bounds()'s
        problem (a joint sitting exactly on its own limit, which shows up
        as a plan failure this bridge can react to), a colliding start
        state cannot be reliably detected from a failed plan_async()
        response. The raw /plan_kinematic_path service pymoveit2 calls
        only ever returns the generic MoveItErrorCodes.FAILURE when a
        PlanningRequestAdapter aborts the pipeline, never the specific
        START_STATE_IN_COLLISION value (confirmed directly: that value
        only surfaces through the separate /move_action interface, which
        this bridge does not use, so plan_to_pose()'s own check for it can
        never actually fire through this call path). Checking collision
        directly, proactively, is the reliable way to detect and recover
        from this.

        Returns:
            True if the state was already collision-free, or was
            corrected via retreat. False if it was in collision and could
            not be corrected (e.g. no _last_safe_joint_state known yet);
            the caller should not plan from a possibly still-invalid state.
        """
        joint_state = self._moveit2.joint_state
        if joint_state is None:
            self.status.emit(
                "Could not read current joint state to check collision, "
                "proceeding without this check."
            )
            return True

        if not self._state_validity_client.wait_for_service(timeout_sec=5.0):
            self.status.emit(
                "Could not reach /check_state_validity, proceeding without this check."
            )
            return True

        request = GetStateValidity.Request()
        request.group_name = self._planning_group
        request.robot_state.joint_state = joint_state

        done_event = threading.Event()
        result_holder = {}

        def _on_done(future):
            result_holder["response"] = future.result()
            done_event.set()

        future = self._state_validity_client.call_async(request)
        future.add_done_callback(_on_done)
        if not done_event.wait(timeout=5.0):
            self.status.emit("Timed out waiting for /check_state_validity response.")
            return True

        response = result_holder.get("response")
        if response is None or response.valid:
            return True

        contact_info = ""
        if response.contacts:
            c = response.contacts[0]
            contact_info = f" ({c.contact_body_1} - {c.contact_body_2})"
        self.status.emit(
            f"Current state is in collision{contact_info}, retreating to the "
            "last known safe configuration before planning..."
        )
        return self._retreat_to_last_safe_state()

    def _is_state_valid(self, positions):
        """Queries /check_state_validity for a candidate joint configuration
        that has not been commanded yet, rather than the arm's live current
        state (see ensure_current_state_not_in_collision(), which checks
        the live state instead). Used to verify a correction is actually
        safe before _send_corrective_trajectory() sends it: both
        ensure_current_state_within_bounds() and
        _retreat_to_last_safe_state() compute a candidate fix and send it
        via a raw trajectory that bypasses MoveIt's own planning pipeline
        entirely, so nothing else checks whether that candidate is itself
        collision-free. A bounds fix that only clamps one out-of-range
        joint has no way to know whether the resulting configuration
        happens to bring some other link into the skull.

        Args:
            positions: Full-length joint position list, in
                self._joint_names order.

        Returns:
            True or False if the check completed, None if it could not be
            performed (service unreachable or timed out).
        """
        if not self._state_validity_client.wait_for_service(timeout_sec=5.0):
            self.status.emit(
                "Could not reach /check_state_validity to verify a correction; "
                "proceeding without this check."
            )
            return None

        request = GetStateValidity.Request()
        request.group_name = self._planning_group
        request.robot_state.joint_state.name = list(self._joint_names)
        request.robot_state.joint_state.position = [float(p) for p in positions]

        done_event = threading.Event()
        result_holder = {}

        def _on_done(future):
            result_holder["response"] = future.result()
            done_event.set()

        future = self._state_validity_client.call_async(request)
        future.add_done_callback(_on_done)
        if not done_event.wait(timeout=5.0):
            self.status.emit(
                "Timed out verifying a correction against /check_state_validity; "
                "proceeding without this check."
            )
            return None

        response = result_holder.get("response")
        if response is None:
            return None
        return bool(response.valid)

    def _send_corrective_trajectory(self, positions) -> bool:
        """Blocking. Sends a single-point trajectory with all joints'
        positions, not just the corrected one(s), bypassing MoveIt's
        planning pipeline entirely via a direct FollowJointTrajectory
        action goal to the controller. Same call_async()+threading.Event
        pattern as set_skull_collision_object(): this node's executor is
        already spinning on _ExecutorThread, so this thread waits for the
        response rather than spinning itself.

        Verifies the candidate `positions` with _is_state_valid() first:
        a bounds or collision correction that bypasses MoveIt's own
        collision checking could otherwise send the arm into a new
        collision while "fixing" an unrelated problem, with nothing to
        catch it. Proceeds anyway, with a loud warning, if the check
        itself could not be performed, rather than making the corrective
        mechanism unusable whenever /check_state_validity is briefly
        unreachable.

        Args:
            positions: Full-length joint position list, in
                self._joint_names order.

        Returns:
            True on success.
        """
        valid = self._is_state_valid(positions)
        if valid is False:
            self.status.emit(
                "Refusing to send this correction: the resulting configuration "
                "would itself be in collision. Manual intervention is needed."
            )
            return False
        if valid is None:
            self.status.emit(
                "Could not verify the correction is collision-free; sending it anyway."
            )

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
        """Blocking (called from a worker thread, not the GUI thread; see
        main_window.py). Calls ensure_current_state_within_bounds() and
        ensure_current_state_not_in_collision() proactively before
        planning (see each method's own docstring for why both are
        needed), then retries planning up to _MAX_PLAN_ATTEMPTS times.

        The in-loop check for MoveItErrorCodes.START_STATE_IN_COLLISION
        below is kept as a second line of defense, but the proactive
        collision check above is the one actually doing this job: that
        specific error code is never returned by the plan_async() call
        this method makes (see ensure_current_state_not_in_collision()'s
        docstring), so without the proactive check, a colliding start
        state would previously exhaust every retry attempt identically
        and report a plain, unexplained failure. Any other failure (e.g.
        RRTConnect not finding a path on this random seed) is retried
        as-is, since an identical request can succeed on a later attempt
        with no other change.

        If every attempt above still fails, _attempt_nudge_recovery() is
        tried once as a last resort: a small move in a different
        direction, in case the arm's own current configuration, not the
        target, is what makes a plan hard to find. If a nudge succeeds,
        the original target is retried up to _NUDGE_RETRY_ATTEMPTS times,
        the same reasoning as _MAX_PLAN_ATTEMPTS above.

        Returns:
            The planned trajectory, or None if every attempt failed.
        """
        self.ensure_current_state_within_bounds()
        self.ensure_current_state_not_in_collision()

        plan_kwargs = {"position": [pose.position.x, pose.position.y, pose.position.z]}
        if self._targeting.enforce_orientation:
            plan_kwargs["quat_xyzw"] = [
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w,
            ]

        already_retreated = False
        attempt = 0
        for attempt in range(1, _MAX_PLAN_ATTEMPTS + 1):
            self.status.emit(f"Planning (attempt {attempt}/{_MAX_PLAN_ATTEMPTS})...")
            start_state = self._moveit2.joint_state
            try:
                trajectory, error_code = self._plan_with_error_code(plan_kwargs)
            except Exception as e:  # noqa: BLE001, reported to the GUI instead of crashing it
                self.status.emit(f"Planning FAILED: {e}")
                trajectory, error_code = None, None

            if trajectory is not None:
                if start_state is not None:
                    self._last_safe_joint_state = start_state
                self.status.emit("Plan ready. Review it before executing.")
                self.plan_ready.emit(trajectory)
                return trajectory

            if error_code == MoveItErrorCodes.START_STATE_IN_COLLISION and not already_retreated:
                already_retreated = True
                self.status.emit(
                    "Start state is in collision -- retreating to the last "
                    "known safe configuration before retrying..."
                )
                if not self._retreat_to_last_safe_state():
                    self.status.emit("Retreat failed. Giving up on this target.")
                    break

        nudge_retry_attempt = 0
        if self._attempt_nudge_recovery():
            # Same reasoning as the main retry loop above: an identical
            # request can succeed on a later attempt with no other change,
            # and from a worker thread this costs nothing but time, so it
            # is worth repeating as many times as a person clicking Try
            # Again by hand would, not just once.
            for nudge_retry_attempt in range(1, _NUDGE_RETRY_ATTEMPTS + 1):
                self.status.emit(
                    f"Retrying the original target after the nudge "
                    f"(attempt {nudge_retry_attempt}/{_NUDGE_RETRY_ATTEMPTS})..."
                )
                try:
                    trajectory, _error_code = self._plan_with_error_code(plan_kwargs)
                except Exception as e:  # noqa: BLE001
                    self.status.emit(f"Planning FAILED: {e}")
                    trajectory = None
                if trajectory is not None:
                    self.status.emit("Plan ready. Review it before executing.")
                    self.plan_ready.emit(trajectory)
                    return trajectory

        self.status.emit(
            f"Planning FAILED after {attempt + nudge_retry_attempt} attempt(s)."
        )
        self.plan_ready.emit(None)
        return None

    def plan_to_joint_positions(self, joint_positions, max_attempts=3):
        """Plans a joint-space goal (e.g. the robot's configured home
        configuration), as opposed to plan_to_pose()'s Cartesian goal.
        Reuses the same proactive bounds/collision checks and RRTConnect-
        retry reasoning as plan_to_pose() (an identical request can
        succeed on a later attempt with no other change), but skips its
        skull-specific nudge-recovery fallback: a joint-space goal like
        home is not expected to sit near a collision boundary the way a
        picked surface target can, so that complexity isn't needed here.

        Returns:
            The planned trajectory, or None if every attempt failed.
        """
        self.ensure_current_state_within_bounds()
        self.ensure_current_state_not_in_collision()

        plan_kwargs = {
            "joint_positions": [float(p) for p in joint_positions],
            "joint_names": list(self._joint_names),
        }
        for attempt in range(1, max_attempts + 1):
            self.status.emit(f"Planning to home (attempt {attempt}/{max_attempts})...")
            try:
                trajectory, _error_code = self._plan_with_error_code(plan_kwargs)
            except Exception as e:  # noqa: BLE001, reported to the GUI instead of crashing it
                self.status.emit(f"Planning to home FAILED: {e}")
                trajectory = None
            if trajectory is not None:
                return trajectory
        self.status.emit(f"Planning to home FAILED after {max_attempts} attempt(s).")
        return None

    def reset_to_home(self):
        """Blocking (call from a worker thread): plans and executes a move
        to the robot's configured home_joint_positions. Used before
        loading a new mesh, so the arm starts from a known, obstacle-
        agnostic configuration instead of wherever a previous target
        happened to leave it, before that mesh's own collision geometry
        replaces whatever was there.

        Returns:
            True if the move succeeded, False if planning or execution
            failed.
        """
        trajectory = self.plan_to_joint_positions(self._home_joint_positions)
        if trajectory is None:
            return False
        return self.execute(trajectory)

    def _attempt_nudge_recovery(self) -> bool:
        """Called once, after plan_to_pose()'s normal retry attempts are
        all exhausted. Tries a small (_NUDGE_MAGNITUDE_M) move in each of
        the 10 _NUDGE_DIRECTIONS in turn, via the normal, collision-checked
        planning pipeline, stopping at and executing the first one that
        plans successfully.

        Returns:
            True if a nudge was successfully planned and executed. False
            if every direction failed, in which case the caller's own
            normal failure path runs unchanged.
        """
        current = self.get_current_end_effector_pose()
        if current is None:
            self.status.emit("Could not read current position for nudge recovery.")
            return False

        # A 1cm move is a far simpler problem than the original target, so
        # it should not need the full allowed_planning_time (30s) to
        # determine feasibility; without this override, a hard nudge
        # direction can itself burn the full 30s, and with 10 directions
        # that made the whole recovery step cost up to 300s in the worst
        # case (confirmed directly). Restored in `finally` so a later,
        # normal plan_to_pose() call always gets the real configured value
        # back, regardless of how this loop exits.
        original_planning_time = self._moveit2.allowed_planning_time
        self._moveit2.allowed_planning_time = _NUDGE_PLANNING_TIME_S
        try:
            self.status.emit("Still stuck after normal retries, trying a small nudge...")
            for dx, dy, dz in _NUDGE_DIRECTIONS:
                nudge_kwargs = {"position": [
                    current[0] + dx * _NUDGE_MAGNITUDE_M,
                    current[1] + dy * _NUDGE_MAGNITUDE_M,
                    current[2] + dz * _NUDGE_MAGNITUDE_M,
                ]}
                try:
                    trajectory, _error_code = self._plan_with_error_code(nudge_kwargs)
                except Exception as e:  # noqa: BLE001
                    self.status.emit(f"Nudge planning FAILED: {e}")
                    continue
                if trajectory is None:
                    continue
                if self.execute(trajectory):
                    self.status.emit(
                        f"Nudged {_NUDGE_MAGNITUDE_M * 1000:.0f}mm toward "
                        f"({dx:.2f}, {dy:.2f}, {dz:.2f})."
                    )
                    return True

            self.status.emit("Could not find any safe nudge direction.")
            return False
        finally:
            self._moveit2.allowed_planning_time = original_planning_time

    def _plan_with_error_code(self, plan_kwargs):
        """One planning attempt, returning the real MoveItErrorCodes value
        alongside the trajectory. Distinct from calling self._moveit2.plan()
        directly: that method spins the node itself while waiting
        (rclpy.spin_once in a loop), which is unsafe here since
        _executor_thread already spins this node continuously (see
        __init__), and it discards the specific error code, logging it and
        returning bare None either way.

        Args:
            plan_kwargs: Keyword arguments forwarded to MoveIt2.plan_async().

        Returns:
            (trajectory, error_code): trajectory is a JointTrajectory on
            success, else None; error_code is the MoveItErrorCodes.val,
            or None if the request could not even be sent.
        """
        future = self._moveit2.plan_async(**plan_kwargs)
        if future is None:
            return None, None

        done_event = threading.Event()
        future.add_done_callback(lambda _f: done_event.set())
        timeout = self._targeting.planning_time_s + 10.0
        if not done_event.wait(timeout=timeout):
            self.status.emit("Timed out waiting for the planning service response.")
            return None, None

        response = future.result().motion_plan_response
        error_code = response.error_code.val
        if error_code == MoveItErrorCodes.SUCCESS:
            return response.trajectory.joint_trajectory, error_code
        self.status.emit(f"Planning failed: {enum_to_str(MoveItErrorCodes, error_code)}")
        return None, error_code

    def _retreat_to_last_safe_state(self) -> bool:
        """Sends the arm directly to _last_safe_joint_state, bypassing
        MoveIt's planning pipeline (see _send_corrective_trajectory), the
        same way ensure_current_state_within_bounds() recovers from an
        out-of-bounds start state. This is the realization of "back off
        and find a new path": a fresh MoveIt planning request cannot do
        this itself, since CheckStartStateCollision rejects any request,
        including one whose goal is to retreat, while the current state
        is in collision.

        Returns:
            True if a safe state was known and the retreat succeeded.
        """
        if self._last_safe_joint_state is None:
            self.status.emit("No previously known safe joint state to retreat to.")
            return False

        current = dict(zip(
            self._last_safe_joint_state.name, self._last_safe_joint_state.position
        ))
        try:
            positions = [current[name] for name in self._joint_names]
        except KeyError:
            self.status.emit("Saved safe state is missing one or more joints; cannot retreat.")
            return False

        return self._send_corrective_trajectory(positions)

    def execute(self, trajectory):
        """This call blocks and must be called from a worker thread. It
        executes an already-planned trajectory and waits for completion."""
        self.status.emit("Executing...")
        try:
            self._moveit2.execute(trajectory)
            success = self._wait_for_execution()
        except Exception as e:  # noqa: BLE001
            self.status.emit(f"Execution FAILED: {e}")
            success = False

        self.status.emit("Execution succeeded." if success else "Execution FAILED.")
        self.execute_finished.emit(success)
        return success

    def _wait_for_execution(self, timeout: float = 60.0) -> bool:
        """Polls MoveIt2.query_state() instead of calling
        MoveIt2.wait_until_executed(), which spins self._node itself in a
        blocking loop (rclpy.spin_once) from the calling thread. That is a
        second, concurrent spin of the same node _executor_thread already
        spins in the background, and it corrupts the executor's future
        tracking for the rest of the session: every plan_async() call
        issued afterward silently loses its response's done callback and
        times out, even though move_group itself answers correctly and
        quickly. query_state() only reads mutex-protected flags that the
        already-running executor thread updates via its own callbacks, so
        polling it needs no spin call of its own.

        Args:
            timeout: Seconds to wait before giving up.

        Returns:
            True if the motion finished successfully within timeout.
        """
        if self._moveit2.query_state() == MoveIt2State.IDLE:
            self.status.emit("No motion is in progress; the execute request may have been rejected.")
            return False

        deadline = time.monotonic() + timeout
        while self._moveit2.query_state() != MoveIt2State.IDLE:
            if time.monotonic() > deadline:
                self.status.emit("Timed out waiting for execution to finish.")
                return False
            time.sleep(0.05)
        return self._moveit2.motion_suceeded

    def cancel_execution(self):
        """Best-effort request to stop a currently-executing trajectory
        immediately, via MoveIt2.cancel_execution() (publishes "stop" on
        /trajectory_execution_event, the same topic move_group's own
        trajectory_execution_manager listens on). A no-op, not an error,
        if nothing is currently executing."""
        self._moveit2.cancel_execution()

    def set_skull_collision_object(self, vertices_base_frame, triangles):
        """Blocking (call from a worker thread). Publishes (ADD, which
        also replaces any existing object with the same id) a MoveIt
        collision object named "skull_mesh" via /apply_planning_scene, so
        the planner avoids it, not just something drawn in the GUI's own
        PyVista view, which move_group has no knowledge of otherwise.

        Uses call_async() plus a threading.Event, not
        rclpy.spin_until_future_complete(): this node's executor is
        already spinning on _ExecutorThread (see __init__), and spinning
        the same node from a second thread concurrently would be unsafe.
        The already-running executor services this call's response; this
        thread only waits for it.

        Args:
            vertices_base_frame: (N, 3) points, already in base_frame (see
                Registration.transform_points_to_base_frame()).
            triangles: (M, 3) int indices into vertices.

        Returns:
            True on success.
        """
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
        """Removes the skull collision object. ADD already replaces by id,
        so this is mainly for an explicit "no skull" state."""
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

    def get_skull_collision_pose(self):
        """Blocking (call from a worker thread). Reads the skull collision
        object's current `pose` field from the live planning scene.

        set_skull_collision_object() always publishes with an identity
        pose (the registration transform is baked directly into the
        vertex coordinates instead), so a non-identity pose here can only
        have come from something else moving the object afterward, e.g. a
        drag + Publish in RViz's Scene Objects panel. That offset is
        otherwise invisible to this GUI: it changes what the planner
        avoids without changing anything the GUI itself knows about,
        which is what makes an already-picked target's pose go stale
        relative to the true collision geometry (see main_window.py's
        _on_sync_scene_pose_clicked()).

        Returns:
            (position_xyz, quaternion_xyzw) tuple, or None if the object
            doesn't exist, the pose is identity (nothing to sync), or the
            service could not be reached in time.
        """
        if not self._get_planning_scene_client.wait_for_service(timeout_sec=5.0):
            self.status.emit("Could not reach /get_planning_scene service.")
            return None

        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        )

        done_event = threading.Event()
        result_holder = {}

        def _on_done(future):
            result_holder["response"] = future.result()
            done_event.set()

        future = self._get_planning_scene_client.call_async(request)
        future.add_done_callback(_on_done)
        if not done_event.wait(timeout=10.0):
            self.status.emit("Timed out waiting for /get_planning_scene response.")
            return None

        response = result_holder.get("response")
        if response is None:
            return None

        for obj in response.scene.world.collision_objects:
            if obj.id != _SKULL_COLLISION_OBJECT_ID:
                continue
            if _is_identity_pose(obj.pose):
                return None
            p, q = obj.pose.position, obj.pose.orientation
            return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
        return None

    def _on_monitored_planning_scene(self, msg: PlanningScene):
        """Runs on the ROS executor thread for every /monitored_planning_scene
        message. /monitored_planning_scene only carries diffs, not the full
        scene on every message, so a change to some other part of the scene
        will not mention skull_mesh at all -- this only reacts to messages
        that actually include it.

        Not a full replacement for get_skull_collision_pose()/the Sync From
        Scene Move button: a late-joining subscriber (e.g. this GUI
        restarting while the skull was already moved before it came back
        up) will not see that pre-existing offset until another change
        happens to publish it again. The button still exists for that case,
        and as an explicit, on-demand fallback.
        """
        for obj in msg.world.collision_objects:
            if obj.id != _SKULL_COLLISION_OBJECT_ID:
                continue
            if obj.operation == CollisionObject.REMOVE or _is_identity_pose(obj.pose):
                return
            p, q = obj.pose.position, obj.pose.orientation
            self.scene_skull_moved.emit(((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
            return
