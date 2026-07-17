#!/usr/bin/env python3

import math
import os
import time

import pybullet as p
import pybullet_data


URDF_PATH = os.path.expanduser(
    "~/cyton_setup/references/cyton_gamma_1500_trac_ik(4).urdf"
)

ARM_JOINT_NAMES = [
    "shoulder_roll_joint",
    "shoulder_pitch_joint",
    "shoulder_yaw_joint",
    "elbow_pitch_joint",
    "elbow_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
]

END_EFFECTOR_LINK_NAME = "virtual_endeffector"
AXIS_LENGTH = 0.08


def find_indices(robot_id):
    joint_map = {}
    link_map = {}

    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)

        joint_name = info[1].decode("utf-8")
        child_link_name = info[12].decode("utf-8")

        joint_map[joint_name] = joint_index
        link_map[child_link_name] = joint_index

        print(
            f"Index {joint_index:2d} | "
            f"joint: {joint_name:30s} | "
            f"child link: {child_link_name}"
        )

    missing_joints = [
        name
        for name in ARM_JOINT_NAMES
        if name not in joint_map
    ]

    if missing_joints:
        raise RuntimeError(
            "Missing joints: " + ", ".join(missing_joints)
        )

    if END_EFFECTOR_LINK_NAME not in link_map:
        raise RuntimeError(
            f'Missing end-effector link: "{END_EFFECTOR_LINK_NAME}"'
        )

    arm_joint_indices = [
        joint_map[name]
        for name in ARM_JOINT_NAMES
    ]

    end_effector_link_index = (
        link_map[END_EFFECTOR_LINK_NAME]
    )

    return arm_joint_indices, end_effector_link_index


def create_joint_sliders(robot_id, joint_indices):
    sliders = []

    for joint_index in joint_indices:
        info = p.getJointInfo(robot_id, joint_index)

        joint_name = info[1].decode("utf-8")
        lower_limit = info[8]
        upper_limit = info[9]

        if lower_limit >= upper_limit:
            lower_limit = -math.pi
            upper_limit = math.pi

        slider_id = p.addUserDebugParameter(
            joint_name,
            lower_limit,
            upper_limit,
            0.0,
        )

        sliders.append(
            (joint_index, slider_id)
        )

    return sliders


def apply_slider_positions(robot_id, sliders):
    for joint_index, slider_id in sliders:
        joint_angle = p.readUserDebugParameter(
            slider_id
        )

        p.resetJointState(
            robot_id,
            joint_index,
            targetValue=joint_angle,
        )


def draw_end_effector_axes(
    robot_id,
    end_effector_link_index,
    line_ids,
):
    state = p.getLinkState(
        robot_id,
        end_effector_link_index,
        computeForwardKinematics=True,
    )

    position = state[4]
    orientation = state[5]

    rotation_matrix = p.getMatrixFromQuaternion(
        orientation
    )

    x_axis = (
        rotation_matrix[0],
        rotation_matrix[3],
        rotation_matrix[6],
    )

    y_axis = (
        rotation_matrix[1],
        rotation_matrix[4],
        rotation_matrix[7],
    )

    z_axis = (
        rotation_matrix[2],
        rotation_matrix[5],
        rotation_matrix[8],
    )

    axes = [
        x_axis,
        y_axis,
        z_axis,
    ]

    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    for axis_index, axis in enumerate(axes):
        endpoint = [
            position[i] + AXIS_LENGTH * axis[i]
            for i in range(3)
        ]

        line_ids[axis_index] = p.addUserDebugLine(
            position,
            endpoint,
            colors[axis_index],
            lineWidth=4,
            replaceItemUniqueId=line_ids[axis_index],
        )

    return position, orientation


def print_end_effector_pose(
    position,
    orientation,
):
    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    print("\nEnd-effector pose")

    print(
        "Position [m]: "
        f"x={position[0]:.6f}, "
        f"y={position[1]:.6f}, "
        f"z={position[2]:.6f}"
    )

    print(
        "Orientation RPY [rad]: "
        f"roll={roll:.6f}, "
        f"pitch={pitch:.6f}, "
        f"yaw={yaw:.6f}"
    )

    print(
        "Orientation RPY [deg]: "
        f"roll={math.degrees(roll):.3f}, "
        f"pitch={math.degrees(pitch):.3f}, "
        f"yaw={math.degrees(yaw):.3f}"
    )


def main():
    if not os.path.isfile(URDF_PATH):
        raise FileNotFoundError(
            f"URDF not found: {URDF_PATH}"
        )

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError(
            "Could not open the PyBullet GUI."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.resetSimulation()

    p.setGravity(
        0,
        0,
        -9.81,
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=1.25,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[
            0,
            0,
            0.35,
        ],
    )

    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(
        URDF_PATH,
        basePosition=[
            0,
            0,
            0,
        ],
        baseOrientation=p.getQuaternionFromEuler(
            [0, 0, 0]
        ),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    print("\nLoaded URDF:")
    print(URDF_PATH)

    print("\nJoints and child links:")

    arm_joint_indices, end_effector_link_index = (
        find_indices(robot_id)
    )

    print(
        "\nEnd-effector link index:",
        end_effector_link_index,
    )

    sliders = create_joint_sliders(
        robot_id,
        arm_joint_indices,
    )

    for joint_index in arm_joint_indices:
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            p.VELOCITY_CONTROL,
            force=0,
        )

    print("\nControls:")
    print("Move the joint sliders in the PyBullet panel.")
    print("Press P to print the end-effector pose.")
    print("Press Q to quit.")
    print("Red = end-effector X axis.")
    print("Green = end-effector Y axis.")
    print("Blue = end-effector Z axis.")

    line_ids = [
        -1,
        -1,
        -1,
    ]

    while p.isConnected():
        apply_slider_positions(
            robot_id,
            sliders,
        )

        position, orientation = (
            draw_end_effector_axes(
                robot_id,
                end_effector_link_index,
                line_ids,
            )
        )

        keyboard = p.getKeyboardEvents()

        if (
            ord("p") in keyboard
            and keyboard[ord("p")]
            & p.KEY_WAS_TRIGGERED
        ):
            print_end_effector_pose(
                position,
                orientation,
            )

        if (
            ord("q") in keyboard
            and keyboard[ord("q")]
            & p.KEY_WAS_TRIGGERED
        ):
            break

        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()