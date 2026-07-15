#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>

#include <trac_ik/trac_ik.hpp>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

namespace
{
constexpr std::size_t JOINT_COUNT = 7;

struct JointDescription
{
    const char* jointName;
    const char* linkName;

    double x;
    double y;
    double z;

    double axisX;
    double axisY;
    double axisZ;
};

const std::array<JointDescription, JOINT_COUNT> CYTON_JOINTS = {{
    {
        "shoulder_roll_joint",
        "shoulder_roll",
        0.0, 0.0, 0.05315,
        0.0, 0.0, 1.0
    },
    {
        "shoulder_pitch_joint",
        "shoulder_pitch",
        0.0205, 0.0, 0.12435,
        1.0, 0.0, 0.0
    },
    {
        "shoulder_yaw_joint",
        "shoulder_yaw",
        -0.0215, -0.0205, 0.1255,
        0.0, -1.0, 0.0
    },
    {
        "elbow_pitch_joint",
        "elbow_pitch",
        0.018, 0.0206, 0.1158,
        1.0, 0.0, 0.0
    },
    {
        "elbow_yaw_joint",
        "elbow_yaw",
        -0.0171, -0.018, 0.09746,
        0.0, -1.0, 0.0
    },
    {
        "wrist_pitch_joint",
        "wrist_pitch",
        0.02626, 0.018, 0.0718,
        1.0, 0.0, 0.0
    },
    {
        "wrist_roll_joint",
        "wrist_roll",
        -0.026255, 0.0, 0.051425,
        0.0, 0.0, 1.0
    }
}};

void waitForEnter(const std::string& message)
{
    std::cout << message;
    std::cin.get();
}

KDL::Chain createCytonChain()
{
    KDL::Chain chain;

    for (const JointDescription& description : CYTON_JOINTS)
    {
        const KDL::Vector origin(
            description.x,
            description.y,
            description.z
        );

        const KDL::Vector axis(
            description.axisX,
            description.axisY,
            description.axisZ
        );

        chain.addSegment(
            KDL::Segment(
                description.linkName,
                KDL::Joint(
                    description.jointName,
                    origin,
                    axis,
                    KDL::Joint::RotAxis
                ),
                KDL::Frame(origin)
            )
        );
    }

    chain.addSegment(
        KDL::Segment(
            "virtual_endeffector",
            KDL::Joint(
                "virtual_endeffector_joint",
                KDL::Joint::None
            ),
            KDL::Frame(
                KDL::Vector(
                    -0.002316,
                    0.0079,
                    0.079425
                )
            )
        )
    );

    return chain;
}

void buildJointLimits(
    KDL::JntArray& lowerLimits,
    KDL::JntArray& upperLimits
)
{
    if (jointCalibrations.size() < JOINT_COUNT)
    {
        throw std::runtime_error(
            "Fewer than seven arm calibrations were found."
        );
    }

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& joint =
            jointCalibrations[i];

        const double angleAtMinimum =
            ticksToRadians(joint, joint.minTick);

        const double angleAtMaximum =
            ticksToRadians(joint, joint.maxTick);

        lowerLimits(i) =
            std::min(angleAtMinimum, angleAtMaximum);

        upperLimits(i) =
            std::max(angleAtMinimum, angleAtMaximum);
    }
}

void printPose(const KDL::Frame& pose)
{
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    pose.M.GetRPY(
        roll,
        pitch,
        yaw
    );

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "Position:\n"
        << "  x = " << pose.p.x() << " m\n"
        << "  y = " << pose.p.y() << " m\n"
        << "  z = " << pose.p.z() << " m\n"
        << "Orientation:\n"
        << "  roll  = " << roll << " rad\n"
        << "  pitch = " << pitch << " rad\n"
        << "  yaw   = " << yaw << " rad\n";
}

bool convertSolutionToTicks(
    const KDL::JntArray& solution,
    std::vector<uint16_t>& targetTicks
)
{
    targetTicks.clear();
    targetTicks.reserve(JOINT_COUNT);

    bool allSafe = true;

    std::cout << "\nSolved joint values:\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& joint =
            jointCalibrations[i];

        const double radians =
            solution(i);

        const int tick =
            radiansToTicks(
                joint,
                radians
            );

        const bool safe =
            tick >= joint.minTick &&
            tick <= joint.maxTick &&
            tick >= 0 &&
            tick <= 4095;

        std::cout
            << "Motor " << joint.id
            << " | "
            << std::fixed
            << std::setprecision(6)
            << radians
            << " rad"
            << " | target "
            << tick
            << " ticks"
            << " | safe range ["
            << joint.minTick
            << ", "
            << joint.maxTick
            << "]"
            << " | "
            << (safe ? "SAFE" : "UNSAFE")
            << '\n';

        if (!safe)
        {
            allSafe = false;
            continue;
        }

        targetTicks.push_back(
            static_cast<uint16_t>(tick)
        );
    }

    return allSafe &&
           targetTicks.size() == JOINT_COUNT;
}

bool checkStartingPositions(
    DynamixelMotor& motor,
    const std::vector<int>& motorIds
)
{
    bool allSafe = true;

    std::cout << "\nCurrent motor positions:\n";

    for (int id : motorIds)
    {
        uint16_t position = 0;

        if (!motor.readPosition(id, position))
        {
            std::cerr
                << "Failed to read motor "
                << id << ".\n";

            allSafe = false;
            continue;
        }

        const bool safe =
            motor.isPositionSafe(
                id,
                position
            );

        std::cout
            << "Motor " << id
            << " | current "
            << position
            << " ticks"
            << " | "
            << (safe ? "SAFE" : "OUTSIDE SAFE RANGE")
            << '\n';

        if (!safe)
        {
            allSafe = false;
        }
    }

    return allSafe;
}

void disableAll(
    DynamixelMotor& motor,
    const std::vector<int>& motorIds
)
{
    for (int id : motorIds)
    {
        motor.disableTorque(id);
    }
}
}

int main()
{
    constexpr const char* DEVICE_NAME =
        "/dev/ttyUSB1";

    constexpr int BAUD_RATE = 1000000;
    constexpr float PROTOCOL_VERSION = 1.0;

    constexpr uint16_t MOVING_SPEED = 10;
    constexpr int TOLERANCE = 15;
    constexpr int TIMEOUT_SECONDS = 30;

    const std::vector<int> motorIds = {
        0, 1, 2, 3, 4, 5, 6
    };

    try
    {
        KDL::Chain chain =
            createCytonChain();

        if (chain.getNrOfJoints() != JOINT_COUNT)
        {
            std::cerr
                << "Expected seven joints but found "
                << chain.getNrOfJoints()
                << ".\n";

            return 1;
        }

        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        const double roll = 0.246037;
        const double pitch = -0.019880;
        const double yaw = 0.341417;

        std::cout
            << "Enter the desired end-effector pose.\n"
            << "Position is measured from base_link in metres.\n"
            << "Orientation is roll, pitch, yaw in radians.\n\n";

        std::cout << "x: ";
        std::cin >> x;

        std::cout << "y: ";
        std::cin >> y;

        std::cout << "z: ";
        std::cin >> z;

        if (!std::cin)
        {
            std::cerr
                << "Invalid numerical input.\n";

            return 1;
        }

        std::cin.ignore(
            std::numeric_limits<std::streamsize>::max(),
            '\n'
        );

        const KDL::Frame targetPose(
            KDL::Rotation::RPY(
                roll,
                pitch,
                yaw
            ),
            KDL::Vector(
                x,
                y,
                z
            )
        );

        std::cout << "\nRequested pose:\n";
        printPose(targetPose);

        KDL::JntArray lowerLimits(JOINT_COUNT);
        KDL::JntArray upperLimits(JOINT_COUNT);

        buildJointLimits(
            lowerLimits,
            upperLimits
        );

        KDL::JntArray seed(JOINT_COUNT);
        KDL::JntArray solution(JOINT_COUNT);

        for (std::size_t i = 0; i < JOINT_COUNT; ++i)
        {
            seed(i) = 0.0;
            solution(i) = 0.0;
        }

        TRAC_IK::TRAC_IK solver(
            chain,
            lowerLimits,
            upperLimits,
            0.50,
            1e-5,
            TRAC_IK::Distance
        );

        const int ikResult =
            solver.CartToJnt(
                seed,
                targetPose,
                solution
            );

        if (ikResult < 0)
        {
            std::cerr
                << "\nTRAC-IK could not find a solution.\n"
                << "The pose may be unreachable or outside "
                << "the joint limits.\n";

            return 1;
        }

        KDL::ChainFkSolverPos_recursive fkSolver(
            chain
        );

        KDL::Frame verifiedPose;

        if (fkSolver.JntToCart(
                solution,
                verifiedPose
            ) < 0)
        {
            std::cerr
                << "Forward-kinematics verification failed.\n";

            return 1;
        }

        const KDL::Twist poseError =
            KDL::diff(
                verifiedPose,
                targetPose
            );

        std::cout
            << "\nTRAC-IK found a solution.\n"
            << "Position error: "
            << poseError.vel.Norm()
            << " m\n"
            << "Orientation error: "
            << poseError.rot.Norm()
            << " rad\n";

        if (poseError.vel.Norm() > 1e-4 ||
            poseError.rot.Norm() > 1e-4)
        {
            std::cerr
                << "IK verification error is too large.\n";

            return 1;
        }

        std::vector<uint16_t> targetTicks;

        if (!convertSolutionToTicks(
                solution,
                targetTicks
            ))
        {
            std::cerr
                << "\nMovement cancelled because at least one "
                << "target is outside its safe range.\n";

            return 1;
        }

        std::cout
            << "\nMotor 7 is the gripper and will not move.\n";

        DynamixelMotor motor(
            DEVICE_NAME,
            BAUD_RATE,
            PROTOCOL_VERSION
        );

        if (!motor.connect())
        {
            return 1;
        }

        if (!checkStartingPositions(
                motor,
                motorIds
            ))
        {
            std::cerr
                << "\nMovement cancelled because at least one "
                << "current motor position is unsafe.\n";

            motor.disconnect();
            return 1;
        }

        std::cout
            << "\nMovement summary:\n";

        for (std::size_t i = 0;
             i < motorIds.size();
             ++i)
        {
            uint16_t currentPosition = 0;

            if (!motor.readPosition(
                    motorIds[i],
                    currentPosition
                ))
            {
                std::cerr
                    << "Failed to reread motor "
                    << motorIds[i] << ".\n";

                motor.disconnect();
                return 1;
            }

            const int movement =
                static_cast<int>(targetTicks[i]) -
                static_cast<int>(currentPosition);

            std::cout
                << "Motor " << motorIds[i]
                << ": "
                << currentPosition
                << " -> "
                << targetTicks[i]
                << " | change "
                << movement
                << " ticks\n";
        }

        std::cout
            << "\nMoving speed: "
            << MOVING_SPEED
            << '\n';

        waitForEnter(
            "\nCheck all printed targets and clear the arm.\n"
            "Press Enter to enable torque and move motors 0-6..."
        );

        const bool movementSucceeded =
            motor.moveJointsSafely(
                motorIds,
                targetTicks,
                MOVING_SPEED,
                TOLERANCE,
                TIMEOUT_SECONDS,
                true
            );

        if (!movementSucceeded)
        {
            std::cerr
                << "\nMovement failed. Disabling torque.\n";

            motor.emergencyShutdown(motorIds);
            return 1;
        }

        std::cout
            << "\nTarget pose reached.\n"
            << "Torque remains enabled to hold the arm.\n";

        waitForEnter(
            "Press Enter to disable torque and end..."
        );

        disableAll(
            motor,
            motorIds
        );

        motor.disconnect();

        std::cout
            << "Torque disabled. Program complete.\n";

        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "Test failed: "
            << exception.what()
            << '\n';

        return 1;
    }
}