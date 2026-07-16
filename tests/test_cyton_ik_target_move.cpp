#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

constexpr const char* DEVICE_NAME = "/dev/ttyUSB1";
constexpr int BAUD_RATE = 1000000;
constexpr float PROTOCOL_VERSION = 1.0F;

constexpr uint16_t MOVING_SPEED = 10;
constexpr int TOLERANCE = 15;
constexpr int TIMEOUT_SECONDS = 30;

/*
 * Fixed end-effector orientation.
 */
constexpr double FIXED_ROLL = 0;
constexpr double FIXED_PITCH = 0;
constexpr double FIXED_YAW = 0;

struct PoseInput
{
    double x;
    double y;
    double z;
};

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

KDL::Frame makeTargetFrame(const PoseInput& pose)
{
    return KDL::Frame(
        KDL::Rotation::RPY(
            FIXED_ROLL,
            FIXED_PITCH,
            FIXED_YAW
        ),
        KDL::Vector(
            pose.x,
            pose.y,
            pose.z
        )
    );
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

        const double minimumAngle =
            ticksToRadians(joint, joint.minTick);

        const double maximumAngle =
            ticksToRadians(joint, joint.maxTick);

        lowerLimits(i) =
            std::min(minimumAngle, maximumAngle);

        upperLimits(i) =
            std::max(minimumAngle, maximumAngle);
    }
}

bool convertSolutionToTicks(
    const std::string& poseName,
    const KDL::JntArray& solution,
    std::vector<uint16_t>& targetTicks
)
{
    targetTicks.clear();
    targetTicks.reserve(JOINT_COUNT);

    bool allSafe = true;

    std::cout
        << "\n"
        << poseName
        << " solved motor targets:\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& joint =
            jointCalibrations[i];

        const double radians = solution(i);

        const int tick =
            radiansToTicks(joint, radians);

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
            << " | "
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

bool solvePose(
    const std::string& poseName,
    const KDL::Frame& targetPose,
    KDL::Chain& chain,
    const KDL::JntArray& lowerLimits,
    const KDL::JntArray& upperLimits,
    KDL::JntArray& seed,
    KDL::JntArray& solution,
    std::vector<uint16_t>& targetTicks
)
{
    TRAC_IK::TRAC_IK solver(
        chain,
        lowerLimits,
        upperLimits,
        0.50,
        1e-5,
        TRAC_IK::Distance
    );

    const int result =
        solver.CartToJnt(
            seed,
            targetPose,
            solution
        );

    if (result < 0)
    {
        std::cerr
            << "\nTRAC-IK could not solve "
            << poseName
            << ".\n";

        return false;
    }

    KDL::ChainFkSolverPos_recursive fkSolver(chain);

    KDL::Frame verifiedPose;

    if (fkSolver.JntToCart(
            solution,
            verifiedPose
        ) < 0)
    {
        std::cerr
            << "Forward-kinematics verification failed for "
            << poseName
            << ".\n";

        return false;
    }

    const KDL::Twist error =
        KDL::diff(
            verifiedPose,
            targetPose
        );

    std::cout
        << "\n"
        << poseName
        << " IK verification:\n"
        << "Position error: "
        << error.vel.Norm()
        << " m\n"
        << "Orientation error: "
        << error.rot.Norm()
        << " rad\n";

    if (error.vel.Norm() > 1e-4 ||
        error.rot.Norm() > 1e-4)
    {
        std::cerr
            << poseName
            << " verification error is too large.\n";

        return false;
    }

    return convertSolutionToTicks(
        poseName,
        solution,
        targetTicks
    );
}

bool checkCurrentPositions(
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
            motor.isPositionSafe(id, position);

        std::cout
            << "Motor " << id
            << " | "
            << position
            << " ticks"
            << " | "
            << (safe ? "SAFE" : "UNSAFE")
            << '\n';

        if (!safe)
        {
            allSafe = false;
        }
    }

    return allSafe;
}

void printMovementSummary(
    DynamixelMotor& motor,
    const std::vector<int>& motorIds,
    const std::vector<uint16_t>& targetTicks,
    const std::string& poseName
)
{
    std::cout
        << "\nMovement summary for "
        << poseName
        << ":\n";

    for (std::size_t i = 0; i < motorIds.size(); ++i)
    {
        uint16_t currentPosition = 0;

        if (!motor.readPosition(
                motorIds[i],
                currentPosition
            ))
        {
            throw std::runtime_error(
                "Failed to read current position."
            );
        }

        const int change =
            static_cast<int>(targetTicks[i]) -
            static_cast<int>(currentPosition);

        std::cout
            << "Motor " << motorIds[i]
            << ": "
            << currentPosition
            << " -> "
            << targetTicks[i]
            << " | change "
            << change
            << " ticks\n";
    }
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
    const std::vector<int> motorIds = {
        0, 1, 2, 3, 4, 5, 6
    };

    try
    {
        std::cout
            << "Pose A to Pose B IK movement test\n"
            << "Position values are in metres.\n"
            << "Orientation is fixed at:\n"
            << "  roll  = " << FIXED_ROLL << '\n'
            << "  pitch = " << FIXED_PITCH << '\n'
            << "  yaw   = " << FIXED_YAW << '\n';

        const PoseInput poseA = {
            -0.036864,  // x
            0.019448,  // y
            0.646952   // z
        };

        const PoseInput poseB = {
            -0.002411,  // x
            0.008000,  // y
            0.718910   // z
        };

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

        KDL::JntArray lowerLimits(JOINT_COUNT);
        KDL::JntArray upperLimits(JOINT_COUNT);

        buildJointLimits(
            lowerLimits,
            upperLimits
        );

        KDL::JntArray seedA(JOINT_COUNT);
        KDL::JntArray solutionA(JOINT_COUNT);

        for (std::size_t i = 0; i < JOINT_COUNT; ++i)
        {
            seedA(i) = 0.0;
            solutionA(i) = 0.0;
        }

        std::vector<uint16_t> poseATicks;

        if (!solvePose(
                "Pose A",
                makeTargetFrame(poseA),
                chain,
                lowerLimits,
                upperLimits,
                seedA,
                solutionA,
                poseATicks
            ))
        {
            std::cerr
                << "Pose A is invalid or unreachable.\n";

            return 1;
        }

        /*
         * Seed Pose B using the Pose A solution.
         * This encourages a nearby, continuous configuration.
         */
        KDL::JntArray seedB = solutionA;
        KDL::JntArray solutionB(JOINT_COUNT);

        std::vector<uint16_t> poseBTicks;

        if (!solvePose(
                "Pose B",
                makeTargetFrame(poseB),
                chain,
                lowerLimits,
                upperLimits,
                seedB,
                solutionB,
                poseBTicks
            ))
        {
            std::cerr
                << "Pose B is invalid or unreachable.\n";

            return 1;
        }

        std::cout
            << "\nBoth poses passed IK and motor-limit checks.\n"
            << "Motor 7 will not move.\n";

        DynamixelMotor motor(
            DEVICE_NAME,
            BAUD_RATE,
            PROTOCOL_VERSION
        );

        if (!motor.connect())
        {
            return 1;
        }

        if (!checkCurrentPositions(
                motor,
                motorIds
            ))
        {
            std::cerr
                << "Movement cancelled because a starting "
                << "position is unsafe.\n";

            motor.disconnect();
            return 1;
        }

        printMovementSummary(
            motor,
            motorIds,
            poseATicks,
            "Pose A"
        );

        waitForEnter(
            "\nCheck the Pose A targets and clear the arm.\n"
            "Press Enter to move to Pose A..."
        );

        if (!motor.moveJointsSafely(
                motorIds,
                poseATicks,
                MOVING_SPEED,
                TOLERANCE,
                TIMEOUT_SECONDS,
                true
            ))
        {
            std::cerr
                << "Movement to Pose A failed.\n";

            motor.emergencyShutdown(motorIds);
            return 1;
        }

        std::cout
            << "\nPose A reached successfully.\n";

        printMovementSummary(
            motor,
            motorIds,
            poseBTicks,
            "Pose B"
        );

        waitForEnter(
            "\nCheck the Pose B targets and clear the arm.\n"
            "Press Enter to move from Pose A to Pose B..."
        );

        if (!motor.moveJointsSafely(
                motorIds,
                poseBTicks,
                MOVING_SPEED,
                TOLERANCE,
                TIMEOUT_SECONDS,
                true
            ))
        {
            std::cerr
                << "Movement to Pose B failed.\n";

            motor.emergencyShutdown(motorIds);
            return 1;
        }

        std::cout
            << "\nPose B reached successfully.\n"
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