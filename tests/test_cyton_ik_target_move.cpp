#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
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

constexpr const char* DEVICE_NAME = "/dev/ttyUSB0";
constexpr int BAUD_RATE = 1000000;
constexpr float PROTOCOL_VERSION = 1.0F;

constexpr uint16_t MOVING_SPEED = 10;

/*
 * Reduced from 10 to 5 ticks for tighter final positioning.
 */
constexpr int TOLERANCE = 15;
constexpr int TIMEOUT_SECONDS = 30;

/*
 * Fixed end-effector orientation.
 */
constexpr double FIXED_ROLL = 0.246037;
constexpr double FIXED_PITCH = -0.019880;
constexpr double FIXED_YAW = 0.341417;

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

double applyHorizontalDistanceCorrection(double desiredDx)
{
    constexpr double HORIZONTAL_CORRECTION = 0.014; // 14 mm

    if (desiredDx > 0.0)
    {
        return desiredDx + HORIZONTAL_CORRECTION;
    }

    if (desiredDx < 0.0)
    {
        return desiredDx - HORIZONTAL_CORRECTION;
    }

    return 0.0;
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
            ticksToRadians(
                joint,
                joint.minTick
            );

        const double maximumAngle =
            ticksToRadians(
                joint,
                joint.maxTick
            );

        lowerLimits(i) =
            std::min(
                minimumAngle,
                maximumAngle
            );

        upperLimits(i) =
            std::max(
                minimumAngle,
                maximumAngle
            );
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

    std::cout
        << "\nCurrent motor positions:\n";

    for (int id : motorIds)
    {
        uint16_t position = 0;

        if (!motor.readPosition(
                id,
                position
            ))
        {
            std::cerr
                << "Failed to read motor "
                << id
                << ".\n";

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

bool readActualMotorTicks(
    DynamixelMotor& motor,
    const std::vector<int>& motorIds,
    std::vector<uint16_t>& actualTicks
)
{
    actualTicks.clear();
    actualTicks.reserve(motorIds.size());

    for (int id : motorIds)
    {
        uint16_t position = 0;

        if (!motor.readPosition(
                id,
                position
            ))
        {
            std::cerr
                << "Failed to read final position for motor "
                << id
                << ".\n";

            actualTicks.clear();
            return false;
        }

        if (!motor.isPositionSafe(
                id,
                position
            ))
        {
            std::cerr
                << "Final position for motor "
                << id
                << " is outside its safe range: "
                << position
                << " ticks.\n";

            actualTicks.clear();
            return false;
        }

        actualTicks.push_back(position);
    }

    return actualTicks.size() == JOINT_COUNT;
}

KDL::JntArray ticksToJointArray(
    const std::vector<uint16_t>& ticks
)
{
    if (ticks.size() != JOINT_COUNT)
    {
        throw std::runtime_error(
            "Expected seven motor positions."
        );
    }

    if (jointCalibrations.size() < JOINT_COUNT)
    {
        throw std::runtime_error(
            "Fewer than seven arm calibrations were found."
        );
    }

    KDL::JntArray joints(JOINT_COUNT);

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        joints(i) =
            ticksToRadians(
                jointCalibrations[i],
                static_cast<int>(ticks[i])
            );
    }

    return joints;
}

KDL::Frame calculatePoseFromActualTicks(
    const KDL::Chain& chain,
    const std::vector<uint16_t>& ticks
)
{
    const KDL::JntArray joints =
        ticksToJointArray(ticks);

    KDL::ChainFkSolverPos_recursive fkSolver(
        chain
    );

    KDL::Frame pose;

    if (fkSolver.JntToCart(
            joints,
            pose
        ) < 0)
    {
        throw std::runtime_error(
            "Forward kinematics failed for actual motor ticks."
        );
    }

    return pose;
}

void printMotorTickComparison(
    const std::string& poseName,
    const std::vector<uint16_t>& targetTicks,
    const std::vector<uint16_t>& actualTicks
)
{
    if (targetTicks.size() != JOINT_COUNT ||
        actualTicks.size() != JOINT_COUNT)
    {
        throw std::runtime_error(
            "Invalid motor tick comparison sizes."
        );
    }

    std::cout
        << "\n"
        << poseName
        << " target versus actual ticks:\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const int error =
            static_cast<int>(targetTicks[i]) -
            static_cast<int>(actualTicks[i]);

        std::cout
            << "Motor " << i
            << " | Target: "
            << targetTicks[i]
            << " | Actual: "
            << actualTicks[i]
            << " | Signed error: "
            << error
            << " ticks"
            << " | Absolute error: "
            << std::abs(error)
            << " ticks\n";
    }
}

void printPose(
    const std::string& poseName,
    const KDL::Frame& pose
)
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
        << "\n"
        << poseName
        << ":\n"
        << std::fixed
        << std::setprecision(6)
        << "  x = " << pose.p.x() << " m\n"
        << "  y = " << pose.p.y() << " m\n"
        << "  z = " << pose.p.z() << " m\n"
        << "  roll  = " << roll << " rad\n"
        << "  pitch = " << pitch << " rad\n"
        << "  yaw   = " << yaw << " rad\n";
}

void printDisplacementAnalysis(
    const PoseInput& requestedPoseA,
    const PoseInput& requestedPoseB,
    const KDL::Frame& actualPoseA,
    const KDL::Frame& actualPoseB
)
{
    const double commandedDx =
        requestedPoseB.x -
        requestedPoseA.x;

    const double commandedDy =
        requestedPoseB.y -
        requestedPoseA.y;

    const double commandedDz =
        requestedPoseB.z -
        requestedPoseA.z;

    const double commandedDistance =
        std::sqrt(
            commandedDx * commandedDx +
            commandedDy * commandedDy +
            commandedDz * commandedDz
        );

    const double actualDx =
        actualPoseB.p.x() -
        actualPoseA.p.x();

    const double actualDy =
        actualPoseB.p.y() -
        actualPoseA.p.y();

    const double actualDz =
        actualPoseB.p.z() -
        actualPoseA.p.z();

    const double actualDistance =
        std::sqrt(
            actualDx * actualDx +
            actualDy * actualDy +
            actualDz * actualDz
        );

    const double errorDx =
        actualDx -
        commandedDx;

    const double errorDy =
        actualDy -
        commandedDy;

    const double errorDz =
        actualDz -
        commandedDz;

    const double distanceError =
        actualDistance -
        commandedDistance;

    std::cout
        << "\n========================================\n"
        << "ACTUAL FK DISPLACEMENT ANALYSIS\n"
        << "========================================\n"
        << std::fixed
        << std::setprecision(6)

        << "\nCommanded displacement:\n"
        << "  dx = " << commandedDx
        << " m (" << commandedDx * 1000.0 << " mm)\n"
        << "  dy = " << commandedDy
        << " m (" << commandedDy * 1000.0 << " mm)\n"
        << "  dz = " << commandedDz
        << " m (" << commandedDz * 1000.0 << " mm)\n"
        << "  straight-line distance = "
        << commandedDistance
        << " m (" << commandedDistance * 1000.0
        << " mm)\n"

        << "\nActual displacement calculated from reached ticks:\n"
        << "  dx = " << actualDx
        << " m (" << actualDx * 1000.0 << " mm)\n"
        << "  dy = " << actualDy
        << " m (" << actualDy * 1000.0 << " mm)\n"
        << "  dz = " << actualDz
        << " m (" << actualDz * 1000.0 << " mm)\n"
        << "  straight-line distance = "
        << actualDistance
        << " m (" << actualDistance * 1000.0
        << " mm)\n"

        << "\nDisplacement error according to the KDL model:\n"
        << "  x error = " << errorDx
        << " m (" << errorDx * 1000.0 << " mm)\n"
        << "  y error = " << errorDy
        << " m (" << errorDy * 1000.0 << " mm)\n"
        << "  z error = " << errorDz
        << " m (" << errorDz * 1000.0 << " mm)\n"
        << "  distance error = "
        << distanceError
        << " m (" << distanceError * 1000.0
        << " mm)\n";

    if (commandedDistance > 0.0)
    {
        const double percentage =
            actualDistance /
            commandedDistance *
            100.0;

        std::cout
            << "  achieved distance = "
            << percentage
            << "% of commanded distance\n";
    }

    std::cout
        << "\nInterpretation:\n"
        << "  If actual FK displacement is also short, motor tracking\n"
        << "  and stopping tolerance are contributing to the error.\n"
        << "  If actual FK displacement is correct but the physical\n"
        << "  measurement is short, the KDL model, joint calibration,\n"
        << "  end-effector offset, or measurement reference is wrong.\n"
        << "========================================\n";
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
            << "  yaw   = " << FIXED_YAW << '\n'
            << "Motor tolerance: "
            << TOLERANCE
            << " ticks\n";

        /*
         * Current test:
         * Pose B is exactly 0.150000 m farther in x.
         */
        const PoseInput poseA = {
            -0.036864, // x
            0.019448, // y
            0.646952  // z
        };

        constexpr double DESIRED_X_MOVEMENT = 0.1;

        const double correctedXMovement =
            applyHorizontalDistanceCorrection(
                DESIRED_X_MOVEMENT
            );

        const PoseInput poseB = {
            poseA.x + correctedXMovement,
            poseA.y,
            poseA.z
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
         * Seed Pose B using Pose A's solution.
         * This encourages a nearby continuous IK solution.
         */
        KDL::JntArray seedB =
            solutionA;

        KDL::JntArray solutionB(
            JOINT_COUNT
        );

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

            motor.emergencyShutdown(
                motorIds
            );

            return 1;
        }

        std::vector<uint16_t> actualPoseATicks;

        if (!readActualMotorTicks(
                motor,
                motorIds,
                actualPoseATicks
            ))
        {
            std::cerr
                << "Could not record actual Pose A ticks.\n";

            motor.emergencyShutdown(
                motorIds
            );

            return 1;
        }

        const KDL::Frame actualPoseA =
            calculatePoseFromActualTicks(
                chain,
                actualPoseATicks
            );

        std::cout
            << "\nPose A reached.\n";

        printMotorTickComparison(
            "Pose A",
            poseATicks,
            actualPoseATicks
        );

        printPose(
            "Actual Pose A calculated from reached ticks",
            actualPoseA
        );

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

            motor.emergencyShutdown(
                motorIds
            );

            return 1;
        }

        std::vector<uint16_t> actualPoseBTicks;

        if (!readActualMotorTicks(
                motor,
                motorIds,
                actualPoseBTicks
            ))
        {
            std::cerr
                << "Could not record actual Pose B ticks.\n";

            motor.emergencyShutdown(
                motorIds
            );

            return 1;
        }

        const KDL::Frame actualPoseB =
            calculatePoseFromActualTicks(
                chain,
                actualPoseBTicks
            );

        std::cout
            << "\nPose B reached.\n"
            << "Torque remains enabled to hold the arm.\n";

        printMotorTickComparison(
            "Pose B",
            poseBTicks,
            actualPoseBTicks
        );

        printPose(
            "Actual Pose B calculated from reached ticks",
            actualPoseB
        );

        printDisplacementAnalysis(
            poseA,
            poseB,
            actualPoseA,
            actualPoseB
        );

        waitForEnter(
            "\nPress Enter to disable torque and end..."
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