#include <algorithm>
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
#include <kdl/tree.hpp>

#include <kdl_parser/kdl_parser.hpp>

#include <trac_ik/trac_ik.hpp>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

namespace
{
constexpr std::size_t JOINT_COUNT = 7;

constexpr const char* DEVICE_NAME = "/dev/ttyUSB0";
constexpr int BAUD_RATE = 1000000;
constexpr float PROTOCOL_VERSION = 1.0F;

constexpr const char* URDF_PATH =
    "/home/tzeng/cyton_setup/references/cyton_gamma_1500_trac_ik.urdf";

constexpr const char* BASE_LINK = "base_link";
constexpr const char* END_EFFECTOR_LINK = "virtual_endeffector";

constexpr uint16_t MOVING_SPEED = 10;

constexpr int TOLERANCE = 15;
constexpr int TIMEOUT_SECONDS = 30;

constexpr double FIXED_ROLL = 0.246037;
constexpr double FIXED_PITCH = -0.019880;
constexpr double FIXED_YAW = 0.341417;

constexpr double CARTESIAN_WAYPOINT_SPACING = 0.005; // 5 mm

struct PoseInput
{
    double x;
    double y;
    double z;
};

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

KDL::Chain createCytonChainFromUrdf()
{
    KDL::Tree tree;

    if (!kdl_parser::treeFromFile(URDF_PATH, tree))
    {
        throw std::runtime_error(
            std::string("Failed to load KDL tree from URDF: ") +
            URDF_PATH
        );
    }

    KDL::Chain chain;

    if (!tree.getChain(
            BASE_LINK,
            END_EFFECTOR_LINK,
            chain
        ))
    {
        throw std::runtime_error(
            std::string("Could not extract chain from '") +
            BASE_LINK +
            "' to '" +
            END_EFFECTOR_LINK +
            "'. Check the URDF link names."
        );
    }

    std::cout
        << "\nLoaded robot model from URDF:\n"
        << "  File: " << URDF_PATH << '\n'
        << "  Base link: " << BASE_LINK << '\n'
        << "  End-effector link: " << END_EFFECTOR_LINK << '\n'
        << "  Segments: " << chain.getNrOfSegments() << '\n'
        << "  Moving joints: " << chain.getNrOfJoints() << '\n';

    return chain;
}

void verifyChainJointOrder(const KDL::Chain& chain)
{
    if (jointCalibrations.size() < JOINT_COUNT)
    {
        throw std::runtime_error(
            "Fewer than seven arm calibrations were found."
        );
    }

    std::size_t jointIndex = 0;

    std::cout << "\nURDF chain joint order:\n";

    for (unsigned int segmentIndex = 0;
         segmentIndex < chain.getNrOfSegments();
         ++segmentIndex)
    {
        const KDL::Joint& joint =
            chain.getSegment(segmentIndex).getJoint();

        if (joint.getType() == KDL::Joint::None)
        {
            continue;
        }

        if (jointIndex >= JOINT_COUNT)
        {
            throw std::runtime_error(
                "URDF chain contains more than seven moving joints."
            );
        }

        const JointCalibration& calibration =
            jointCalibrations[jointIndex];

        if (calibration.id != static_cast<int>(jointIndex))
        {
            throw std::runtime_error(
                "jointCalibrations must be ordered by motor ID 0 through 6."
            );
        }

        std::cout
            << "  Joint " << jointIndex
            << ": " << joint.getName()
            << " -> motor " << calibration.id
            << '\n';

        ++jointIndex;
    }

    if (jointIndex != JOINT_COUNT)
    {
        throw std::runtime_error(
            "URDF chain does not contain exactly seven moving joints."
        );
    }
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
    const PoseInput& poseA,
    const PoseInput& desiredPoseB,
    const PoseInput& commandedPoseB,
    const KDL::Frame& actualPoseA,
    const KDL::Frame& actualPoseB
)
{
    const double desiredDx = desiredPoseB.x - poseA.x;
    const double desiredDy = desiredPoseB.y - poseA.y;
    const double desiredDz = desiredPoseB.z - poseA.z;

    const double desiredDistance =
        std::sqrt(
            desiredDx * desiredDx +
            desiredDy * desiredDy +
            desiredDz * desiredDz
        );

    const double commandedDx = commandedPoseB.x - poseA.x;
    const double commandedDy = commandedPoseB.y - poseA.y;
    const double commandedDz = commandedPoseB.z - poseA.z;

    const double commandedDistance =
        std::sqrt(
            commandedDx * commandedDx +
            commandedDy * commandedDy +
            commandedDz * commandedDz
        );

    const double actualDx = actualPoseB.p.x() - actualPoseA.p.x();
    const double actualDy = actualPoseB.p.y() - actualPoseA.p.y();
    const double actualDz = actualPoseB.p.z() - actualPoseA.p.z();

    const double actualDistance =
        std::sqrt(
            actualDx * actualDx +
            actualDy * actualDy +
            actualDz * actualDz
        );

    const double desiredErrorDx = actualDx - desiredDx;
    const double desiredErrorDy = actualDy - desiredDy;
    const double desiredErrorDz = actualDz - desiredDz;
    const double desiredDistanceError = actualDistance - desiredDistance;

    const double modelTrackingErrorDx = actualDx - commandedDx;
    const double modelTrackingErrorDy = actualDy - commandedDy;
    const double modelTrackingErrorDz = actualDz - commandedDz;
    const double modelTrackingDistanceError =
        actualDistance - commandedDistance;

    std::cout
        << "\n========================================\n"
        << "DISPLACEMENT ANALYSIS\n"
        << "========================================\n"
        << std::fixed
        << std::setprecision(6)

        << "\nDesired physical displacement:\n"
        << "  dx = " << desiredDx
        << " m (" << desiredDx * 1000.0 << " mm)\n"
        << "  dy = " << desiredDy
        << " m (" << desiredDy * 1000.0 << " mm)\n"
        << "  dz = " << desiredDz
        << " m (" << desiredDz * 1000.0 << " mm)\n"
        << "  distance = " << desiredDistance
        << " m (" << desiredDistance * 1000.0 << " mm)\n"

        << "\nCorrected displacement sent to IK:\n"
        << "  dx = " << commandedDx
        << " m (" << commandedDx * 1000.0 << " mm)\n"
        << "  dy = " << commandedDy
        << " m (" << commandedDy * 1000.0 << " mm)\n"
        << "  dz = " << commandedDz
        << " m (" << commandedDz * 1000.0 << " mm)\n"
        << "  distance = " << commandedDistance
        << " m (" << commandedDistance * 1000.0 << " mm)\n"

        << "\nDisplacement calculated from reached motor ticks:\n"
        << "  dx = " << actualDx
        << " m (" << actualDx * 1000.0 << " mm)\n"
        << "  dy = " << actualDy
        << " m (" << actualDy * 1000.0 << " mm)\n"
        << "  dz = " << actualDz
        << " m (" << actualDz * 1000.0 << " mm)\n"
        << "  distance = " << actualDistance
        << " m (" << actualDistance * 1000.0 << " mm)\n"

        << "\nError relative to desired physical movement:\n"
        << "  x error = " << desiredErrorDx
        << " m (" << desiredErrorDx * 1000.0 << " mm)\n"
        << "  y error = " << desiredErrorDy
        << " m (" << desiredErrorDy * 1000.0 << " mm)\n"
        << "  z error = " << desiredErrorDz
        << " m (" << desiredErrorDz * 1000.0 << " mm)\n"
        << "  distance error = " << desiredDistanceError
        << " m (" << desiredDistanceError * 1000.0 << " mm)\n"

        << "\nMotor/model tracking error relative to corrected IK command:\n"
        << "  x error = " << modelTrackingErrorDx
        << " m (" << modelTrackingErrorDx * 1000.0 << " mm)\n"
        << "  y error = " << modelTrackingErrorDy
        << " m (" << modelTrackingErrorDy * 1000.0 << " mm)\n"
        << "  z error = " << modelTrackingErrorDz
        << " m (" << modelTrackingErrorDz * 1000.0 << " mm)\n"
        << "  distance error = " << modelTrackingDistanceError
        << " m (" << modelTrackingDistanceError * 1000.0 << " mm)\n";

    if (desiredDistance > 0.0)
    {
        std::cout
            << "  achieved desired distance = "
            << actualDistance / desiredDistance * 100.0
            << "%\n";
    }

    std::cout
        << "\nImportant: this value is model-based FK from motor ticks.\n"
        << "It does not independently verify the real physical distance.\n"
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

bool buildCartesianWaypointTargets(
    const PoseInput& startPose,
    const PoseInput& endPose,
    KDL::Chain& chain,
    const KDL::JntArray& lowerLimits,
    const KDL::JntArray& upperLimits,
    const KDL::JntArray& startingSeed,
    std::vector<std::vector<uint16_t>>& waypointTicks
)
{
    waypointTicks.clear();

    const double dx =
        endPose.x - startPose.x;

    const double dy =
        endPose.y - startPose.y;

    const double dz =
        endPose.z - startPose.z;

    const double distance =
        std::sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        );

    if (distance <= 0.0)
    {
        std::cerr
            << "Cartesian path distance must be greater than zero.\n";

        return false;
    }

    const std::size_t waypointCount =
        static_cast<std::size_t>(
            std::ceil(
                distance /
                CARTESIAN_WAYPOINT_SPACING
            )
        );

    std::cout
        << "\nBuilding Cartesian path:\n"
        << "  Distance: "
        << distance * 1000.0
        << " mm\n"
        << "  Waypoint spacing: "
        << CARTESIAN_WAYPOINT_SPACING * 1000.0
        << " mm\n"
        << "  Waypoints: "
        << waypointCount
        << '\n';

    TRAC_IK::TRAC_IK solver(
        chain,
        lowerLimits,
        upperLimits,
        0.50,
        1e-5,
        TRAC_IK::Distance
    );

    KDL::ChainFkSolverPos_recursive fkSolver(
        chain
    );

    KDL::JntArray seed =
        startingSeed;

    waypointTicks.reserve(
        waypointCount
    );

    for (std::size_t waypointIndex = 1;
         waypointIndex <= waypointCount;
         ++waypointIndex)
    {
        const double fraction =
            static_cast<double>(waypointIndex) /
            static_cast<double>(waypointCount);

        const PoseInput waypointPose = {
            startPose.x + fraction * dx,
            startPose.y + fraction * dy,
            startPose.z + fraction * dz
        };

        const KDL::Frame targetFrame =
            makeTargetFrame(
                waypointPose
            );

        KDL::JntArray solution(
            JOINT_COUNT
        );

        const int result =
            solver.CartToJnt(
                seed,
                targetFrame,
                solution
            );

        if (result < 0)
        {
            std::cerr
                << "TRAC-IK failed at waypoint "
                << waypointIndex
                << " of "
                << waypointCount
                << ".\n";

            waypointTicks.clear();
            return false;
        }

        KDL::Frame verifiedFrame;

        if (fkSolver.JntToCart(
                solution,
                verifiedFrame
            ) < 0)
        {
            std::cerr
                << "FK verification failed at waypoint "
                << waypointIndex
                << ".\n";

            waypointTicks.clear();
            return false;
        }

        const KDL::Twist error =
            KDL::diff(
                verifiedFrame,
                targetFrame
            );

        if (error.vel.Norm() > 1e-4 ||
            error.rot.Norm() > 1e-4)
        {
            std::cerr
                << "Waypoint "
                << waypointIndex
                << " exceeded IK verification tolerance.\n"
                << "Position error: "
                << error.vel.Norm()
                << " m\n"
                << "Orientation error: "
                << error.rot.Norm()
                << " rad\n";

            waypointTicks.clear();
            return false;
        }

        std::vector<uint16_t> ticks;
        ticks.reserve(JOINT_COUNT);

        for (std::size_t jointIndex = 0;
             jointIndex < JOINT_COUNT;
             ++jointIndex)
        {
            const JointCalibration& calibration =
                jointCalibrations[jointIndex];

            const int tick =
                radiansToTicks(
                    calibration,
                    solution(jointIndex)
                );

            const bool safe =
                tick >= calibration.minTick &&
                tick <= calibration.maxTick &&
                tick >= 0 &&
                tick <= 4095;

            if (!safe)
            {
                std::cerr
                    << "Unsafe motor target at waypoint "
                    << waypointIndex
                    << ", motor "
                    << calibration.id
                    << ": "
                    << tick
                    << " ticks.\n";

                waypointTicks.clear();
                return false;
            }

            ticks.push_back(
                static_cast<uint16_t>(tick)
            );
        }

        waypointTicks.push_back(
            ticks
        );

        /*
         * The next waypoint uses the current solution as its seed.
         * This encourages a continuous nearby joint configuration.
         */
        seed =
            solution;
    }

    return waypointTicks.size() ==
           waypointCount;
}

bool executeCartesianWaypointPath(
    DynamixelMotor& motor,
    const std::vector<int>& motorIds,
    const std::vector<std::vector<uint16_t>>& waypointTicks
)
{
    if (waypointTicks.empty())
    {
        std::cerr
            << "Cartesian waypoint list is empty.\n";

        return false;
    }

    std::cout
        << "\nExecuting Cartesian path with "
        << waypointTicks.size()
        << " waypoints.\n";

    for (std::size_t waypointIndex = 0;
         waypointIndex < waypointTicks.size();
         ++waypointIndex)
    {
        std::cout
            << "\rMoving through waypoint "
            << waypointIndex + 1
            << " of "
            << waypointTicks.size()
            << std::flush;

        if (!motor.moveJointsSafely(
                motorIds,
                waypointTicks[waypointIndex],
                MOVING_SPEED,
                TOLERANCE,
                TIMEOUT_SECONDS,
                true
            ))
        {
            std::cerr
                << "\nMovement failed at Cartesian waypoint "
                << waypointIndex + 1
                << ".\n";

            return false;
        }
    }

    std::cout
        << "\nCartesian path completed.\n";

    return true;
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
         * Desired physical movement is 150 mm in +X.
         * A separate corrected target is sent to IK.
         */
        const PoseInput poseA = {
            -0.036864, // x
            0.019448, // y
            0.646952  // z
        };

        constexpr double DESIRED_X_MOVEMENT = 0.10;

        const double correctedXMovement =
            applyHorizontalDistanceCorrection(
                DESIRED_X_MOVEMENT
            );

        const PoseInput desiredPoseB = {
            poseA.x + DESIRED_X_MOVEMENT,
            poseA.y,
            poseA.z
        };

        const PoseInput commandedPoseB = {
            poseA.x + correctedXMovement,
            poseA.y,
            poseA.z
        };

        std::cout
            << "\nDesired X movement: "
            << DESIRED_X_MOVEMENT * 1000.0
            << " mm\n"
            << "Corrected X movement sent to IK: "
            << correctedXMovement * 1000.0
            << " mm\n";

        KDL::Chain chain =
            createCytonChainFromUrdf();

        verifyChainJointOrder(chain);

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
        std::vector<std::vector<uint16_t>> cartesianWaypointTicks;

        if (!buildCartesianWaypointTargets(
                poseA,
                commandedPoseB,
                chain,
                lowerLimits,
                upperLimits,
                solutionA,
                cartesianWaypointTicks
            ))
        {
            std::cerr
                << "Could not generate the Cartesian path.\n";

            return 1;
        }

        /*
        * The final waypoint is the corrected Pose B motor target.
        */
        const std::vector<uint16_t>& poseBTicks =
            cartesianWaypointTicks.back();

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

        if (!executeCartesianWaypointPath(
                motor,
                motorIds,
                cartesianWaypointTicks
            ))
        {
            std::cerr
                << "Cartesian movement to Pose B failed.\n";

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
            desiredPoseB,
            commandedPoseB,
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