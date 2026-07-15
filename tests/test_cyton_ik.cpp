#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstddef>
#include <stdexcept>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>

#include <trac_ik/trac_ik.hpp>

#include "robot_calibration.h"

namespace
{
constexpr std::size_t JOINT_COUNT = 7;

struct JointDescription
{
    const char* jointName;
    const char* childLinkName;

    double originX;
    double originY;
    double originZ;

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

KDL::Chain createCytonChain()
{
    KDL::Chain chain;

    for (const JointDescription& description : CYTON_JOINTS)
    {
        const KDL::Vector jointOrigin(
            description.originX,
            description.originY,
            description.originZ
        );

        const KDL::Vector jointAxis(
            description.axisX,
            description.axisY,
            description.axisZ
        );

        const KDL::Joint joint(
            description.jointName,
            jointOrigin,
            jointAxis,
            KDL::Joint::RotAxis
        );

        const KDL::Frame frameToChild(jointOrigin);

        chain.addSegment(
            KDL::Segment(
                description.childLinkName,
                joint,
                frameToChild
            )
        );
    }

    /*
     * Fixed transform from wrist_roll to virtual_endeffector.
     *
     * From the URDF:
     * xyz="-0.002316 0.0079 0.079425"
     */
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
            "robot_calibration.cpp contains fewer than seven arm joints."
        );
    }

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& calibration =
            jointCalibrations[i];

        const double angleAtMinimumTick =
            ticksToRadians(
                calibration,
                calibration.minTick
            );

        const double angleAtMaximumTick =
            ticksToRadians(
                calibration,
                calibration.maxTick
            );

        lowerLimits(i) = std::min(
            angleAtMinimumTick,
            angleAtMaximumTick
        );

        upperLimits(i) = std::max(
            angleAtMinimumTick,
            angleAtMaximumTick
        );
    }
}

void printJointLimits(
    const KDL::JntArray& lowerLimits,
    const KDL::JntArray& upperLimits
)
{
    std::cout << "Joint limits from physical calibration:\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& calibration =
            jointCalibrations[i];

        std::cout
            << "  Joint " << i
            << " / Motor " << calibration.id
            << " [" << lowerLimits(i)
            << ", " << upperLimits(i)
            << "] rad"
            << " | ticks ["
            << calibration.minTick
            << ", "
            << calibration.maxTick
            << "]"
            << " | zero tick "
            << calibration.zeroTick
            << " | direction "
            << calibration.direction
            << '\n';
    }
}

void printJointSolution(
    const std::string& label,
    const KDL::JntArray& joints
)
{
    std::cout << label << '\n';

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& calibration =
            jointCalibrations[i];

        const double radians = joints(i);

        const int ticks =
            radiansToTicks(
                calibration,
                radians
            );

        const bool tickIsSafe =
            ticks >= calibration.minTick &&
            ticks <= calibration.maxTick;

        std::cout
            << "  Joint " << i
            << " / Motor " << calibration.id
            << " | "
            << std::fixed
            << std::setprecision(6)
            << radians
            << " rad"
            << " | "
            << ticks
            << " ticks"
            << " | "
            << (tickIsSafe ? "SAFE" : "OUTSIDE SAFE RANGE")
            << '\n';
    }
}

void printFrame(
    const std::string& label,
    const KDL::Frame& frame
)
{
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    frame.M.GetRPY(
        roll,
        pitch,
        yaw
    );

    std::cout << label << '\n';

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "  Position:\n"
        << "    x = " << frame.p.x() << " m\n"
        << "    y = " << frame.p.y() << " m\n"
        << "    z = " << frame.p.z() << " m\n";

    std::cout
        << "  Orientation:\n"
        << "    roll  = " << roll << " rad\n"
        << "    pitch = " << pitch << " rad\n"
        << "    yaw   = " << yaw << " rad\n";
}

bool solutionIsSafe(
    const KDL::JntArray& joints,
    const KDL::JntArray& lowerLimits,
    const KDL::JntArray& upperLimits
)
{
    bool allSafe = true;

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& calibration =
            jointCalibrations[i];

        const double radians = joints(i);

        const int ticks =
            radiansToTicks(
                calibration,
                radians
            );

        const bool radiansSafe =
            radians >= lowerLimits(i) &&
            radians <= upperLimits(i);

        const bool ticksSafe =
            ticks >= calibration.minTick &&
            ticks <= calibration.maxTick;

        if (!radiansSafe || !ticksSafe)
        {
            std::cerr
                << "Unsafe IK result for joint "
                << i
                << " / motor "
                << calibration.id
                << ".\n";

            allSafe = false;
        }
    }

    return allSafe;
}
}

int main()
{
    try
    {
        KDL::Chain chain =
            createCytonChain();

        std::cout
            << "Cyton Gamma 1500 IK test\n"
            << "Base link: base_link\n"
            << "Tip link: virtual_endeffector\n\n";

        std::cout
            << "Chain joints: "
            << chain.getNrOfJoints()
            << '\n';

        std::cout
            << "Chain segments: "
            << chain.getNrOfSegments()
            << "\n\n";

        if (chain.getNrOfJoints() != JOINT_COUNT)
        {
            std::cerr
                << "Expected seven movable joints, but the chain has "
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

        printJointLimits(
            lowerLimits,
            upperLimits
        );

        /*
         * Use a known safe joint configuration to generate a reachable
         * target pose through forward kinematics.
         *
         * These are arm joint radians for motors 0 through 6.
         * Motor 7 is the gripper and is not part of IK.
         */
        KDL::JntArray expectedJoints(JOINT_COUNT);

        expectedJoints(0) = 0.20;
        expectedJoints(1) = -0.25;
        expectedJoints(2) = 0.15;
        expectedJoints(3) = 0.30;
        expectedJoints(4) = -0.15;
        expectedJoints(5) = 0.20;
        expectedJoints(6) = 0.10;

        if (!solutionIsSafe(
                expectedJoints,
                lowerLimits,
                upperLimits
            ))
        {
            std::cerr
                << "The test configuration is outside the calibrated "
                << "joint limits.\n";

            return 1;
        }

        KDL::ChainFkSolverPos_recursive forwardKinematics(
            chain
        );

        KDL::Frame targetPose;

        const int targetFkResult =
            forwardKinematics.JntToCart(
                expectedJoints,
                targetPose
            );

        if (targetFkResult < 0)
        {
            std::cerr
                << "Failed to generate the target pose using "
                << "forward kinematics.\n";

            return 1;
        }

        std::cout << '\n';

        printJointSolution(
            "Known joint configuration used to generate target:",
            expectedJoints
        );

        std::cout << '\n';

        printFrame(
            "Requested target pose:",
            targetPose
        );

        /*
         * Start TRAC-IK from the zero-radian pose.
         */
        KDL::JntArray seedJoints(JOINT_COUNT);
        KDL::JntArray solvedJoints(JOINT_COUNT);

        for (std::size_t i = 0; i < JOINT_COUNT; ++i)
        {
            seedJoints(i) = 0.0;
            solvedJoints(i) = 0.0;
        }

        constexpr double MAXIMUM_SOLVE_TIME_SECONDS = 0.25;
        constexpr double SOLVER_TOLERANCE = 1e-5;

        TRAC_IK::TRAC_IK solver(
            chain,
            lowerLimits,
            upperLimits,
            MAXIMUM_SOLVE_TIME_SECONDS,
            SOLVER_TOLERANCE,
            TRAC_IK::Distance
        );

        const int ikResult =
            solver.CartToJnt(
                seedJoints,
                targetPose,
                solvedJoints
            );

        if (ikResult < 0)
        {
            std::cerr
                << "\nTRAC-IK failed to find a solution."
                << " Return code: "
                << ikResult
                << '\n';

            return 1;
        }

        std::cout
            << "\nTRAC-IK succeeded."
            << " Return value: "
            << ikResult
            << "\n\n";

        printJointSolution(
            "Solved joint configuration:",
            solvedJoints
        );

        if (!solutionIsSafe(
                solvedJoints,
                lowerLimits,
                upperLimits
            ))
        {
            std::cerr
                << "\nIK found a solution, but the result failed "
                << "the calibrated safety check.\n";

            return 1;
        }

        KDL::Frame solvedPose;

        const int solvedFkResult =
            forwardKinematics.JntToCart(
                solvedJoints,
                solvedPose
            );

        if (solvedFkResult < 0)
        {
            std::cerr
                << "Failed to run forward kinematics on the "
                << "IK solution.\n";

            return 1;
        }

        std::cout << '\n';

        printFrame(
            "Pose produced by IK solution:",
            solvedPose
        );

        const KDL::Twist poseError =
            KDL::diff(
                solvedPose,
                targetPose
            );

        const double positionError =
            poseError.vel.Norm();

        const double orientationError =
            poseError.rot.Norm();

        std::cout
            << "\nVerification results:\n"
            << "  Position error: "
            << std::scientific
            << positionError
            << " m\n"
            << "  Orientation error: "
            << orientationError
            << " rad\n";

        constexpr double MAX_POSITION_ERROR = 1e-4;
        constexpr double MAX_ORIENTATION_ERROR = 1e-4;

        if (positionError > MAX_POSITION_ERROR ||
            orientationError > MAX_ORIENTATION_ERROR)
        {
            std::cerr
                << "\nIK verification failed because the pose error "
                << "is too large.\n";

            return 1;
        }

        std::cout
            << "\nPASS: Cyton Gamma 1500 IK solution is valid, "
            << "forward-kinematics verified, and within the "
            << "calibrated motor limits.\n";

        std::cout
            << "No serial port was opened and no motors were accessed.\n";

        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "Test failed with exception: "
            << exception.what()
            << '\n';

        return 1;
    }
}
