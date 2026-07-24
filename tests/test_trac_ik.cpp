#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>

#include <trac_ik/trac_ik.hpp>

namespace
{
constexpr double PI = 3.14159265358979323846;
constexpr double TICKS_PER_REVOLUTION = 4096.0;

double ticksToRadians(
    int ticks,
    int zeroTick,
    int direction
)
{
    return direction *
           static_cast<double>(ticks - zeroTick) *
           (2.0 * PI / TICKS_PER_REVOLUTION);
}

KDL::Chain createTestChain()
{
    KDL::Chain chain;

    /*
     * This is a software-only seven-joint test chain.
     *
     * The dimensions are temporary test dimensions, not yet the final
     * measured Cyton Gamma geometry. This test verifies that TRAC-IK
     * compiles and solves a reachable seven-joint problem.
     */

    chain.addSegment(
        KDL::Segment(
            "link_1",
            KDL::Joint("joint_1", KDL::Joint::RotZ),
            KDL::Frame(KDL::Vector(0.0, 0.0, 0.080))
        )
    );

    chain.addSegment(
        KDL::Segment(
            "link_2",
            KDL::Joint("joint_2", KDL::Joint::RotY),
            KDL::Frame(KDL::Vector(0.0, 0.0, 0.075))
        )
    );

    chain.addSegment(
        KDL::Segment(
            "link_3",
            KDL::Joint("joint_3", KDL::Joint::RotX),
            KDL::Frame(KDL::Vector(0.0, 0.0, 0.090))
        )
    );

    chain.addSegment(
        KDL::Segment(
            "link_4",
            KDL::Joint("joint_4", KDL::Joint::RotY),
            KDL::Frame(KDL::Vector(0.0, 0.0, 0.080))
        )
    );

    chain.addSegment(
        KDL::Segment(
            "link_5",
            KDL::Joint("joint_5", KDL::Joint::RotX),
            KDL::Frame(KDL::Vector(0.0, 0.0, 0.065))
        )
    );

    chain.addSegment(
        KDL::Segment(
            "link_6",
            KDL::Joint("joint_6", KDL::Joint::RotY),
            KDL::Frame(KDL::Vector(0.0, 0.0, 0.055))
        )
    );

    chain.addSegment(
        KDL::Segment(
            "link_7",
            KDL::Joint("joint_7", KDL::Joint::RotZ),
            KDL::Frame(KDL::Vector(0.0, 0.0, 0.050))
        )
    );

    return chain;
}

void printJointArray(
    const std::string& label,
    const KDL::JntArray& joints
)
{
    std::cout << label << '\n';

    for (unsigned int i = 0; i < joints.rows(); ++i)
    {
        std::cout
            << "  Joint " << i
            << ": " << std::fixed << std::setprecision(6)
            << joints(i) << " rad\n";
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

    frame.M.GetRPY(roll, pitch, yaw);

    std::cout << label << '\n';

    std::cout
        << "  Position: x=" << frame.p.x()
        << ", y=" << frame.p.y()
        << ", z=" << frame.p.z()
        << " m\n";

    std::cout
        << "  Orientation: roll=" << roll
        << ", pitch=" << pitch
        << ", yaw=" << yaw
        << " rad\n";
}
}

int main()
{
    KDL::Chain chain = createTestChain();

    constexpr std::size_t JOINT_COUNT = 7;

    if (chain.getNrOfJoints() != JOINT_COUNT)
    {
        std::cerr
            << "Expected 7 joints, but the chain contains "
            << chain.getNrOfJoints() << ".\n";

        return 1;
    }

    /*
     * Calibrated raw tick ranges for motor IDs 0 through 6.
     * Motor ID 7 is the gripper and is not part of the IK chain.
     */
    const std::array<int, JOINT_COUNT> minimumTicks = {
        341,
        853,
        912,
        853,
        853,
        853,
        341
    };

    const std::array<int, JOINT_COUNT> maximumTicks = {
        3755,
        3243,
        3320,
        3243,
        3243,
        3243,
        3755
    };

    const std::array<int, JOINT_COUNT> zeroTicks = {
        2048,
        2048,
        2116,
        2048,
        2048,
        2048,
        2048
    };

    const std::array<int, JOINT_COUNT> directions = {
        1,
        1,
        1,
        1,
        1,
        -1,
        1
    };

    KDL::JntArray lowerLimits(JOINT_COUNT);
    KDL::JntArray upperLimits(JOINT_COUNT);

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const double angleAtMinimumTick = ticksToRadians(
            minimumTicks[i],
            zeroTicks[i],
            directions[i]
        );

        const double angleAtMaximumTick = ticksToRadians(
            maximumTicks[i],
            zeroTicks[i],
            directions[i]
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

    std::cout
        << "Chain joints: " << chain.getNrOfJoints()
        << "\nChain segments: " << chain.getNrOfSegments()
        << "\n\n";

    std::cout << "Joint limits:\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        std::cout
            << "  Joint " << i
            << ": [" << lowerLimits(i)
            << ", " << upperLimits(i)
            << "] rad\n";
    }

    /*
     * Select a known valid joint configuration.
     * Forward kinematics creates a guaranteed-reachable target pose.
     */
    KDL::JntArray expectedJoints(JOINT_COUNT);

    expectedJoints(0) = 0.25;
    expectedJoints(1) = -0.30;
    expectedJoints(2) = 0.20;
    expectedJoints(3) = 0.40;
    expectedJoints(4) = -0.20;
    expectedJoints(5) = 0.30;
    expectedJoints(6) = 0.10;

    KDL::ChainFkSolverPos_recursive forwardKinematics(chain);

    KDL::Frame targetPose;

    const int fkResult = forwardKinematics.JntToCart(
        expectedJoints,
        targetPose
    );

    if (fkResult < 0)
    {
        std::cerr
            << "Failed to generate the target pose using "
            << "forward kinematics.\n";

        return 1;
    }

    /*
     * Start IK from the zero-angle configuration.
     */
    KDL::JntArray seedJoints(JOINT_COUNT);
    KDL::JntArray solvedJoints(JOINT_COUNT);

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        seedJoints(i) = 0.0;
    }

    const double maximumSolveTimeSeconds = 0.25;
    const double positionTolerance = 1e-5;

    TRAC_IK::TRAC_IK solver(
        chain,
        lowerLimits,
        upperLimits,
        maximumSolveTimeSeconds,
        positionTolerance,
        TRAC_IK::Distance
    );

    const int ikResult = solver.CartToJnt(
        seedJoints,
        targetPose,
        solvedJoints
    );

    std::cout << '\n';

    printJointArray(
        "Joint configuration used to generate target:",
        expectedJoints
    );

    std::cout << '\n';
    printFrame("Requested target pose:", targetPose);

    if (ikResult < 0)
    {
        std::cerr
            << "\nTRAC-IK failed to find a solution. "
            << "Return code: " << ikResult << '\n';

        return 1;
    }

    std::cout
        << "\nTRAC-IK succeeded and found "
        << ikResult << " solution(s).\n\n";

    printJointArray(
        "Solved joint configuration:",
        solvedJoints
    );

    KDL::Frame solvedPose;

    const int solvedFkResult = forwardKinematics.JntToCart(
        solvedJoints,
        solvedPose
    );

    if (solvedFkResult < 0)
    {
        std::cerr
            << "Failed to verify the solved pose using "
            << "forward kinematics.\n";

        return 1;
    }

    std::cout << '\n';
    printFrame("Pose produced by IK solution:", solvedPose);

    const KDL::Twist poseError = KDL::diff(
        solvedPose,
        targetPose
    );

    std::cout
        << "\nPosition error magnitude: "
        << poseError.vel.Norm() << " m\n";

    std::cout
        << "Orientation error magnitude: "
        << poseError.rot.Norm() << " rad\n";

    std::cout
        << "\nSoftware-only TRAC-IK test completed. "
        << "No motors were accessed.\n";

    return 0;
}