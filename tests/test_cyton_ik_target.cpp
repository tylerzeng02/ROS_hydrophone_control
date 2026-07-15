#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
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

void printPose(
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

bool printAndCheckSolution(
    const KDL::JntArray& solution
)
{
    bool allSafe = true;

    std::cout << "\nSolved joint values:\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i)
    {
        const JointCalibration& joint =
            jointCalibrations[i];

        const double radians =
            solution(i);

        const int ticks =
            radiansToTicks(
                joint,
                radians
            );

        const bool safe =
            ticks >= joint.minTick &&
            ticks <= joint.maxTick;

        std::cout
            << "Motor " << joint.id
            << " | "
            << std::fixed
            << std::setprecision(6)
            << radians
            << " rad"
            << " | "
            << ticks
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

        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;

        std::cout
            << "Enter the desired end-effector pose.\n"
            << "Position values are in metres.\n"
            << "Orientation values are in radians.\n\n";

        std::cout << "x: ";
        std::cin >> x;

        std::cout << "y: ";
        std::cin >> y;

        std::cout << "z: ";
        std::cin >> z;

        std::cout << "roll: ";
        std::cin >> roll;

        std::cout << "pitch: ";
        std::cin >> pitch;

        std::cout << "yaw: ";
        std::cin >> yaw;

        if (!std::cin)
        {
            std::cerr
                << "Invalid numerical input.\n";

            return 1;
        }

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

        KDL::JntArray lowerLimits(JOINT_COUNT);
        KDL::JntArray upperLimits(JOINT_COUNT);

        buildJointLimits(
            lowerLimits,
            upperLimits
        );

        KDL::JntArray seed(JOINT_COUNT);
        KDL::JntArray solution(JOINT_COUNT);

        /*
         * Zero radians corresponds to the calibrated midpoint/home ticks.
         */
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

        std::cout << "\nRequested pose:\n";
        printPose(targetPose);

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

        std::cout
            << "\nTRAC-IK found a solution.\n";

        if (!printAndCheckSolution(solution))
        {
            std::cerr
                << "\nThe result failed the calibrated "
                << "motor-limit check.\n";

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

        const KDL::Twist error =
            KDL::diff(
                verifiedPose,
                targetPose
            );

        std::cout
            << "\nPose produced by the solution:\n";

        printPose(verifiedPose);

        std::cout
            << "\nVerification:\n"
            << "  Position error: "
            << std::scientific
            << error.vel.Norm()
            << " m\n"
            << "  Orientation error: "
            << error.rot.Norm()
            << " rad\n";

        if (error.vel.Norm() > 1e-4 ||
            error.rot.Norm() > 1e-4)
        {
            std::cerr
                << "\nThe solution error is too large.\n";

            return 1;
        }

        std::cout
            << "\nPASS: The entered end-effector pose has a "
            << "valid and safe IK solution.\n"
            << "No motors were accessed.\n";

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