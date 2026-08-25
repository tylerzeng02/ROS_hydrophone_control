/**
 * @file test_radians_pose.cpp
 * @brief Converts a hardcoded raw-tick pose to radians and back, printing
 * both, to sanity-check a full 7-joint radiansToTicks()/ticksToRadians()
 * round trip before moving anything on real hardware.
 */

#include <iostream>
#include <vector>
#include <cstdint>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

/**
 * @brief Converts a full 7-joint raw-tick pose to radians via
 * jointCalibrations.
 * @param rawPose Raw tick values, one per joint, in jointCalibrations
 *        order.
 * @return The corresponding radians, same order. Empty if `rawPose`'s
 *         size does not match jointCalibrations.
 */
std::vector<double> rawPoseToRadians(const std::vector<uint16_t>& rawPose)
{
    std::vector<double> radiansPose;

    if (rawPose.size() != jointCalibrations.size())
    {
        std::cerr << "Raw pose size does not match joint calibration size." << std::endl;
        return radiansPose;
    }

    for (size_t i = 0; i < rawPose.size(); ++i)
    {
        const JointCalibration& joint = jointCalibrations[i];
        double radians = ticksToRadians(joint, rawPose[i]);
        radiansPose.push_back(radians);
    }

    return radiansPose;
}

/**
 * @brief Converts each joint's radians back to a target tick via
 * radiansToTicks() and prints motor ID, angle, and target tick.
 * @param radiansPose Joint angles in radians, in jointCalibrations order.
 */
void printPoseTargets(const std::vector<double>& radiansPose)
{
    std::cout << "\nRadians pose target conversion:" << std::endl;

    for (size_t i = 0; i < radiansPose.size(); ++i)
    {
        const JointCalibration& joint = jointCalibrations[i];

        int targetTick = radiansToTicks(joint, radiansPose[i]);

        std::cout << "Motor " << joint.id
                  << " | radians: " << radiansPose[i]
                  << " | target tick: " << targetTick
                  << " | safe range: [" << joint.minTick
                  << ", " << joint.maxTick << "]"
                  << std::endl;
    }
}

bool poseTargetsAreSafe(
    DynamixelMotor& motor,
    const std::vector<double>& radiansPose
)
{
    if (radiansPose.size() != jointCalibrations.size())
    {
        std::cerr << "Pose size does not match calibration size." << std::endl;
        return false;
    }

    for (size_t i = 0; i < radiansPose.size(); ++i)
    {
        const JointCalibration& joint = jointCalibrations[i];

        uint16_t targetTick = static_cast<uint16_t>(
            radiansToTicks(joint, radiansPose[i])
        );

        if (!motor.isPositionSafe(joint.id, targetTick))
        {
            std::cerr << "Unsafe target detected." << std::endl;
            std::cerr << "Motor ID: " << joint.id << std::endl;
            std::cerr << "Target tick: " << targetTick << std::endl;
            return false;
        }
    }

    return true;
}

bool trajectoryTargetsAreSafe(
    DynamixelMotor& motor,
    const std::vector<std::vector<double>>& trajectory
)
{
    for (size_t i = 0; i < trajectory.size(); ++i)
    {
        if (!poseTargetsAreSafe(motor, trajectory[i]))
        {
            std::cerr << "Trajectory point " << i
                      << " contains unsafe target." << std::endl;
            return false;
        }
    }

    return true;
}

bool moveRadiansTrajectory(
    DynamixelMotor& motor,
    const std::vector<std::vector<double>>& trajectory,
    uint16_t speed,
    int tolerance,
    int timeoutSeconds,
    bool holdTorque
)
{
    if (trajectory.empty())
    {
        std::cerr << "Radians trajectory is empty." << std::endl;
        return false;
    }

    for (size_t i = 0; i < trajectory.size(); ++i)
    {
        std::cout << "\nMoving to radians trajectory point "
                  << i << "..." << std::endl;

        printPoseTargets(trajectory[i]);

        if (!motor.moveJointRadiansPose(
                trajectory[i],
                speed,
                tolerance,
                timeoutSeconds,
                holdTorque
            ))
        {
            std::cerr << "Failed at radians trajectory point "
                      << i << std::endl;
            return false;
        }
    }

    std::cout << "\nRadians trajectory complete." << std::endl;
    return true;
}

int main()
{
    const char* DEVICENAME = "/dev/ttyUSB0";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0;

    DynamixelMotor motor(DEVICENAME, BAUDRATE, PROTOCOL_VERSION);

    if (!motor.connect())
    {
        std::cerr << "Failed to connect to Dynamixel motors." << std::endl;
        return 1;
    }

    const uint16_t speed = 25;
    const int homeTolerance = 15;
    const int trajectoryTolerance = 20;
    const int timeoutSeconds = 25;
    const bool holdTorque = true;

    std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<uint16_t> homePoseRaw = {
        2015, 857, 935, 3239, 1200, 3087, 1967, 2350
    };

    std::vector<uint16_t> testPose1Raw = {
        1776, 2408, 1144, 2991, 1495, 2312, 1759, 2350
    };

    std::vector<std::vector<uint16_t>> rawTrajectoryToTestPose1 = {
        homePoseRaw,
        {1955, 1245, 987, 3177, 1274, 2893, 1915, 2350},
        {1896, 1633, 1040, 3115, 1348, 2700, 1863, 2350},
        {1836, 2020, 1092, 3053, 1421, 2506, 1811, 2350},
        testPose1Raw
    };

    std::vector<std::vector<double>> radiansTrajectoryToTestPose1;

    for (const auto& rawPose : rawTrajectoryToTestPose1)
    {
        radiansTrajectoryToTestPose1.push_back(
            rawPoseToRadians(rawPose)
        );
    }

    std::vector<std::vector<double>> radiansTrajectoryBackHome;

    for (auto it = radiansTrajectoryToTestPose1.rbegin();
         it != radiansTrajectoryToTestPose1.rend();
         ++it)
    {
        radiansTrajectoryBackHome.push_back(*it);
    }

    std::vector<double> homeRadians = rawPoseToRadians(homePoseRaw);
    std::vector<double> testPose1Radians = rawPoseToRadians(testPose1Raw);

    std::cout << "\n===== FULL RADIANS TRAJECTORY TEST =====" << std::endl;
    std::cout << "Speed: " << speed << std::endl;
    std::cout << "Home tolerance: " << homeTolerance << std::endl;
    std::cout << "Trajectory tolerance: " << trajectoryTolerance << std::endl;

    std::cout << "\nHome pose:" << std::endl;
    printPoseTargets(homeRadians);

    std::cout << "\nTest pose 1:" << std::endl;
    printPoseTargets(testPose1Radians);

    if (!trajectoryTargetsAreSafe(motor, radiansTrajectoryToTestPose1))
    {
        std::cerr << "Trajectory to test pose 1 contains unsafe targets. Cancelling."
                  << std::endl;
        motor.disconnect();
        return 1;
    }

    if (!trajectoryTargetsAreSafe(motor, radiansTrajectoryBackHome))
    {
        std::cerr << "Trajectory back home contains unsafe targets. Cancelling."
                  << std::endl;
        motor.disconnect();
        return 1;
    }

    std::cout << "\nMake sure the arm is clear." << std::endl;
    std::cout << "This test moves:" << std::endl;
    std::cout << "home pose -> trajectory -> test pose 1 -> trajectory back home"
              << std::endl;

    std::cout << "\nPress ENTER to move all motors to home / 0 rad...";
    std::cin.get();

    if (!motor.moveJointRadiansPose(
            homeRadians,
            speed,
            homeTolerance,
            timeoutSeconds,
            holdTorque
        ))
    {
        std::cerr << "Failed to move to home radians pose." << std::endl;
        motor.emergencyShutdown(motorIds);
        return 1;
    }

    std::cout << "\nHome pose reached. Torque is holding." << std::endl;

    std::cout << "\nPress ENTER to run trajectory to test pose 1...";
    std::cin.get();

    if (!moveRadiansTrajectory(
            motor,
            radiansTrajectoryToTestPose1,
            speed,
            trajectoryTolerance,
            timeoutSeconds,
            holdTorque
        ))
    {
        std::cerr << "Failed to complete trajectory to test pose 1." << std::endl;
        motor.emergencyShutdown(motorIds);
        return 1;
    }

    std::cout << "\nTest pose 1 reached. Torque is holding." << std::endl;

    std::cout << "\nPress ENTER to return to home through reverse trajectory...";
    std::cin.get();

    if (!moveRadiansTrajectory(
            motor,
            radiansTrajectoryBackHome,
            speed,
            trajectoryTolerance,
            timeoutSeconds,
            holdTorque
        ))
    {
        std::cerr << "Failed to complete trajectory back home." << std::endl;
        motor.emergencyShutdown(motorIds);
        return 1;
    }

    std::cout << "\nReturned to home pose. Torque is holding." << std::endl;

    std::cout << "\nPress ENTER to disable torque and disconnect...";
    std::cin.get();

    for (int id : motorIds)
    {
        motor.disableTorque(id);
    }

    motor.disconnect();

    std::cout << "Full radians trajectory test complete." << std::endl;

    return 0;
}