#include <iostream>
#include <vector>
#include <cstdint>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

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

    const uint16_t speed = 30;
    const int homeTolerance = 15;
    const int poseTolerance = 15;
    const int timeoutSeconds = 20;
    const bool holdTorque = true;

    std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<double> homeRadians = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

    // Small safe pose based on the directions you successfully tested.
    // Motor 7/gripper stays at 0.0 for now.
    std::vector<double> smallPose = {
        -0.05,  // motor 0
         0.05,  // motor 1
         0.05,  // motor 2
        -0.05,  // motor 3
        -0.05,  // motor 4
         0.05,  // motor 5
         0.05,  // motor 6
         0.0    // motor 7
    };

    std::cout << "\n===== FULL RADIANS POSE TEST =====" << std::endl;
    std::cout << "Speed: " << speed << std::endl;
    std::cout << "Home tolerance: " << homeTolerance << std::endl;
    std::cout << "Pose tolerance: " << poseTolerance << std::endl;

    std::cout << "\nHome pose targets:" << std::endl;
    printPoseTargets(homeRadians);

    std::cout << "\nSmall pose targets:" << std::endl;
    printPoseTargets(smallPose);

    if (!poseTargetsAreSafe(motor, homeRadians))
    {
        std::cerr << "Home radians pose contains unsafe target. Cancelling." << std::endl;
        motor.disconnect();
        return 1;
    }

    if (!poseTargetsAreSafe(motor, smallPose))
    {
        std::cerr << "Small radians pose contains unsafe target. Cancelling." << std::endl;
        motor.disconnect();
        return 1;
    }

    std::cout << "\nMake sure the arm is clear." << std::endl;
    std::cout << "Press ENTER to move all motors to home / 0 rad...";
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

    std::cout << "\nPress ENTER to move to small radians pose...";
    std::cin.get();

    if (!motor.moveJointRadiansPose(
            smallPose,
            speed,
            poseTolerance,
            timeoutSeconds,
            holdTorque
        ))
    {
        std::cerr << "Failed to move to small radians pose." << std::endl;
        motor.emergencyShutdown(motorIds);
        return 1;
    }

    std::cout << "\nSmall radians pose reached. Torque is holding." << std::endl;

    std::cout << "\nPress ENTER to return to home / 0 rad...";
    std::cin.get();

    if (!motor.moveJointRadiansPose(
            homeRadians,
            speed,
            homeTolerance,
            timeoutSeconds,
            holdTorque
        ))
    {
        std::cerr << "Failed to return to home radians pose." << std::endl;
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

    std::cout << "Full radians pose test complete." << std::endl;

    return 0;
}