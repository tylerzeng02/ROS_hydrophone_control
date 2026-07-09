#include <iostream>
#include <vector>
#include <string>
#include "dynamixel_motor.h"

int main()
{
    const char* DEVICENAME = "/dev/ttyUSB0";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0;

    DynamixelMotor motor(DEVICENAME, BAUDRATE, PROTOCOL_VERSION);

    if (!motor.connect())
    {
        return 1;
    }

    std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6, 7};

    std::cout << "\nMake sure the arm is clear." << std::endl;
    std::cout << "Press Enter to enable torque on all motors...";
    std::cin.get();

    for (int id : motorIds)
    {
        if (!motor.enableTorque(id))
        {
            std::cerr << "Failed to enable torque on motor " << id << std::endl;

            for (int disableId : motorIds)
            {
                motor.disableTorque(disableId);
            }

            motor.disconnect();
            return 1;
        }
    }

    std::cout << "\nTorque enabled on all motors." << std::endl;
    std::cout << "\nChecking all motor positions after enabling torque..." << std::endl;

    bool allMotorsSafe = true;

    for (int id : motorIds)
    {
        uint16_t position = 0;

        if (!motor.readPosition(id, position))
        {
            std::cerr << "Failed to read motor ID " << id << std::endl;
            allMotorsSafe = false;
            continue;
        }

        std::cout << "Motor " << id << " position: " << position;

        if (motor.isPositionSafe(id, position))
        {
            std::cout << " safe" << std::endl;
        }
        else
        {
            std::cout << " OUTSIDE SAFE RANGE" << std::endl;
            allMotorsSafe = false;
        }
    }

    if (!allMotorsSafe)
    {
        std::cerr << "\nStartup safety check failed." << std::endl;
        std::cerr << "Disabling torque." << std::endl;
        std::cerr << "Move the arm manually into a safe range first." << std::endl;

        for (int id : motorIds)
        {
            motor.disableTorque(id);
        }

        motor.disconnect();
        return 1;
    }

    std::cout << "\nAll motors are inside safe ranges." << std::endl;
    std::cout << "\nElectrical status before movement:" << std::endl;
    motor.printElectricalStatusForMotors(motorIds);

    // Named poses.
    // Motor order is always:
    // {0, 1, 2, 3, 4, 5, 6, 7}

    std::vector<uint16_t> homePose = {
        2015, 857, 935, 3239, 1200, 3087, 1967, 2350
    };

    std::vector<uint16_t> testPose1 = {
        1776, 2408, 1144, 2991, 1495, 2312, 1759, 2350
    };

    std::vector<std::vector<uint16_t>> trajectoryToTestPose1 = {
        homePose,
        {1955, 1245, 987, 3177, 1274, 2893, 1915, 2350},
        {1896, 1633, 1040, 3115, 1348, 2700, 1863, 2350},
        {1836, 2020, 1092, 3053, 1421, 2506, 1811, 2350},
        testPose1
    };

    const uint16_t movingSpeed = 20;

    std::cout << "\nNamed pose movement test." << std::endl;
    std::cout << "Moving speed: " << movingSpeed << std::endl;

    std::cout << "\nMake sure the arm is clear and your hand is near the power switch." << std::endl;
    std::cout << "Press Enter to move to test pose 1...";
    std::cin.get();

    for (size_t i = 0; i < trajectoryToTestPose1.size(); ++i)
        {
            std::cout << "\nMoving to trajectory point " << i << "..." << std::endl;

            if (!motor.moveToPose(trajectoryToTestPose1[i], movingSpeed, 15, 20, true))
            {
                std::cerr << "Failed to move to trajectory point " << i << std::endl;
                motor.disconnect();
                return 1;
            }
        }
    std::string testPose1ElectricalSnapshot = motor.getLastElectricalSnapshot();

    std::cout << "\nTest pose 1 reached." << std::endl;
    std::cout << "Torque is still enabled and holding position." << std::endl;
    std::cout << "Press Enter to move back to home pose...";
    std::cin.get();

    if (!motor.moveToPose(homePose, movingSpeed, 13, 20, true))
    {
        std::cerr << "Failed to move back to home pose." << std::endl;
        motor.disconnect();
        return 1;
    }

    std::string homePoseElectricalSnapshot = motor.getLastElectricalSnapshot();

    std::cout << "\nHome pose reached." << std::endl;
    std::cout << "Torque is still enabled and holding position." << std::endl;

    std::cout << "\n===== MID-MOVEMENT ELECTRICAL SNAPSHOTS =====" << std::endl;

    std::cout << "\nDuring movement to testPose1:" << std::endl;
    std::cout << testPose1ElectricalSnapshot << std::endl;

    std::cout << "\nDuring movement back to homePose:" << std::endl;
    std::cout << homePoseElectricalSnapshot << std::endl;

    std::cout << "Press Enter to disable torque and end program...";
    std::cin.get();

    for (int id : motorIds)
    {
        motor.disableTorque(id);
    }

    motor.disconnect();

    return 0;
}