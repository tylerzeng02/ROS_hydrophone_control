#include <iostream>
#include <vector>
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

    std::cout << "\nChecking all motor positions before enabling torque..." << std::endl;

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
        std::cerr << "Torque will NOT be enabled." << std::endl;
        std::cerr << "Move the arm manually into a safe range first." << std::endl;

        motor.disconnect();
        return 1;
    }

    std::cout << "\nAll motors are inside safe ranges." << std::endl;

    const int testMotorId = 5;
    const uint16_t targetPosition = 2260;
    const uint16_t movingSpeed = 50;

    std::cout << "\nMake sure the arm is clear and your hand is near the power switch." << std::endl;
    std::cout << "Press Enter to move motor " << testMotorId << "...";
    std::cin.get();

    bool success = motor.moveJointSafely(
        testMotorId,
        targetPosition,
        movingSpeed
    );

    if (!success)
    {
        std::cerr << "Safe movement failed or was stopped." << std::endl;
        motor.disconnect();
        return 1;
    }

    motor.disconnect();

    return 0;
}