#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
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

    // Small one-motor movement test.
    // Motor 6 has full range 0–4095 and was near center in your last test.
    const int testMotorId = 6;
    const uint16_t targetPosition = 2050;
    const uint16_t movingSpeed = 50;

    uint16_t startPosition = 0;
    if (!motor.readPosition(testMotorId, startPosition))
    {
        std::cerr << "Could not read starting position for motor "
                  << testMotorId << std::endl;

        motor.disconnect();
        return 1;
    }

    std::cout << "\nMotor " << testMotorId
              << " starting position: " << startPosition << std::endl;

    if (!motor.isPositionSafe(testMotorId, targetPosition))
    {
        std::cerr << "Target position is not safe. Command cancelled." << std::endl;

        motor.disconnect();
        return 1;
    }

    std::cout << "Target position: " << targetPosition << std::endl;
    std::cout << "Moving speed: " << movingSpeed << std::endl;

    std::cout << "\nMake sure the arm is clear and your hand is near the power switch." << std::endl;
    std::cout << "Press Enter to enable torque and move motor "
              << testMotorId << "...";
    std::cin.get();

    if (!motor.enableTorque(testMotorId))
    {
        std::cerr << "Failed to enable torque on motor "
                  << testMotorId << std::endl;

        motor.disconnect();
        return 1;
    }

    if (!motor.setMovingSpeed(testMotorId, movingSpeed))
    {
        std::cerr << "Failed to set moving speed." << std::endl;

        motor.disableTorque(testMotorId);
        motor.disconnect();
        return 1;
    }

    if (!motor.setGoalPosition(testMotorId, targetPosition))
    {
        std::cerr << "Movement command rejected." << std::endl;

        motor.disableTorque(testMotorId);
        motor.disconnect();
        return 1;
    }

    std::cout << "Move command sent. Waiting 2 seconds..." << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(2));

    uint16_t finalPosition = 0;
    if (motor.readPosition(testMotorId, finalPosition))
    {
        std::cout << "Motor " << testMotorId
                  << " final position: " << finalPosition << std::endl;
    }

    motor.disableTorque(testMotorId);
    motor.disconnect();

    return 0;
}