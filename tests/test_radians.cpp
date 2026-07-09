#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

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

    const int jointIndex = 0;
    const JointCalibration& joint = jointCalibrations[jointIndex];

    const int motorId = joint.id;
    const double testRadians = 0.05;
    const uint16_t speed = 40;
    const int tolerance = 10;
    const int timeoutSeconds = 8;

    int homeTick = radiansToTicks(joint, 0.0);
    int testTick = radiansToTicks(joint, testRadians);

    std::cout << "Testing motor ID " << motorId << std::endl;
    std::cout << "Home tick: " << homeTick << std::endl;
    std::cout << "Target for +" << testRadians << " rad: " << testTick << std::endl;

    if (!motor.isPositionSafe(motorId, static_cast<uint16_t>(homeTick)))
    {
        std::cerr << "Home tick is outside safe range. Cancelling." << std::endl;
        motor.disconnect();
        return 1;
    }

    if (!motor.isPositionSafe(motorId, static_cast<uint16_t>(testTick)))
    {
        std::cerr << "Test tick is outside safe range. Cancelling." << std::endl;
        motor.disconnect();
        return 1;
    }

    std::cout << "\nThis will move motor " << motorId
              << " from home to +" << testRadians
              << " rad, then back home." << std::endl;

    std::cout << "Press ENTER to continue, or Ctrl+C to cancel..." << std::endl;
    std::cin.get();

    std::cout << "Moving motor " << motorId << " to test position..." << std::endl;

    if (!motor.moveJointSafely(
            motorId,
            static_cast<uint16_t>(testTick),
            speed,
            tolerance,
            timeoutSeconds
        ))
    {
        std::cerr << "Failed to move motor to test position." << std::endl;
        motor.disconnect();
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    uint16_t currentPosition = 0;
    if (motor.readPosition(motorId, currentPosition))
    {
        double currentRadians = ticksToRadians(joint, currentPosition);

        std::cout << "Current tick after test move: " << currentPosition << std::endl;
        std::cout << "Current radians from home: " << currentRadians << std::endl;
    }

    std::cout << "Returning motor " << motorId << " to home..." << std::endl;

    if (!motor.moveJointSafely(
            motorId,
            static_cast<uint16_t>(homeTick),
            speed,
            tolerance,
            timeoutSeconds
        ))
    {
        std::cerr << "Failed to return motor home." << std::endl;
        motor.disconnect();
        return 1;
    }

    motor.disconnect();

    std::cout << "Radians test complete." << std::endl;
    return 0;
}