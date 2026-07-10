#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <vector>

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

    const int jointIndex = 7;
    const JointCalibration& joint = jointCalibrations[jointIndex];

    const int motorId = joint.id;
    const double testRadians = 0.05;
    const uint16_t speed = 40;
    const int tolerance =7;
    const int homeTolerance =15;
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

    std::cout << "\nMake sure the arm is clear." << std::endl;
    std::cout << "This test will first move motor " << motorId
              << " to home / 0 rad." << std::endl;
    std::cout << "Then it will move to +" << testRadians
              << " rad, then return home." << std::endl;

    std::cout << "\nPress ENTER to move motor " << motorId
              << " to home position...";
    std::cin.get();

    if (!motor.moveJointsSafely(
            std::vector<int>{motorId},
            std::vector<uint16_t>{static_cast<uint16_t>(homeTick)},
            speed,
            homeTolerance,
            timeoutSeconds,
            true   // keep torque ON after reaching home
        ))
    {
        std::cerr << "Failed to move motor to home position." << std::endl;
        motor.disconnect();
        return 1;
    }

    std::cout << "Motor reached home. Torque is still enabled." << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    uint16_t homeReadback = 0;
    if (motor.readPosition(motorId, homeReadback))
    {
        std::cout << "\nHome readback tick: " << homeReadback << std::endl;
        std::cout << "Home readback radians: "
                  << ticksToRadians(joint, homeReadback) << std::endl;
    }
    else
    {
        std::cerr << "Warning: failed to read home position." << std::endl;
    }

    std::cout << "\nPress ENTER to run calibration move to +"
              << testRadians << " rad...";
    std::cin.get();

    std::cout << "Moving motor " << motorId << " to test position..." << std::endl;

    if (!motor.moveJointsSafely(
            std::vector<int>{motorId},
            std::vector<uint16_t>{static_cast<uint16_t>(testTick)},
            speed,
            tolerance,
            timeoutSeconds,
            true   // keep torque ON after test move too
        ))
    {
        std::cerr << "Failed to move motor to test position." << std::endl;
        motor.disconnect();
        return 1;
    }

    std::cout << "Motor reached test position. Torque is still enabled." << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    uint16_t currentPosition = 0;
    if (motor.readPosition(motorId, currentPosition))
    {
        double currentRadians = ticksToRadians(joint, currentPosition);

        std::cout << "\n===== CALIBRATION RESULT =====" << std::endl;
        std::cout << "Commanded radians: " << testRadians << std::endl;
        std::cout << "Expected target tick: " << testTick << std::endl;
        std::cout << "Actual tick after test move: " << currentPosition << std::endl;
        std::cout << "Current radians from home: " << currentRadians << std::endl;
        std::cout << "Tick error: "
                  << static_cast<int>(testTick) - static_cast<int>(currentPosition)
                  << std::endl;
        std::cout << "Radian error: "
                  << testRadians - currentRadians
                  << std::endl;
    }
    else
    {
        std::cerr << "Warning: failed to read position after test move." << std::endl;
    }

    std::cout << "\nPress ENTER to return motor " << motorId
              << " to home...";
    std::cin.get();

    if (!motor.moveJointsSafely(
            std::vector<int>{motorId},
            std::vector<uint16_t>{static_cast<uint16_t>(homeTick)},
            speed,
            homeTolerance,
            timeoutSeconds,
            true   // keep torque ON after returning home
        ))
    {
        std::cerr << "Failed to return motor home." << std::endl;
        motor.disconnect();
        return 1;
    }

    std::cout << "Motor returned home. Torque is still enabled." << std::endl;

    std::cout << "\nPress ENTER to disable torque and disconnect...";
    std::cin.get();

    motor.disableTorque(motorId);
    motor.disconnect();

    std::cout << "Radians test complete." << std::endl;
    return 0;
}