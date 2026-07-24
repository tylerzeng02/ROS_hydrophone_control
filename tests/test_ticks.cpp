#include <iostream>
#include <cstdint>
#include <limits>

#include "dynamixel_motor.h"

void waitForEnter(const std::string& message)
{
    std::cout << message;
    std::cin.ignore(
        std::numeric_limits<std::streamsize>::max(),
        '\n'
    );
}

int main()
{
    const char* DEVICENAME = "/dev/ttyUSB1";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0;

    // Change these values.
    const int MOTOR_ID = 1;
    const uint16_t TARGET_POSITION = 2041;

    const uint16_t SPEED = 20;
    const int TOLERANCE = 5;
    const int TIMEOUT_SECONDS = 15;

    DynamixelMotor motor(
        DEVICENAME,
        BAUDRATE,
        PROTOCOL_VERSION
    );

    if (!motor.connect())
    {
        std::cerr << "Failed to connect to the Dynamixel motors."
                  << std::endl;
        return 1;
    }

    uint16_t currentPosition = 0;

    if (!motor.readPosition(MOTOR_ID, currentPosition))
    {
        std::cerr << "Failed to read motor "
                  << MOTOR_ID << "."
                  << std::endl;

        motor.disconnect();
        return 1;
    }

    std::cout << "Motor " << MOTOR_ID
              << " current position: "
              << currentPosition << " ticks"
              << std::endl;

    if (!motor.isPositionSafe(MOTOR_ID, TARGET_POSITION))
    {
        std::cerr << "Target position "
                  << TARGET_POSITION
                  << " is outside the safe limits for motor "
                  << MOTOR_ID << "."
                  << std::endl;

        motor.disconnect();
        return 1;
    }

    waitForEnter(
        "\nPress Enter to enable torque..."
    );

    if (!motor.enableTorque(MOTOR_ID))
    {
        std::cerr << "Failed to enable torque for motor "
                  << MOTOR_ID << "."
                  << std::endl;

        motor.disconnect();
        return 1;
    }

    waitForEnter(
        "Press Enter to move the motor..."
    );

    std::cout << "Moving motor "
              << MOTOR_ID
              << " to "
              << TARGET_POSITION
              << " ticks..."
              << std::endl;

    bool moveSuccessful = motor.moveJointSafely(
        MOTOR_ID,
        TARGET_POSITION,
        SPEED,
        TOLERANCE,
        TIMEOUT_SECONDS
    );

    if (moveSuccessful)
    {
        uint16_t finalPosition = 0;

        if (motor.readPosition(MOTOR_ID, finalPosition))
        {
            std::cout << "Motor reached "
                      << finalPosition
                      << " ticks."
                      << std::endl;
        }
        else
        {
            std::cout << "Movement completed, but the final "
                         "position could not be read."
                      << std::endl;
        }
    }
    else
    {
        std::cerr << "Motor movement failed or timed out."
                  << std::endl;
    }

    waitForEnter(
        "\nPress Enter to disable torque and end..."
    );

    if (!motor.disableTorque(MOTOR_ID))
    {
        std::cerr << "Failed to disable torque."
                  << std::endl;
    }

    motor.disconnect();

    std::cout << "Program ended." << std::endl;

    return moveSuccessful ? 0 : 1;
}