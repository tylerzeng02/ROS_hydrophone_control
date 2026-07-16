#include <iostream>
#include <vector>
#include "dynamixel_motor.h"

int main()
{
    const char* DEVICENAME = "/dev/ttyUSB1";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0;

    DynamixelMotor motor(
        DEVICENAME,
        BAUDRATE,
        PROTOCOL_VERSION
    );

    if (!motor.connect())
    {
        std::cerr << "Failed to connect to Dynamixel motors." << std::endl;
        return 1;
    }

    // Motor order:
    // {0, 1, 2, 3, 4, 5, 6, 7}
    std::vector<int> motorIds = {
        0, 1, 2, 3, 4, 5, 6, 7
    };

    /*
     * Change these values to change the home position.
     *
     * Motor 7 is the gripper.
     */
    std::vector<uint16_t> homePose = {
        2048, // Motor 0
        2048, // Motor 1
        2066, // Motor 2
        2108, // Motor 3
        2078, // Motor 4
        2048, // Motor 5
        2048, // Motor 6
        2048  // Motor 7
    };

    const uint16_t movingSpeed = 20;
    const int positionTolerance = 6;
    const int timeoutSeconds = 20;

    std::cout << "\nMake sure the arm is clear." << std::endl;
    std::cout << "The motors will hold their current positions when torque is enabled."
              << std::endl;
    std::cout << "Press Enter to enable torque on all motors...";
    std::cin.get();

    /*
     * Read each motor's current position before enabling torque.
     * Set that position as its goal so it does not suddenly move when
     * torque is enabled.
     */
    std::vector<uint16_t> currentPose;
    currentPose.reserve(motorIds.size());

    for (int id : motorIds)
    {
        uint16_t position = 0;

        if (!motor.readPosition(id, position))
        {
            std::cerr
                << "Failed to read motor ID "
                << id
                << "."
                << std::endl;

            motor.emergencyShutdown(motorIds);
            return 1;
        }

        currentPose.push_back(position);

        std::cout
            << "Motor "
            << id
            << " current position: "
            << position
            << " ticks"
            << std::endl;
    }

    // Check the current positions before enabling torque.
    bool allMotorsSafe = true;

    std::cout << "\nChecking current motor positions..." << std::endl;

    for (std::size_t i = 0; i < motorIds.size(); ++i)
    {
        const int id = motorIds[i];
        const uint16_t position = currentPose[i];

        std::cout
            << "Motor "
            << id
            << ": "
            << position;

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
        std::cerr << "Move the arm manually into its safe range first."
                  << std::endl;

        motor.emergencyShutdown(motorIds);
        return 1;
    }

    // Verify that every requested home position is safe.
    std::cout << "\nChecking requested home position..." << std::endl;

    for (std::size_t i = 0; i < motorIds.size(); ++i)
    {
        const int id = motorIds[i];
        const uint16_t homePosition = homePose[i];

        std::cout
            << "Motor "
            << id
            << " home position: "
            << homePosition;

        if (motor.isPositionSafe(id, homePosition))
        {
            std::cout << " safe" << std::endl;
        }
        else
        {
            std::cout << " OUTSIDE SAFE RANGE" << std::endl;

            std::cerr
                << "\nUnsafe home position detected for motor "
                << id
                << "."
                << std::endl;

            motor.emergencyShutdown(motorIds);
            return 1;
        }
    }

    /*
     * Set each goal position to the motor's current position before
     * enabling torque. This makes the motors hold where they currently are.
     */
    for (std::size_t i = 0; i < motorIds.size(); ++i)
    {
        const int id = motorIds[i];

        if (!motor.setGoalPosition(id, currentPose[i]))
        {
            std::cerr
                << "Failed to set the initial holding position for motor "
                << id
                << "."
                << std::endl;

            motor.emergencyShutdown(motorIds);
            return 1;
        }
    }

    // Enable torque while holding the current positions.
    for (int id : motorIds)
    {
        if (!motor.enableTorque(id))
        {
            std::cerr
                << "Failed to enable torque on motor "
                << id
                << "."
                << std::endl;

            motor.emergencyShutdown(motorIds);
            return 1;
        }
    }

    std::cout << "\nTorque enabled." << std::endl;
    std::cout << "The motors are holding their current positions."
              << std::endl;

    std::cout << "\nElectrical status before movement:" << std::endl;
    motor.printElectricalStatusForMotors(motorIds);

    std::cout << "\nRequested home pose:" << std::endl;

    for (std::size_t i = 0; i < motorIds.size(); ++i)
    {
        std::cout
            << "Motor "
            << motorIds[i]
            << ": "
            << homePose[i]
            << " ticks"
            << std::endl;
    }

    std::cout << "\nMake sure the arm is clear and your hand is near the power switch."
              << std::endl;
    std::cout << "Press Enter to move to the home position...";
    std::cin.get();

    if (!motor.moveToPose(
            homePose,
            movingSpeed,
            positionTolerance,
            timeoutSeconds,
            true))
    {
        std::cerr << "Failed to move to the home position." << std::endl;
        motor.emergencyShutdown(motorIds);
        return 1;
    }

    std::string homeElectricalSnapshot =
        motor.getLastElectricalSnapshot();

    std::cout << "\nHome position reached." << std::endl;
    std::cout << "Torque is still enabled and holding the arm."
              << std::endl;

    std::cout << "\nElectrical snapshot during movement:" << std::endl;
    std::cout << homeElectricalSnapshot << std::endl;

    std::cout << "\nPress Enter to disable torque and end the program...";
    std::cin.get();

    for (int id : motorIds)
    {
        motor.disableTorque(id);
    }

    motor.disconnect();

    std::cout << "\nTorque disabled. Program ended." << std::endl;

    return 0;
}
