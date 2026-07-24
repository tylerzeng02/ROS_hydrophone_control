#include <cstdint>
#include <iostream>
#include <vector>

#include "dynamixel_motor.h"

int main()
{
    const char* DEVICENAME = "/dev/ttyUSB1";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0F;

    const std::vector<int> motorIds = {
        0, 1, 2, 3, 4, 5, 6, 7
    };

    DynamixelMotor motor(
        DEVICENAME,
        BAUDRATE,
        PROTOCOL_VERSION
    );

    if (!motor.connect())
    {
        std::cerr << "Failed to connect to Dynamixel motors."
                  << std::endl;
        return 1;
    }

    std::cout << "\nMake sure the arm is clear and supported."
              << std::endl;
    std::cout << "Press Enter to enable torque on all motors...";
    std::cin.get();

    for (int id : motorIds)
    {
        if (!motor.enableTorque(id))
        {
            std::cerr << "\nFailed to enable torque on motor "
                      << id << "." << std::endl;

            for (int disableId : motorIds)
            {
                motor.disableTorque(disableId);
            }

            motor.disconnect();
            return 1;
        }
    }

    std::cout << "\nTorque enabled. No goal positions were sent."
              << std::endl;
    std::cout << "Press Enter to read all motor positions in ticks...";
    std::cin.get();

    std::cout << "\nCurrent motor positions:\n" << std::endl;

    bool allReadsSuccessful = true;

    for (int id : motorIds)
    {
        uint16_t position = 0;

        if (!motor.readPosition(id, position))
        {
            std::cerr << "Motor " << id
                      << ": failed to read position."
                      << std::endl;

            allReadsSuccessful = false;
            continue;
        }

        std::cout << "Motor " << id
                  << ": " << position
                  << " ticks" << std::endl;
    }

    std::cout << "\nPress Enter to disable torque and end the program...";
    std::cin.get();

    bool allDisablesSuccessful = true;

    for (int id : motorIds)
    {
        if (!motor.disableTorque(id))
        {
            std::cerr << "\nFailed to disable torque on motor "
                      << id << "." << std::endl;

            allDisablesSuccessful = false;
        }
    }

    motor.disconnect();

    std::cout << "\nTorque disabled. Program ended." << std::endl;

    if (!allReadsSuccessful || !allDisablesSuccessful)
    {
        return 1;
    }

    return 0;
}