/**
 * @file read_motor_positions.cpp
 * @brief Connects to the arm and prints each of the 7 joints' present
 * position in raw ticks. Read-only; does not enable torque or move
 * anything.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "dynamixel_motor.h"

int main()
{
    const char* DEVICENAME = "COM4";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0F;

    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(DEVICENAME, BAUDRATE, PROTOCOL_VERSION);

    if (!motor.connect())
    {
        std::cerr << "Failed to connect to Dynamixel motors on "
                  << DEVICENAME << "." << std::endl;
        return 1;
    }

    std::cout << "\nCurrent motor positions:\n" << std::endl;

    bool allReadsSuccessful = true;

    for (int id : motorIds)
    {
        uint16_t position = 0;

        if (!motor.readPosition(id, position))
        {
            std::cerr << "Motor " << id << ": failed to read position."
                      << std::endl;
            allReadsSuccessful = false;
            continue;
        }

        std::cout << "Motor " << id << ": " << position << " ticks"
                   << std::endl;
    }

    motor.disconnect();

    return allReadsSuccessful ? 0 : 1;
}
