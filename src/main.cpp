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

    for (int id : motorIds)
    {
        std::cout << "\n--- Motor ID " << id << " ---" << std::endl;

        motor.pingMotor(id);

        uint16_t position = 0;
        if (motor.readPosition(id, position))
        {
            std::cout << "Raw position: " << position << std::endl;
            std::cout << "Radians: " << motor.rawPositionToRadians(position) << std::endl;
        }

        uint8_t voltage = 0;
        if (motor.readVoltage(id, voltage))
        {
            std::cout << "Voltage: " << static_cast<int>(voltage) / 10.0 << " V" << std::endl;
        }

        uint8_t temperature = 0;
        if (motor.readTemperature(id, temperature))
        {
            std::cout << "Temperature: " << static_cast<int>(temperature) << " C" << std::endl;
        }
    }

    std::cout << "\nTesting safety rejection..." << std::endl;

    // This should be rejected because motor 3 safe range is 855–3245.
    if (!motor.setGoalPosition(3, 500))
    {
        std::cout << "Safety test passed: unsafe command was rejected." << std::endl;
    }

    motor.disconnect();

    return 0;
}