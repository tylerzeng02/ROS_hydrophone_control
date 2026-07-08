#include <iostream>
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

    int motorId = 3;

    motor.pingMotor(motorId);

    uint16_t position = 0;
    if (motor.readPosition(motorId, position))
    {
        std::cout << "Motor " << motorId
                  << " raw position: " << position
                  << std::endl;

        std::cout << "Motor " << motorId
                  << " radians: "
                  << motor.rawPositionToRadians(position)
                  << std::endl;
    }

    uint8_t voltage = 0;
    if (motor.readVoltage(motorId, voltage))
    {
        std::cout << "Voltage: "
                  << static_cast<int>(voltage) / 10.0
                  << " V" << std::endl;
    }

    uint8_t temperature = 0;
    if (motor.readTemperature(motorId, temperature))
    {
        std::cout << "Temperature: "
                  << static_cast<int>(temperature)
                  << " C" << std::endl;
    }

    motor.disconnect();

    return 0;
}