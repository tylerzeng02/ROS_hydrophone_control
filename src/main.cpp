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

    uint16_t position = 0;
    int motorId = 3;

    if (motor.readPosition(motorId, position))
    {
        std::cout << "Motor " << motorId << " position: " << position << std::endl;
    }

    motor.disconnect();

    return 0;
}