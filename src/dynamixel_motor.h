#ifndef DYNAMIXEL_MOTOR_H
#define DYNAMIXEL_MOTOR_H

#include <cstdint>
#include "dynamixel_sdk/dynamixel_sdk.h"

class DynamixelMotor
{
public:
    DynamixelMotor(const char* deviceName, int baudRate, float protocolVersion);

    bool connect();
    void disconnect();

    bool readPosition(int motorId, uint16_t& position);

private:
    const char* deviceName_;
    int baudRate_;
    float protocolVersion_;

    dynamixel::PortHandler* portHandler_;
    dynamixel::PacketHandler* packetHandler_;

    static const int ADDR_PRESENT_POSITION = 36;
};

#endif