#ifndef DYNAMIXEL_MOTOR_H
#define DYNAMIXEL_MOTOR_H

#include <cstdint>
#include "dynamixel_sdk/dynamixel_sdk.h"

class DynamixelMotor
{
public:
    DynamixelMotor(const char* deviceName, int baudRate, float protocolVersion);
    ~DynamixelMotor();

    bool connect();
    void disconnect();

    bool pingMotor(int motorId);

    bool enableTorque(int motorId);
    bool disableTorque(int motorId);

    bool setGoalPosition(int motorId, uint16_t position);
    bool setMovingSpeed(int motorId, uint16_t speed);

    bool readPosition(int motorId, uint16_t& position);
    bool readSpeed(int motorId, uint16_t& speed);
    bool readLoad(int motorId, uint16_t& load);
    bool readVoltage(int motorId, uint8_t& voltage);
    bool readTemperature(int motorId, uint8_t& temperature);

    bool moveJointRadians(int motorId, double radians);

    double rawPositionToRadians(uint16_t rawPosition) const;
    uint16_t radiansToRawPosition(double radians) const;

private:
    const char* deviceName_;
    int baudRate_;
    float protocolVersion_;
    bool connected_;

    dynamixel::PortHandler* portHandler_;
    dynamixel::PacketHandler* packetHandler_;

    bool checkCommResult(int commResult, uint8_t dxlError, int motorId, const char* action);

    bool isPositionWithinLimit(int motorId, uint16_t position) const;

    static const int ADDR_TORQUE_ENABLE = 24;
    static const int ADDR_GOAL_POSITION = 30;
    static const int ADDR_MOVING_SPEED = 32;
    static const int ADDR_PRESENT_POSITION = 36;
    static const int ADDR_PRESENT_SPEED = 38;
    static const int ADDR_PRESENT_LOAD = 40;
    static const int ADDR_PRESENT_VOLTAGE = 42;
    static const int ADDR_PRESENT_TEMPERATURE = 43;

    static const int TORQUE_ENABLE = 1;
    static const int TORQUE_DISABLE = 0;

    static const uint16_t MIN_RAW_POSITION = 0;
    static const uint16_t MAX_RAW_POSITION = 4095;
};

#endif