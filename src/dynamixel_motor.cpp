#include "dynamixel_motor.h"
#include <iostream>
#include <cmath>

DynamixelMotor::DynamixelMotor(const char* deviceName, int baudRate, float protocolVersion)
    : deviceName_(deviceName),
      baudRate_(baudRate),
      protocolVersion_(protocolVersion),
      connected_(false),
      portHandler_(dynamixel::PortHandler::getPortHandler(deviceName)),
      packetHandler_(dynamixel::PacketHandler::getPacketHandler(protocolVersion))
{
}

DynamixelMotor::~DynamixelMotor()
{
    if (connected_)
    {
        disconnect();
    }
}

bool DynamixelMotor::connect()
{
    if (!portHandler_->openPort())
    {
        std::cerr << "Failed to open port: " << deviceName_ << std::endl;
        return false;
    }

    std::cout << "Opened port successfully." << std::endl;

    if (!portHandler_->setBaudRate(baudRate_))
    {
        std::cerr << "Failed to set baud rate: " << baudRate_ << std::endl;
        portHandler_->closePort();
        return false;
    }

    std::cout << "Baud rate set successfully." << std::endl;
    connected_ = true;
    return true;
}

void DynamixelMotor::disconnect()
{
    if (connected_)
    {
        portHandler_->closePort();
        connected_ = false;
        std::cout << "Closed port." << std::endl;
    }
}

bool DynamixelMotor::checkCommResult(int commResult, uint8_t dxlError, int motorId, const char* action)
{
    if (commResult != COMM_SUCCESS)
    {
        std::cerr << action << " failed for motor ID " << motorId << ": "
                  << packetHandler_->getTxRxResult(commResult)
                  << std::endl;
        return false;
    }

    if (dxlError != 0)
    {
        std::cerr << "Dynamixel error during " << action
                  << " for motor ID " << motorId << ": "
                  << packetHandler_->getRxPacketError(dxlError)
                  << std::endl;
        return false;
    }

    return true;
}

bool DynamixelMotor::pingMotor(int motorId)
{
    uint8_t dxlError = 0;
    uint16_t modelNumber = 0;

    int commResult = packetHandler_->ping(
        portHandler_,
        motorId,
        &modelNumber,
        &dxlError
    );

    if (!checkCommResult(commResult, dxlError, motorId, "Ping"))
    {
        return false;
    }

    std::cout << "Ping successful. Motor ID " << motorId
              << " model number: " << modelNumber << std::endl;

    return true;
}

bool DynamixelMotor::enableTorque(int motorId)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_TORQUE_ENABLE,
        TORQUE_ENABLE,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Enable torque");
}

bool DynamixelMotor::disableTorque(int motorId)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->write1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_TORQUE_ENABLE,
        TORQUE_DISABLE,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Disable torque");
}

bool DynamixelMotor::setGoalPosition(int motorId, uint16_t position)
{
    if (position > MAX_RAW_POSITION)
    {
        std::cerr << "Goal position out of range: " << position
                  << ". Valid range is 0 to " << MAX_RAW_POSITION << std::endl;
        return false;
    }

    uint8_t dxlError = 0;

    int commResult = packetHandler_->write2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_GOAL_POSITION,
        position,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Set goal position");
}

bool DynamixelMotor::setMovingSpeed(int motorId, uint16_t speed)
{
    if (speed > 1023)
    {
        std::cerr << "Moving speed out of range: " << speed
                  << ". Valid range is 0 to 1023." << std::endl;
        return false;
    }

    uint8_t dxlError = 0;

    int commResult = packetHandler_->write2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_MOVING_SPEED,
        speed,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Set moving speed");
}

bool DynamixelMotor::readPosition(int motorId, uint16_t& position)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_POSITION,
        &position,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read position");
}

bool DynamixelMotor::readSpeed(int motorId, uint16_t& speed)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_SPEED,
        &speed,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read speed");
}

bool DynamixelMotor::readLoad(int motorId, uint16_t& load)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_LOAD,
        &load,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read load");
}

bool DynamixelMotor::readVoltage(int motorId, uint8_t& voltage)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_VOLTAGE,
        &voltage,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read voltage");
}

bool DynamixelMotor::readTemperature(int motorId, uint8_t& temperature)
{
    uint8_t dxlError = 0;

    int commResult = packetHandler_->read1ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_TEMPERATURE,
        &temperature,
        &dxlError
    );

    return checkCommResult(commResult, dxlError, motorId, "Read temperature");
}

bool DynamixelMotor::moveJointRadians(int motorId, double radians)
{
    uint16_t rawPosition = radiansToRawPosition(radians);
    return setGoalPosition(motorId, rawPosition);
}

double DynamixelMotor::rawPositionToRadians(uint16_t rawPosition) const
{
    if (rawPosition > MAX_RAW_POSITION)
    {
        rawPosition = MAX_RAW_POSITION;
    }

    const double PI = 3.14159265358979323846;

    // AX-style Dynamixel motors commonly use 300 degrees of position range.
    // 300 degrees = 5.23599 radians.
    const double rangeRadians = 300.0 * PI / 180.0;

    double normalized = static_cast<double>(rawPosition) / static_cast<double>(MAX_RAW_POSITION);

    // Convert from raw range [0, 1023] to radians around center:
    // raw 0    -> about -150 degrees
    // raw 512  -> about 0 degrees
    // raw 1023 -> about +150 degrees
    return (normalized * rangeRadians) - (rangeRadians / 2.0);
}

uint16_t DynamixelMotor::radiansToRawPosition(double radians) const
{
    const double PI = 3.14159265358979323846;

    const double rangeRadians = 300.0 * PI / 180.0;
    const double minRadians = -rangeRadians / 2.0;
    const double maxRadians =  rangeRadians / 2.0;

    if (radians < minRadians)
    {
        radians = minRadians;
    }

    if (radians > maxRadians)
    {
        radians = maxRadians;
    }

    double normalized = (radians - minRadians) / rangeRadians;
    double raw = normalized * static_cast<double>(MAX_RAW_POSITION);

    return static_cast<uint16_t>(std::round(raw));
}