#include "dynamixel_motor.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>

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

bool DynamixelMotor::isPositionWithinLimit(int motorId, uint16_t position) const
{
    switch (motorId)
    {
        case 0:
            return position >= 0 && position <= 4095;

        case 1:
            return position >= 855 && position <= 3245;

        case 2:
            return position >= 855 && position <= 3245;

        case 3:
            return position >= 855 && position <= 3245;

        case 4:
            return position >= 855 && position <= 3245;

        case 5:
            return position >= 855 && position <= 3245;

        case 6:
            return position >= 0 && position <= 4095;

        case 7:
            return position >= 1578 && position <= 3172;

        default:
            std::cerr << "Unknown motor ID: " << motorId << std::endl;
            return false;
    }
}

bool DynamixelMotor::isPositionSafe(int motorId, uint16_t position) const
{
    return isPositionWithinLimit(motorId, position);
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
        std::cerr << "Goal position out of full MX range: " << position
                  << ". Valid full range is 0 to " << MAX_RAW_POSITION << std::endl;
        return false;
    }

    if (!isPositionWithinLimit(motorId, position))
    {
        std::cerr << "Command cancelled. Goal position " << position
                  << " is outside the safe joint range for motor ID "
                  << motorId << std::endl;
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

bool DynamixelMotor::moveJointSafely(
    int motorId,
    uint16_t targetPosition,
    uint16_t speed,
    int tolerance,
    int timeoutSeconds
)
{
    if (!isPositionWithinLimit(motorId, targetPosition))
    {
        std::cerr << "Target position " << targetPosition
                  << " is outside the safe range for motor ID "
                  << motorId << std::endl;
        return false;
    }

    uint16_t startPosition = 0;
    if (!readPosition(motorId, startPosition))
    {
        std::cerr << "Could not read starting position for motor ID "
                  << motorId << std::endl;
        return false;
    }

    if (!isPositionWithinLimit(motorId, startPosition))
    {
        std::cerr << "Starting position " << startPosition
                  << " is outside the safe range for motor ID "
                  << motorId << std::endl;
        return false;
    }

    std::cout << "Motor " << motorId
              << " starting position: " << startPosition << std::endl;

    if (!enableTorque(motorId))
    {
        std::cerr << "Failed to enable torque on motor ID "
                  << motorId << std::endl;
        return false;
    }

    if (!setMovingSpeed(motorId, speed))
    {
        std::cerr << "Failed to set moving speed on motor ID "
                  << motorId << std::endl;
        disableTorque(motorId);
        return false;
    }

    if (!setGoalPosition(motorId, targetPosition))
    {
        std::cerr << "Failed to set goal position on motor ID "
                  << motorId << std::endl;
        disableTorque(motorId);
        return false;
    }

    std::cout << "Move command sent. Monitoring motor "
              << motorId << "..." << std::endl;

    auto startTime = std::chrono::steady_clock::now();

    while (true)
    {
        uint16_t currentPosition = 0;

        if (!readPosition(motorId, currentPosition))
        {
            std::cerr << "Failed to read position during movement. Stopping motor."
                      << std::endl;
            disableTorque(motorId);
            return false;
        }

        if (!isPositionWithinLimit(motorId, currentPosition))
        {
            std::cerr << "SAFETY STOP: motor " << motorId
                      << " moved outside safe range at position "
                      << currentPosition << std::endl;

            disableTorque(motorId);
            return false;
        }

        int error = static_cast<int>(targetPosition) - static_cast<int>(currentPosition);

        if (error < 0)
        {
            error = -error;
        }

        std::cout << "Current position: " << currentPosition
                  << " | Error: " << error << std::endl;

        if (error <= tolerance)
        {
            std::cout << "Target reached." << std::endl;
            disableTorque(motorId);
            return true;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsedSeconds =
            std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

        if (elapsedSeconds >= timeoutSeconds)
        {
            std::cerr << "Timeout: motor did not reach target in time." << std::endl;
            disableTorque(motorId);
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
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

    // MX position range:
    // raw 0    -> about -pi radians
    // raw 2048 -> about 0 radians
    // raw 4095 -> about +pi radians
    const double rangeRadians = 2.0 * PI;

    double normalized =
        static_cast<double>(rawPosition) / static_cast<double>(MAX_RAW_POSITION);

    return (normalized * rangeRadians) - PI;
}

uint16_t DynamixelMotor::radiansToRawPosition(double radians) const
{
    const double PI = 3.14159265358979323846;

    if (radians < -PI)
    {
        radians = -PI;
    }

    if (radians > PI)
    {
        radians = PI;
    }

    double normalized = (radians + PI) / (2.0 * PI);
    double raw = normalized * static_cast<double>(MAX_RAW_POSITION);

    return static_cast<uint16_t>(std::round(raw));
}