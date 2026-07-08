#include "dynamixel_motor.h"
#include <iostream>

DynamixelMotor::DynamixelMotor(const char* deviceName, int baudRate, float protocolVersion)
    : deviceName_(deviceName),
      baudRate_(baudRate),
      protocolVersion_(protocolVersion),
      portHandler_(dynamixel::PortHandler::getPortHandler(deviceName)),
      packetHandler_(dynamixel::PacketHandler::getPacketHandler(protocolVersion))
{
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
    return true;
}

void DynamixelMotor::disconnect()
{
    portHandler_->closePort();
    std::cout << "Closed port." << std::endl;
}

bool DynamixelMotor::readPosition(int motorId, uint16_t& position)
{
    uint8_t dxl_error = 0;

    int dxl_comm_result = packetHandler_->read2ByteTxRx(
        portHandler_,
        motorId,
        ADDR_PRESENT_POSITION,
        &position,
        &dxl_error
    );

    if (dxl_comm_result != COMM_SUCCESS)
    {
        std::cerr << "Read failed for motor ID " << motorId << ": "
                  << packetHandler_->getTxRxResult(dxl_comm_result)
                  << std::endl;
        return false;
    }

    if (dxl_error != 0)
    {
        std::cerr << "Dynamixel error for motor ID " << motorId << ": "
                  << packetHandler_->getRxPacketError(dxl_error)
                  << std::endl;
        return false;
    }

    return true;
}