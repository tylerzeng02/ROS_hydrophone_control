#include <iostream>
#include "dynamixel_sdk/dynamixel_sdk.h"

int main()
{
    // modify according to dynamixel wizard
    const char* DEVICENAME = "/dev/ttyUSB0";
    const int BAUDRATE = 1000000;
    const int DXL_ID = 1;
    const float PROTOCOL_VERSION = 1.0;

    // control table addresses 
    const int ADDR_TORQUE_ENABLE = 24;
    const int ADDR_GOAL_POSITION = 30;
    const int ADDR_PRESENT_POSITION = 36;

    const int TORQUE_ENABLE = 1;
    const int TORQUE_DISABLE = 0;

    dynamixel::PortHandler* portHandler =
        dynamixel::PortHandler::getPortHandler(DEVICENAME);

    dynamixel::PacketHandler* packetHandler =
        dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    if (!portHandler->openPort())
    {
        std::cerr << "Failed to open port: " << DEVICENAME << std::endl;
        return 1;
    }

    std::cout << "Opened port successfully." << std::endl;

    if (!portHandler->setBaudRate(BAUDRATE))
    {
        std::cerr << "Failed to set baud rate: " << BAUDRATE << std::endl;
        portHandler->closePort();
        return 1;
    }

    std::cout << "Baud rate set successfully." << std::endl;

    uint8_t dxl_error = 0;
    int dxl_comm_result = COMM_TX_FAIL;

    // Read current position first
    uint16_t present_position = 0;

    dxl_comm_result = packetHandler->read2ByteTxRx(
        portHandler,
        DXL_ID,
        ADDR_PRESENT_POSITION,
        &present_position,
        &dxl_error
    );

    if (dxl_comm_result != COMM_SUCCESS)
    {
        std::cerr << "Read failed: "
                  << packetHandler->getTxRxResult(dxl_comm_result)
                  << std::endl;
        portHandler->closePort();
        return 1;
    }

    if (dxl_error != 0)
    {
        std::cerr << "Dynamixel error: "
                  << packetHandler->getRxPacketError(dxl_error)
                  << std::endl;
    }

    std::cout << "Current position: " << present_position << std::endl;

    portHandler->closePort();
    std::cout << "Closed port." << std::endl;

    return 0;
}