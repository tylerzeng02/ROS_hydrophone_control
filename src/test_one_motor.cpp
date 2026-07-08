#include <iostream>
#include "dynamixel_sdk/dynamixel_sdk.h"

int main()
{
    const char* DEVICENAME = "/dev/ttyUSB0";
    const int BAUDRATE = 1000000;
    const float PROTOCOL_VERSION = 1.0;

    const int ADDR_PRESENT_POSITION = 36;

    int motorIds[] = {0, 1, 2, 3, 4, 5, 6, 7};

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

    for (int id : motorIds)
    {
        uint8_t dxl_error = 0;
        uint16_t position = 0;

        int dxl_comm_result = packetHandler->read2ByteTxRx(
            portHandler,
            id,
            ADDR_PRESENT_POSITION,
            &position,
            &dxl_error
        );

        if (dxl_comm_result == COMM_SUCCESS && dxl_error == 0)
        {
            std::cout << "Motor " << id << " position: " << position << std::endl;
        }
        else
        {
            std::cout << "Failed to read motor " << id << ": "
                      << packetHandler->getTxRxResult(dxl_comm_result)
                      << std::endl;

            if (dxl_error != 0)
            {
                std::cout << "Dynamixel error: "
                          << packetHandler->getRxPacketError(dxl_error)
                          << std::endl;
            }
        }
    }

    portHandler->closePort();
    std::cout << "Closed port." << std::endl;

    return 0;
}