/**
 * @file disable_torque_prompt.cpp
 * @brief Waits for Enter, then disables torque on all 7 joints and
 * disconnects.
 *
 * Meant to be run once, as the last step after a sequence of other tools
 * (e.g. a sweep script running --validate repeatedly across many I-gain
 * candidates) that deliberately leave torque enabled across their own
 * process exits, so the arm stays supported for the whole sequence and
 * only gets released here, on explicit confirmation.
 */
#include <iostream>
#include <string>

#include "dynamixel_motor.h"

namespace {
constexpr const char* CYTON_DEVICE = "COM4";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;
}  // namespace

int main() {
    DynamixelMotor motor(CYTON_DEVICE, CYTON_BAUD_RATE, CYTON_PROTOCOL_VERSION);

    if (!motor.connect()) {
        std::cerr << "Could not connect to the Cyton motors.\n";
        return 1;
    }

    std::cout << "Torque is currently held on all joints. Press Enter to "
                 "disable torque and disconnect..." << std::flush;
    std::string line;
    std::getline(std::cin, line);

    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};
    for (int id : motorIds) {
        motor.disableTorque(id);
    }
    std::cout << "Torque disabled on all 7 joints.\n";

    motor.disconnect();
    return 0;
}
