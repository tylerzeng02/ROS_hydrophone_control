// set_i_gain: writes a given I Gain to one motor and LEAVES it applied --
// deliberately the opposite of test_i_gain.cpp/autotune_i_gain.cpp, which
// always restore the original value at the end. This is for actually
// applying a candidate value found by autotune_i_gain.cpp so its real
// effect can be measured against a full accuracy test (e.g.
// test_five_pose_ndi_capture --validate), not just the tick-error sweep
// that found it.
//
// Reminder (see dynamixel_motor.h's compliance-margin comment block): this
// register lives in RAM, not EEPROM, so it will NOT survive a power
// cycle -- this only affects the CURRENT session. Does not move the arm.
//
// Usage: set_i_gain <motorId> <iGainValue>

#include <iostream>
#include "dynamixel_motor.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: set_i_gain <motorId> <iGainValue>\n";
        return 1;
    }
    const int motorId = std::stoi(argv[1]);
    const uint8_t iGain = static_cast<uint8_t>(std::stoi(argv[2]));

    DynamixelMotor motor("/dev/ttyUSB0", 1000000, 1.0F);
    if (!motor.connect()) {
        std::cerr << "Could not connect to /dev/ttyUSB0.\n";
        return 1;
    }
    if (!motor.pingMotor(motorId)) {
        std::cerr << "Could not ping motor " << motorId << ".\n";
        motor.disconnect();
        return 1;
    }

    uint8_t dGain = 0, oldIGain = 0, pGain = 0;
    if (!motor.readGains(motorId, dGain, oldIGain, pGain)) {
        std::cerr << "Could not read current gains.\n";
        motor.disconnect();
        return 1;
    }
    std::cout << "Motor " << motorId << " current gains: D=" << static_cast<int>(dGain)
              << " I=" << static_cast<int>(oldIGain) << " P=" << static_cast<int>(pGain) << '\n';

    if (!motor.writeGains(motorId, dGain, iGain, pGain)) {
        std::cerr << "FAILED to write new I gain.\n";
        motor.disconnect();
        return 1;
    }

    uint8_t confirmD = 0, confirmI = 0, confirmP = 0;
    motor.readGains(motorId, confirmD, confirmI, confirmP);
    std::cout << "Motor " << motorId << " gains now: D=" << static_cast<int>(confirmD)
              << " I=" << static_cast<int>(confirmI) << " P=" << static_cast<int>(confirmP)
              << (confirmI == iGain ? " (confirmed)\n" : " (WARNING: I gain did not confirm)\n")
              << "Left applied -- this is RAM, not EEPROM, so it resets to factory default on the "
                 "next power cycle, but persists for the rest of this session.\n";

    motor.disconnect();
    return 0;
}
