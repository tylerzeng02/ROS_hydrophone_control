// One-off utility: explicitly verify and reset every motor's compliance
// slope and punch back to factory defaults (32, 32), rather than trusting
// that ndi_capture_and_validate's automatic restore ran (it should have,
// on a normal exit, but this confirms it directly after the compliance/
// punch experiment that caused audible vibration). Read-only on margin/
// torque enable; does not move the arm.
#include <iostream>
#include "dynamixel_motor.h"

int main() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};
    constexpr uint8_t FACTORY_SLOPE = 32;
    constexpr uint16_t FACTORY_PUNCH = 32;

    DynamixelMotor motor("COM4", 1000000, 1.0F);

    if (!motor.connect()) {
        std::cerr << "Could not connect to the Cyton motors." << std::endl;
        return 1;
    }

    bool anyMismatch = false;

    for (int id : motorIds) {
        if (!motor.pingMotor(id)) {
            std::cerr << "Could not ping motor " << id << std::endl;
            continue;
        }

        uint8_t cwSlope = 0, ccwSlope = 0;
        uint16_t punch = 0;
        if (!motor.readComplianceSlopes(id, cwSlope, ccwSlope) ||
            !motor.readPunch(id, punch)) {
            std::cerr << "Could not read compliance/punch for motor " << id << std::endl;
            continue;
        }

        std::cout << "Motor " << id << " BEFORE: CW slope=" << (int)cwSlope
                   << " CCW slope=" << (int)ccwSlope
                   << " punch=" << punch << std::endl;

        if (cwSlope != FACTORY_SLOPE || ccwSlope != FACTORY_SLOPE || punch != FACTORY_PUNCH) {
            anyMismatch = true;
            std::cout << "  -> NOT at factory defaults, resetting..." << std::endl;

            if (!motor.writeComplianceSlopes(id, FACTORY_SLOPE, FACTORY_SLOPE) ||
                !motor.writePunch(id, FACTORY_PUNCH)) {
                std::cerr << "  FAILED to reset motor " << id << std::endl;
                continue;
            }

            uint8_t newCwSlope = 0, newCcwSlope = 0;
            uint16_t newPunch = 0;
            motor.readComplianceSlopes(id, newCwSlope, newCcwSlope);
            motor.readPunch(id, newPunch);
            std::cout << "  Motor " << id << " AFTER:  CW slope=" << (int)newCwSlope
                       << " CCW slope=" << (int)newCcwSlope
                       << " punch=" << newPunch << std::endl;
        } else {
            std::cout << "  -> already at factory defaults." << std::endl;
        }
    }

    std::cout << "\n" << (anyMismatch ? "Some motors needed resetting -- now done."
                                      : "All motors were already at factory defaults.")
              << std::endl;

    motor.disconnect();
    return 0;
}
