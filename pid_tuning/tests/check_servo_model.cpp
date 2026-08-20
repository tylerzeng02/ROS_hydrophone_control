// check_servo_model: reads each joint motor's Model Number register
// (address 0, identical location on every Protocol 1.0 Dynamixel) to
// settle, from real hardware rather than assumption, exactly which servo
// series this arm uses -- this project's code/comments long assumed
// AX-12A, but the servos are actually MX-64/MX-28, which matters a lot:
// on MX-series, register addresses 26/27/28 are D Gain/I Gain/P Gain
// (real PID), not AX-style Compliance Margin/Slope. See
// dynamixel_motor.h's readComplianceMargins()/readComplianceSlopes() for
// why this distinction matters.
//
// Read-only -- does not move the arm, does not write any register.
//
// Known Protocol 1.0 model numbers (from ROBOTIS's own documentation,
// listed here for reference when interpreting this program's output --
// double check against the current ROBOTIS e-Manual if in doubt):
//   AX-12A = 12, AX-12W = 300, AX-18A = 18
//   MX-12W = 360, MX-28 = 29, MX-64 = 310, MX-106 = 320

#include <iostream>
#include "dynamixel_motor.h"

int main() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor("/dev/ttyUSB0", 1000000, 1.0F);
    if (!motor.connect()) {
        std::cerr << "Could not connect to /dev/ttyUSB0.\n";
        return 1;
    }

    std::cout << "motor_id  model_number\n";
    bool anyFailed = false;
    for (int id : motorIds) {
        if (!motor.pingMotor(id)) {
            std::cerr << id << "         (ping failed)\n";
            anyFailed = true;
            continue;
        }
        uint16_t modelNumber = 0;
        if (!motor.readModelNumber(id, modelNumber)) {
            std::cerr << id << "         (read failed)\n";
            anyFailed = true;
            continue;
        }
        std::cout << id << "         " << modelNumber << '\n';
    }

    if (anyFailed) {
        std::cerr << "\nOne or more motors could not be read -- see above.\n";
    }

    motor.disconnect();
    return anyFailed ? 1 : 0;
}
