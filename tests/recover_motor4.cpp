// One-off recovery utility (2026-08-07): drives motor 4 (elbow_yaw) back to
// its locked validation tick (2095) when it's physically outside the safe
// range jointCalibrations enforces -- moveJointSafely()'s pre-move check
// requires the CURRENT position to already be in-range, not just the
// target, so it refuses to fix an out-of-range starting position itself.
// Uses setGoalPosition() directly (still safety-checks the TARGET position,
// just not the current one), same pattern as record_hand_poses.cpp's
// auto-drive-to-tick polling loop. Not meant to be a general-purpose tool --
// delete once the immediate recovery is done.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "dynamixel_motor.h"

namespace {
constexpr const char* CYTON_DEVICE = "/dev/ttyUSB0";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;

constexpr int RECOVER_MOTOR_ID = 4;
constexpr uint16_t RECOVER_TARGET_TICK = 2095;
constexpr uint16_t RECOVER_MOVING_SPEED = 20;  // slow, deliberate
constexpr int RECOVER_TOLERANCE_TICKS = 4;
constexpr int RECOVER_TIMEOUT_MS = 15000;
constexpr int RECOVER_POLL_MS = 100;
}  // namespace

int main() {
    DynamixelMotor motor(CYTON_DEVICE, CYTON_BAUD_RATE, CYTON_PROTOCOL_VERSION);

    if (!motor.connect()) {
        std::cerr << "Could not connect to the Cyton motors.\n";
        return 1;
    }

    if (!motor.pingMotor(RECOVER_MOTOR_ID)) {
        std::cerr << "Could not ping motor " << RECOVER_MOTOR_ID << ".\n";
        motor.disconnect();
        return 1;
    }

    uint16_t startPos = 0;
    if (!motor.readPosition(RECOVER_MOTOR_ID, startPos)) {
        std::cerr << "Could not read starting position.\n";
        motor.disconnect();
        return 1;
    }
    std::cout << "Motor " << RECOVER_MOTOR_ID << " starting position: "
              << startPos << ". Driving to " << RECOVER_TARGET_TICK << "...\n";

    if (!motor.setMovingSpeed(RECOVER_MOTOR_ID, RECOVER_MOVING_SPEED)) {
        std::cerr << "Could not set moving speed.\n";
        motor.disconnect();
        return 1;
    }
    if (!motor.enableTorque(RECOVER_MOTOR_ID)) {
        std::cerr << "Could not enable torque.\n";
        motor.disconnect();
        return 1;
    }
    if (!motor.setGoalPosition(RECOVER_MOTOR_ID, RECOVER_TARGET_TICK)) {
        std::cerr << "setGoalPosition rejected the target (should be "
                     "impossible -- 2095 is inside the safe range).\n";
        motor.disconnect();
        return 1;
    }

    bool settled = false;
    for (int elapsed = 0; elapsed <= RECOVER_TIMEOUT_MS; elapsed += RECOVER_POLL_MS) {
        uint16_t currentPos = 0;
        if (!motor.readPosition(RECOVER_MOTOR_ID, currentPos)) {
            std::cerr << "Failed to read position while monitoring the move.\n";
            break;
        }
        std::cout << "  current: " << currentPos << " (target "
                  << RECOVER_TARGET_TICK << ")\n";
        if (std::abs(static_cast<int>(currentPos) - static_cast<int>(RECOVER_TARGET_TICK))
            <= RECOVER_TOLERANCE_TICKS) {
            settled = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(RECOVER_POLL_MS));
    }

    uint16_t finalPos = 0;
    motor.readPosition(RECOVER_MOTOR_ID, finalPos);
    std::cout << (settled ? "Settled. " : "Timed out, but reporting final position. ")
              << "Motor " << RECOVER_MOTOR_ID << " final position: " << finalPos << "\n";

    motor.disconnect();
    return settled ? 0 : 1;
}
