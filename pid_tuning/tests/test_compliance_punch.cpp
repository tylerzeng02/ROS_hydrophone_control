// test_compliance_punch: single-joint, one-parameter-at-a-time compliance
// test. An earlier attempt changed slope AND punch together across all
// joints and caused audible vibration (had to be reverted --
// reset_compliance.cpp exists because of that). This test changes only
// Punch, on only one motor, and measures actual before/after
// tick-settling error, not just "did it run."
//
// Usage: test_compliance_punch [motorId] [testPunch] [offsetTicks]
//   motorId:     default 3 (elbow_pitch), a real backlash/gravity hotspot.
//   testPunch:   default 60 (factory default is 32; range 32-1023 per the
//                AX-12A spec, not clamped beyond what the firmware itself
//                enforces).
//   offsetTicks: default 300 -- comfortably clears every joint's measured
//                backlash gap so this isn't confounded by an incomplete
//                reversal.
//
// Methodology: move to START, settle. Move to TARGET at factory punch
// (single setGoalPosition() write, not moveJointSafely(), to avoid its
// backlash-overshoot logic confounding the isolation), poll until
// settled, record BEFORE error. Back to START. Write test punch, confirm
// via read-back. Move to TARGET again, record AFTER error. Back to
// START, restore factory punch, confirm via read-back regardless of
// outcome. Print both poll traces plus a before/after summary.
//
// WATCH AND LISTEN during AFTER. Any vibration/oscillation means abort
// (Ctrl+C is safe -- torque holds the last commanded position) and this
// punch value is too aggressive.
//
// Single-trial diagnostic, not a validated fix -- repeat a few times, then
// confirm with a real NDI before/after comparison before trusting it.

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

namespace {

constexpr const char* DEVICE = "/dev/ttyUSB0";
constexpr int BAUD_RATE = 1000000;
constexpr float PROTOCOL_VERSION = 1.0F;
constexpr uint16_t MOVING_SPEED = 40;  // matches this project's established convention

constexpr uint16_t FACTORY_PUNCH = 32;

constexpr int POLL_INTERVAL_MS = 100;
constexpr int SETTLE_TIMEOUT_MS = 5000;
// Position considered "stopped changing" once consecutive polls land
// within this many ticks of each other -- catches genuine settling
// (including a stall short of target) without always waiting the full
// timeout.
constexpr int STABLE_TICKS = 1;
constexpr int STABLE_POLLS_REQUIRED = 5;  // consecutive stable polls before declaring settled

const JointCalibration* findCalibration(int motorId) {
    for (const auto& cal : jointCalibrations) {
        if (cal.id == motorId) {
            return &cal;
        }
    }
    return nullptr;
}

// Polls position every POLL_INTERVAL_MS, printing each sample and its
// error against `target`, until the position stabilizes
// (STABLE_POLLS_REQUIRED consecutive polls within STABLE_TICKS of each
// other) or SETTLE_TIMEOUT_MS elapses. Returns the final observed
// position.
uint16_t pollUntilSettled(DynamixelMotor& motor, int motorId, uint16_t target) {
    int elapsedMs = 0;
    int stableCount = 0;
    uint16_t lastPosition = 0;
    bool havePrevious = false;

    while (elapsedMs < SETTLE_TIMEOUT_MS) {
        uint16_t position = 0;
        if (!motor.readPosition(motorId, position)) {
            std::cout << "    (read failed at t=" << elapsedMs << "ms)\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
            elapsedMs += POLL_INTERVAL_MS;
            continue;
        }

        const int errorTicks = static_cast<int>(position) - static_cast<int>(target);
        std::cout << "    t=" << elapsedMs << "ms  tick=" << position
                  << "  error_vs_target=" << errorTicks << '\n';

        if (havePrevious &&
            std::abs(static_cast<int>(position) - static_cast<int>(lastPosition)) <= STABLE_TICKS) {
            ++stableCount;
            if (stableCount >= STABLE_POLLS_REQUIRED) {
                std::cout << "    -> settled.\n";
                return position;
            }
        } else {
            stableCount = 0;
        }

        lastPosition = position;
        havePrevious = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        elapsedMs += POLL_INTERVAL_MS;
    }

    std::cout << "    -> timed out without settling (still moving/oscillating at t="
              << SETTLE_TIMEOUT_MS << "ms) -- treat this run's result with suspicion.\n";
    return lastPosition;
}

}  // namespace

int main(int argc, char** argv) {
    const int motorId = argc > 1 ? std::stoi(argv[1]) : 3;
    const uint16_t testPunch = argc > 2 ? static_cast<uint16_t>(std::stoi(argv[2])) : 60;
    const int offsetTicks = argc > 3 ? std::stoi(argv[3]) : 300;

    const JointCalibration* calibration = findCalibration(motorId);
    if (calibration == nullptr) {
        std::cerr << "No jointCalibrations entry for motor " << motorId << ".\n";
        return 1;
    }

    const int midpoint = (calibration->minTick + calibration->maxTick) / 2;
    const int startTickInt = midpoint - offsetTicks;
    const int targetTickInt = midpoint + offsetTicks;
    if (startTickInt < calibration->minTick || targetTickInt > calibration->maxTick) {
        std::cerr << "offsetTicks=" << offsetTicks << " puts start/target outside motor " << motorId
                  << "'s calibrated range [" << calibration->minTick << ", " << calibration->maxTick
                  << "] (midpoint " << midpoint << ") -- pass a smaller offset.\n";
        return 1;
    }
    const uint16_t startTick = static_cast<uint16_t>(startTickInt);
    const uint16_t targetTick = static_cast<uint16_t>(targetTickInt);

    std::cout << "=== Compliance/Punch test: motor " << motorId << " ===\n"
              << "start=" << startTick << "  target=" << targetTick
              << "  testPunch=" << testPunch << " (factory=" << FACTORY_PUNCH << ")\n\n"
              << "WATCH AND LISTEN to this joint for the whole test. Any vibration or "
                 "oscillation means abort (Ctrl+C is safe -- torque holds the last commanded "
                 "position, no ongoing host communication needed) and this punch value is too "
                 "aggressive for this joint.\n\n"
              << "Press Enter to begin: ";
    std::cin.get();

    DynamixelMotor motor(DEVICE, BAUD_RATE, PROTOCOL_VERSION);
    if (!motor.connect()) {
        std::cerr << "Could not connect to " << DEVICE << ".\n";
        return 1;
    }
    if (!motor.pingMotor(motorId)) {
        std::cerr << "Could not ping motor " << motorId << ".\n";
        motor.disconnect();
        return 1;
    }
    if (!motor.enableTorque(motorId)) {
        std::cerr << "Could not enable torque on motor " << motorId << ".\n";
        motor.disconnect();
        return 1;
    }

    uint16_t basePunch = 0;
    if (!motor.readPunch(motorId, basePunch)) {
        std::cerr << "Could not read baseline punch.\n";
    } else {
        std::cout << "Baseline punch on motor " << motorId << ": " << basePunch
                  << (basePunch == FACTORY_PUNCH
                          ? " (factory default)\n"
                          : " (NOT factory default -- was this motor already modified? consider "
                            "running reset_compliance first)\n");
    }

    // --- BEFORE: factory punch ---
    std::cout << "\n--- Moving to START (settle) ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    // moveJointSafely() unconditionally disables torque once it reaches
    // target -- re-enable immediately so the joint doesn't hang
    // unsupported under gravity.
    motor.enableTorque(motorId);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "\n--- BEFORE (factory punch=" << basePunch << "): moving to TARGET, polling ---\n";
    motor.setGoalPosition(motorId, targetTick);
    const uint16_t beforeFinal = pollUntilSettled(motor, motorId, targetTick);
    const int beforeError = static_cast<int>(beforeFinal) - static_cast<int>(targetTick);

    std::cout << "\n--- Moving back to START (settle) ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    // moveJointSafely() unconditionally disables torque once it reaches
    // target -- re-enable immediately so the joint doesn't hang
    // unsupported under gravity.
    motor.enableTorque(motorId);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // --- Apply test punch ---
    std::cout << "\n--- Writing testPunch=" << testPunch << " to motor " << motorId << " ---\n";
    if (!motor.writePunch(motorId, testPunch)) {
        std::cerr << "FAILED to write test punch -- aborting rather than moving with an unknown "
                     "compliance state.\n";
        motor.disconnect();
        return 1;
    }
    uint16_t confirmedPunch = 0;
    motor.readPunch(motorId, confirmedPunch);
    std::cout << "Confirmed punch now reads: " << confirmedPunch << '\n';
    if (confirmedPunch != testPunch) {
        std::cerr << "WARNING: readback (" << confirmedPunch << ") does not match requested ("
                  << testPunch << ").\n";
    }

    // --- AFTER: test punch ---
    std::cout << "\n--- AFTER (testPunch=" << confirmedPunch
              << "): moving to TARGET, polling -- WATCH/LISTEN NOW ---\n";
    motor.setGoalPosition(motorId, targetTick);
    const uint16_t afterFinal = pollUntilSettled(motor, motorId, targetTick);
    const int afterError = static_cast<int>(afterFinal) - static_cast<int>(targetTick);

    // --- Cleanup: back to START, restore factory punch (always attempted,
    // regardless of what happened above). ---
    std::cout << "\n--- Moving back to START, restoring factory punch ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    // moveJointSafely() unconditionally disables torque once it reaches
    // target -- re-enable immediately so the joint doesn't hang
    // unsupported under gravity.
    motor.enableTorque(motorId);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    if (!motor.writePunch(motorId, FACTORY_PUNCH)) {
        std::cerr << "WARNING: FAILED to restore factory punch (" << FACTORY_PUNCH << ") on motor "
                  << motorId << " -- run tests/reset_compliance before trusting this motor's "
                     "behavior in anything else.\n";
    } else {
        uint16_t restoredPunch = 0;
        motor.readPunch(motorId, restoredPunch);
        std::cout << "Restored punch: " << restoredPunch
                  << (restoredPunch == FACTORY_PUNCH ? " (confirmed factory default)\n"
                                                      : " (WARNING: did not confirm -- run "
                                                        "tests/reset_compliance)\n");
    }

    std::cout << "\n=== Summary (motor " << motorId << ", target=" << targetTick << ") ===\n"
              << "BEFORE (punch=" << basePunch << "): final tick=" << beforeFinal
              << "  error=" << beforeError << " ticks\n"
              << "AFTER  (punch=" << testPunch << "): final tick=" << afterFinal
              << "  error=" << afterError << " ticks\n";
    if (std::abs(afterError) < std::abs(beforeError)) {
        std::cout << "-> AFTER settled closer to target. Worth repeating a few times, then "
                     "validating with a real NDI before/after comparison before trusting this as "
                     "a fix.\n";
    } else if (std::abs(afterError) > std::abs(beforeError)) {
        std::cout << "-> AFTER settled FARTHER from target than BEFORE. Do not adopt this punch "
                     "value for this joint.\n";
    } else {
        std::cout << "-> No difference on this single trial -- repeat a few times before "
                     "concluding anything (settling behavior can vary run to run).\n";
    }

    motor.disconnect();
    return 0;
}
