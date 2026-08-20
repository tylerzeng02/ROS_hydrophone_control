// test_i_gain: single-joint, one-parameter-at-a-time PID I-Gain test.
// Servos are MX-64/MX-28 (real PID), not AX-12A -- see check_servo_model.cpp.
//
// I Gain eliminates steady-state error under constant load, matching the
// "settles a few ticks short and never closes the gap" symptom. Unlike
// P Gain (already touched by a past vibration incident), I Gain has
// never been touched before.
//
// Usage: test_i_gain [motorId] [testIGain] [startTick] [targetTick]
//   motorId:   default 3 (elbow_pitch), a real backlash/gravity hotspot.
//   testIGain: default 8. Baseline is read from the servo, not assumed.
//   startTick/targetTick: both required together, or both omitted
//               (defaults to midpoint +/- 300). Pass explicit ticks near
//               one end of the range to land somewhere gravity-loaded --
//               a symmetric-midpoint default may show no baseline error
//               for I Gain to fix (and can look like it hurts via
//               ordinary integral windup instead).
//
// Methodology: read/print baseline D/I/P gains. Move to START, settle.
// Move to TARGET at unmodified gains (single setGoalPosition() write, not
// moveJointSafely(), to avoid its own backlash-overshoot logic
// confounding the isolation), poll until settled, record BEFORE error.
// Back to START. Write only I Gain to the test value (D/P unchanged),
// confirm via read-back. Move to TARGET again, record AFTER error. Back
// to START, restore original D/I/P, confirm via read-back regardless of
// outcome. Print both poll traces plus a before/after summary.
//
// WATCH AND LISTEN during AFTER. Any vibration/oscillation means abort
// (Ctrl+C is safe -- torque holds the last commanded position) and this
// I Gain value is too aggressive.
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
    const uint8_t testIGain = argc > 2 ? static_cast<uint8_t>(std::stoi(argv[2])) : 8;

    const JointCalibration* calibration = findCalibration(motorId);
    if (calibration == nullptr) {
        std::cerr << "No jointCalibrations entry for motor " << motorId << ".\n";
        return 1;
    }

    // startTick/targetTick: explicit overrides (args 3 and 4) -- see header
    // comment. Falls back to midpoint +/- 300 if omitted.
    int startTickInt, targetTickInt;
    if (argc > 4) {
        startTickInt = std::stoi(argv[3]);
        targetTickInt = std::stoi(argv[4]);
    } else {
        const int midpoint = (calibration->minTick + calibration->maxTick) / 2;
        constexpr int DEFAULT_OFFSET_TICKS = 300;
        startTickInt = midpoint - DEFAULT_OFFSET_TICKS;
        targetTickInt = midpoint + DEFAULT_OFFSET_TICKS;
    }
    if (startTickInt < calibration->minTick || startTickInt > calibration->maxTick ||
        targetTickInt < calibration->minTick || targetTickInt > calibration->maxTick) {
        std::cerr << "start=" << startTickInt << " / target=" << targetTickInt
                  << " outside motor " << motorId << "'s calibrated range ["
                  << calibration->minTick << ", " << calibration->maxTick << "].\n";
        return 1;
    }
    const uint16_t startTick = static_cast<uint16_t>(startTickInt);
    const uint16_t targetTick = static_cast<uint16_t>(targetTickInt);

    std::cout << "=== I-Gain test: motor " << motorId << " ===\n"
              << "start=" << startTick << "  target=" << targetTick << "  testIGain="
              << static_cast<int>(testIGain) << "\n\n"
              << "WATCH AND LISTEN to this joint for the whole test. Any vibration or "
                 "oscillation means abort (Ctrl+C is safe -- torque holds the last commanded "
                 "position, no ongoing host communication needed) and this I Gain value is too "
                 "aggressive for this joint.\n\n"
              << "Press Enter to begin: ";
    std::cin.get();

    DynamixelMotor motor(DEVICE, BAUD_RATE, PROTOCOL_VERSION);
    if (!motor.connect()) {
        std::cerr << "Could not connect to " << DEVICE << ".\n";
        return 1;
    }

    // Torque on ALL 7 joints, not just the one under test, so the rest of
    // the arm stays rigid instead of hanging loose under gravity.
    std::cout << "Enabling torque on all 7 joints...\n";
    for (int otherMotorId = 0; otherMotorId < 7; ++otherMotorId) {
        if (!motor.pingMotor(otherMotorId)) {
            std::cerr << "Could not ping motor " << otherMotorId << ".\n";
            motor.disconnect();
            return 1;
        }
        if (!motor.enableTorque(otherMotorId)) {
            std::cerr << "Could not enable torque on motor " << otherMotorId << ".\n";
            motor.disconnect();
            return 1;
        }
    }

    uint8_t originalD = 0, originalI = 0, originalP = 0;
    if (!motor.readGains(motorId, originalD, originalI, originalP)) {
        std::cerr << "Could not read baseline gains -- aborting before touching anything.\n";
        motor.disconnect();
        return 1;
    }
    std::cout << "Baseline gains on motor " << motorId << ": D=" << static_cast<int>(originalD)
              << " I=" << static_cast<int>(originalI) << " P=" << static_cast<int>(originalP)
              << " (read directly, not assumed)\n";

    // --- BEFORE: current (unmodified) gains ---
    std::cout << "\n--- Moving to START (settle) ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    // moveJointSafely() unconditionally disables torque once it reaches
    // target (no parameter can prevent this) -- re-enable immediately so
    // the joint doesn't hang unsupported under gravity.
    motor.enableTorque(motorId);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "\n--- BEFORE (I=" << static_cast<int>(originalI)
              << "): moving to TARGET, polling ---\n";
    motor.setGoalPosition(motorId, targetTick);
    const uint16_t beforeFinal = pollUntilSettled(motor, motorId, targetTick);
    const int beforeError = static_cast<int>(beforeFinal) - static_cast<int>(targetTick);

    std::cout << "\n--- Moving back to START (settle) ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    motor.enableTorque(motorId);  // see the earlier comment on this pattern
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // --- Apply test I Gain (D and P left exactly as read, unchanged) ---
    std::cout << "\n--- Writing testIGain=" << static_cast<int>(testIGain) << " to motor " << motorId
              << " (D=" << static_cast<int>(originalD) << ", P=" << static_cast<int>(originalP)
              << " unchanged) ---\n";
    if (!motor.writeGains(motorId, originalD, testIGain, originalP)) {
        std::cerr << "FAILED to write test I gain -- aborting rather than moving with an unknown "
                     "gain state.\n";
        motor.disconnect();
        return 1;
    }
    uint8_t confirmedD = 0, confirmedI = 0, confirmedP = 0;
    motor.readGains(motorId, confirmedD, confirmedI, confirmedP);
    std::cout << "Confirmed gains now read: D=" << static_cast<int>(confirmedD)
              << " I=" << static_cast<int>(confirmedI) << " P=" << static_cast<int>(confirmedP)
              << '\n';
    if (confirmedI != testIGain) {
        std::cerr << "WARNING: I gain readback (" << static_cast<int>(confirmedI)
                  << ") does not match requested (" << static_cast<int>(testIGain) << ").\n";
    }

    // --- AFTER: test I Gain ---
    std::cout << "\n--- AFTER (I=" << static_cast<int>(confirmedI)
              << "): moving to TARGET, polling -- WATCH/LISTEN NOW ---\n";
    motor.setGoalPosition(motorId, targetTick);
    const uint16_t afterFinal = pollUntilSettled(motor, motorId, targetTick);
    const int afterError = static_cast<int>(afterFinal) - static_cast<int>(targetTick);

    // --- Cleanup: back to START, restore ORIGINAL gains (always
    // attempted, regardless of what happened above). ---
    std::cout << "\n--- Moving back to START, restoring original gains ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    motor.enableTorque(motorId);  // see the earlier comment on this pattern
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    if (!motor.writeGains(motorId, originalD, originalI, originalP)) {
        std::cerr << "WARNING: FAILED to restore original gains (D=" << static_cast<int>(originalD)
                  << " I=" << static_cast<int>(originalI) << " P=" << static_cast<int>(originalP)
                  << ") on motor " << motorId
                  << " -- verify manually before trusting this motor's behavior in anything "
                     "else.\n";
    } else {
        uint8_t restoredD = 0, restoredI = 0, restoredP = 0;
        motor.readGains(motorId, restoredD, restoredI, restoredP);
        const bool matches = restoredD == originalD && restoredI == originalI && restoredP == originalP;
        std::cout << "Restored gains: D=" << static_cast<int>(restoredD)
                  << " I=" << static_cast<int>(restoredI) << " P=" << static_cast<int>(restoredP)
                  << (matches ? " (confirmed matches original)\n" : " (WARNING: does NOT match original)\n");
    }

    std::cout << "\n=== Summary (motor " << motorId << ", target=" << targetTick << ") ===\n"
              << "BEFORE (I=" << static_cast<int>(originalI) << "): final tick=" << beforeFinal
              << "  error=" << beforeError << " ticks\n"
              << "AFTER  (I=" << static_cast<int>(testIGain) << "): final tick=" << afterFinal
              << "  error=" << afterError << " ticks\n";
    if (std::abs(afterError) < std::abs(beforeError)) {
        std::cout << "-> AFTER settled closer to target. Worth repeating a few times, then "
                     "validating with a real NDI before/after comparison before trusting this as "
                     "a fix.\n";
    } else if (std::abs(afterError) > std::abs(beforeError)) {
        std::cout << "-> AFTER settled FARTHER from target than BEFORE (or oscillated -- check the "
                     "poll trace above). Do not adopt this I Gain value for this joint.\n";
    } else {
        std::cout << "-> No difference on this single trial -- repeat a few times before "
                     "concluding anything (settling behavior can vary run to run).\n";
    }

    motor.disconnect();
    return 0;
}
