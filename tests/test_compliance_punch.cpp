// test_compliance_punch: single-joint, one-parameter-at-a-time compliance
// test -- see CLAUDE.md's kinematic-calibration section ("Compliance-
// margin/slope/punch investigated...") for the diagnosis this tests, and
// the caution motivating this program's design. An earlier attempt
// (2026-07-29) changed slope AND punch together, across presumably all
// joints at once, and caused audible vibration -- it had to be reverted
// (tests/reset_compliance.cpp exists because of that incident). This test
// changes ONLY Punch, on ONLY one motor, and measures the actual
// before/after tick-settling error at a real target -- not just "did it
// run without erroring."
//
// Usage: test_compliance_punch [motorId] [testPunch] [offsetTicks]
//   motorId:     which joint to test (default 3 = elbow_pitch -- a real
//                backlash/gravity hotspot with meaningful lever arm to the
//                end-effector, so an improvement here is more likely to be
//                visible in real NDI-measured accuracy -- see CLAUDE.md).
//   testPunch:   the punch value to try (default 60; factory default is
//                32 -- see dynamixel_motor.h's own comment for what this
//                register does. Valid range per the AX-12A spec is
//                32-1023; this program does not clamp/validate beyond
//                what writePunch()/the servo firmware itself enforces).
//   offsetTicks: how far from the joint's calibrated midpoint to place
//                START/TARGET (default 300 -- comfortably clears every
//                joint's measured backlash gap from CLAUDE.md's per-joint
//                backlash survey, so this isn't confounded by an
//                incomplete reversal).
//
// Methodology: move to START, settle. Move to TARGET at FACTORY punch
// (32) via a single setGoalPosition() write (NOT moveJointSafely(), to
// avoid its own backlash-overshoot logic confounding what we're
// isolating), poll position every 100ms until it stops changing or times
// out, record the BEFORE tick error. Move back to START, settle. Write
// the TEST punch value to the one motor under test, confirm via
// read-back. Move to TARGET again the same way, poll the same way, record
// the AFTER tick error. Move back to START, restore FACTORY punch,
// confirm via read-back regardless of what happened above. Print both
// poll traces in full (watch for oscillation in the numbers -- a real
// vibration signature, not just a final-error number) plus a before/after
// summary.
//
// IMPORTANT: watch and LISTEN to the arm during the AFTER phase. Any
// audible vibration or visible oscillation means abort (Ctrl+C is safe --
// see CLAUDE.md's "Torque/Ctrl+C note": the servo just holds its last
// commanded position with no ongoing host communication needed) and this
// punch value is too aggressive for this joint -- try a smaller testPunch
// step instead of a bigger one.
//
// This is a SINGLE-TRIAL diagnostic, not a validated fix. A real fix
// needs this repeated a few times (settling behavior can vary run to
// run), then confirmed with an actual NDI before/after comparison (the
// same standard this project applies to every other change), before being
// trusted or rolled out to other joints.

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
    // moveJointSafely() unconditionally disables torque on this motor once
    // it reaches its target (no parameter can prevent this, unlike the
    // multi-joint moveJointsSafely()'s holdTorque) -- re-enable immediately
    // so the joint doesn't hang unsupported under gravity for however long
    // this program does anything next.
    motor.enableTorque(motorId);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "\n--- BEFORE (factory punch=" << basePunch << "): moving to TARGET, polling ---\n";
    motor.setGoalPosition(motorId, targetTick);
    const uint16_t beforeFinal = pollUntilSettled(motor, motorId, targetTick);
    const int beforeError = static_cast<int>(beforeFinal) - static_cast<int>(targetTick);

    std::cout << "\n--- Moving back to START (settle) ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    // moveJointSafely() unconditionally disables torque on this motor once
    // it reaches its target (no parameter can prevent this, unlike the
    // multi-joint moveJointsSafely()'s holdTorque) -- re-enable immediately
    // so the joint doesn't hang unsupported under gravity for however long
    // this program does anything next.
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
    // moveJointSafely() unconditionally disables torque on this motor once
    // it reaches its target (no parameter can prevent this, unlike the
    // multi-joint moveJointsSafely()'s holdTorque) -- re-enable immediately
    // so the joint doesn't hang unsupported under gravity for however long
    // this program does anything next.
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
