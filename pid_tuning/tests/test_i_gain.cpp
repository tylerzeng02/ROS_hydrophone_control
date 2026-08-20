// test_i_gain: single-joint, one-parameter-at-a-time PID I-Gain test --
// the direct, correctly-targeted successor to test_compliance_punch.cpp
// now that this arm's servos are confirmed MX-64/MX-28 (real PID control),
// not AX-12A (compliance control) -- see tests/check_servo_model.cpp and
// dynamixel_motor.h's corrected header comment on readComplianceMargins().
//
// Why I Gain specifically: it's the PID term that eliminates STEADY-STATE
// error under constant load -- exactly the "servo settles a few ticks
// short of target and never closes the gap" symptom documented in
// CLAUDE.md's kinematic-calibration section. Unlike P Gain (address 28),
// which the 2026-07-29 vibration incident already touched (aggressively,
// across presumably all joints, and had to be reverted), I Gain (address
// 27) has never been touched by anything in this project's history --
// it's a genuinely untested, well-targeted candidate, not a repeat of a
// past mistake.
//
// Usage: test_i_gain [motorId] [testIGain] [startTick] [targetTick]
//   motorId:     which joint to test (default 3 = elbow_pitch -- a real
//                backlash/gravity hotspot with meaningful lever arm to
//                the end-effector, so an improvement here is more likely
//                to be visible in real NDI-measured accuracy).
//   testIGain:   the I Gain value to try (default 8; factory/current
//                value is read directly from the servo and printed, NOT
//                assumed -- I Gain is commonly 0 by default on Dynamixel
//                servos, meaning zero integral action, which is
//                consistent with the steady-state-error symptom this
//                test is chasing). Valid range is 0-254 (1 byte); this
//                program does not clamp/validate beyond what
//                writeGains()/the servo firmware itself enforces.
//   startTick/targetTick: explicit move endpoints (both required together,
//                or both omitted). If omitted, defaults to the joint's
//                calibrated midpoint +/- 300 ticks. IMPORTANT (2026-08-13,
//                found via a real first trial): a symmetric-around-
//                midpoint default has no guarantee of landing anywhere
//                near a gravity-loaded configuration -- a P-only
//                controller's steady-state error scales with however much
//                constant torque is fighting it, so testing at a spot
//                with little load will show little-to-no baseline error
//                for I Gain to fix (and can make I Gain look like it hurts
//                via ordinary integral windup, when really there was
//                nothing to correct). Pass explicit ticks nearer one end
//                of the joint's range (more extended/horizontal
//                configuration, more downstream mass, larger gravity
//                moment arm about this joint's own axis) to actually
//                exercise the failure mode this test is meant to probe.
//
// Methodology: read and print the CURRENT D/I/P gains (baseline -- NOT
// assumed factory values). Move to START, settle. Move to TARGET at the
// CURRENT (unmodified) gains via a single setGoalPosition() write (NOT
// moveJointSafely(), to avoid its own backlash-overshoot logic
// confounding what we're isolating), poll position every 100ms until it
// stops changing or times out, record the BEFORE tick error. Move back to
// START, settle. Write ONLY I Gain to the test value -- D Gain and P Gain
// are read back and written back UNCHANGED, not reset to any assumed
// default, so this test isolates exactly one parameter the same way
// test_compliance_punch.cpp isolated Punch. Confirm via read-back. Move
// to TARGET again the same way, poll the same way, record the AFTER tick
// error. Move back to START, restore the ORIGINAL D/I/P gains read at the
// very start, confirm via read-back regardless of what happened above.
// Print both poll traces in full (watch for oscillation in the numbers --
// a real vibration signature, not just a final-error number) plus a
// before/after summary.
//
// IMPORTANT: watch and LISTEN to the arm during the AFTER phase. Any
// audible vibration or visible oscillation means abort (Ctrl+C is safe --
// see CLAUDE.md's "Torque/Ctrl+C note": the servo just holds its last
// commanded position with no ongoing host communication needed) and this
// I Gain value is too aggressive for this joint -- try a smaller step.
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

    // startTick/targetTick: explicit overrides (args 3 and 4), letting the
    // caller deliberately target a specific, potentially gravity-loaded
    // configuration instead of an arbitrary midpoint-centered offset.
    // Steady-state error under a P-only controller scales with however
    // much constant torque (e.g. gravity) is fighting it -- a
    // symmetric-around-midpoint test has no guarantee of landing anywhere
    // near where that load, and hence the error I Gain is meant to fix,
    // is actually significant. If omitted, falls back to the original
    // midpoint +/- 300 default.
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

    // Enable torque on ALL 7 joints, not just the one under test -- so the
    // rest of the arm stays rigid/holding position while the test joint
    // moves, instead of hanging loose under gravity. This also matches
    // real production dynamics more closely (the whole arm's weight is
    // normally supported by every joint at once), which matters here
    // since gravity-dependent load is part of what this test is
    // implicitly probing.
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
    // moveJointSafely() unconditionally disables torque on this motor once
    // it reaches its target (no parameter can prevent this, unlike the
    // multi-joint moveJointsSafely()'s holdTorque) -- re-enable immediately
    // so the joint doesn't hang unsupported under gravity for however long
    // this program does anything next. See this file's own comment history
    // (2026-08-13) for the real, observed drop this caused before the fix.
    motor.enableTorque(motorId);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "\n--- BEFORE (I=" << static_cast<int>(originalI)
              << "): moving to TARGET, polling ---\n";
    motor.setGoalPosition(motorId, targetTick);
    const uint16_t beforeFinal = pollUntilSettled(motor, motorId, targetTick);
    const int beforeError = static_cast<int>(beforeFinal) - static_cast<int>(targetTick);

    std::cout << "\n--- Moving back to START (settle) ---\n";
    motor.moveJointSafely(motorId, startTick, MOVING_SPEED);
    // moveJointSafely() unconditionally disables torque on this motor once
    // it reaches its target (no parameter can prevent this, unlike the
    // multi-joint moveJointsSafely()'s holdTorque) -- re-enable immediately
    // so the joint doesn't hang unsupported under gravity for however long
    // this program does anything next. See this file's own comment history
    // (2026-08-13) for the real, observed drop this caused before the fix.
    motor.enableTorque(motorId);
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
    // moveJointSafely() unconditionally disables torque on this motor once
    // it reaches its target (no parameter can prevent this, unlike the
    // multi-joint moveJointsSafely()'s holdTorque) -- re-enable immediately
    // so the joint doesn't hang unsupported under gravity for however long
    // this program does anything next. See this file's own comment history
    // (2026-08-13) for the real, observed drop this caused before the fix.
    motor.enableTorque(motorId);
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
