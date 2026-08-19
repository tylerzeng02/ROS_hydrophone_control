#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

namespace {

constexpr const char* CYTON_DEVICE = "COM4";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;

constexpr std::size_t JOINT_COUNT = 7;

// Reduced-DOF data collection (2026-08-04): elbow_yaw (motor 4) is
// commanded to its calibrated range's midpoint and held there (torqued)
// for the whole session -- see CLAUDE.md's kinematic-calibration section
// for why: it's the single largest measured backlash contributor (7.68mm)
// of any joint, and a confirmed joint-coupling/gravity-deflection hotspot.
// Fixing it removes both its own motion error and the IK redundancy it
// otherwise contributes, at the cost of one fewer usable DOF. The other 6
// joints are freely hand-posed (torque off) exactly like the original
// recorded_hand_poses (1)/(2)/(3).csv batches, so this data drops straight
// into the same pipeline used all session (excluded-range extraction,
// TARGET_POSES generation, etc.) -- just with motor 4 constant across
// every row instead of varying.
constexpr int LOCKED_JOINT_ID = 4;
constexpr uint16_t LOCKED_JOINT_TICK = 2095;  // (944 + 3245) / 2, current calibrated range

constexpr uint16_t LOCK_MOVE_SPEED = 40;
constexpr int LOCK_MOVE_TOLERANCE_TICKS = 5;
constexpr int LOCK_MOVE_TIMEOUT_SECONDS = 10;

// Full raw tick range for these servos (4096 ticks/revolution). Widening
// each motor's hardware CW/CCW angle-limit registers to this for the
// duration of this program only lets the servo ACCEPT holding wherever a
// human already physically placed it by hand -- it never causes the arm to
// move anywhere it hasn't already been placed, so it introduces no new
// mechanical risk specific to this tool. (The one exception is the locked
// joint's own commanded move to its midpoint, which goes through the
// normal safety-checked moveJointSafely() before this widening matters.)
constexpr uint16_t FULL_RANGE_CW_LIMIT = 0;
constexpr uint16_t FULL_RANGE_CCW_LIMIT = 4095;

constexpr const char* OUTPUT_CSV = "pid_tuning_data.csv";

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

// Returns true and fills `line` with the raw input if the user just
// pressed Enter (record a pose); returns false if the user typed 'q' (done).
bool promptForNextAction() {
    std::cout
        << "\nPress Enter to record a pose, or type 'q' then Enter to "
        << "finish: " << std::flush;
    std::string line;
    std::getline(std::cin, line);
    if (!line.empty() && (line[0] == 'q' || line[0] == 'Q')) {
        return false;
    }
    return true;
}

std::size_t loadExistingPoseCount(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return 0;
    }
    std::string line;
    std::getline(in, line);  // header, discarded
    std::size_t count = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            ++count;
        }
    }
    return count;
}

}  // namespace

int main() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    const std::size_t existingPoseCount = loadExistingPoseCount(OUTPUT_CSV);
    const bool resuming = existingPoseCount > 0;
    std::size_t poseCount = existingPoseCount;

    std::vector<std::pair<uint16_t, uint16_t>> originalAngleLimits(
        motorIds.size()
    );
    bool angleLimitsWidened = false;

    auto restoreAngleLimits = [&]() {
        if (!angleLimitsWidened) {
            return;
        }
        std::cout << "\nRestoring original hardware angle limits...\n";
        for (std::size_t idx = 0; idx < motorIds.size(); ++idx) {
            const auto& limits = originalAngleLimits[idx];
            if (!motor.writeAngleLimits(
                    motorIds[idx], limits.first, limits.second
                )) {
                std::cerr
                    << "WARNING: failed to restore original angle limits "
                    << "for motor " << motorIds[idx] << ". They were ["
                    << limits.first << ", " << limits.second << "].\n";
            }
        }
    };

    try {
        if (!motor.connect()) {
            throw std::runtime_error(
                "Could not connect to the Cyton motors."
            );
        }

        for (int id : motorIds) {
            if (!motor.pingMotor(id)) {
                throw std::runtime_error(
                    "Could not ping motor " + std::to_string(id)
                );
            }
        }

        for (std::size_t idx = 0; idx < motorIds.size(); ++idx) {
            const int id = motorIds[idx];
            uint16_t cwLimit = 0;
            uint16_t ccwLimit = 0;
            if (!motor.readAngleLimits(id, cwLimit, ccwLimit)) {
                throw std::runtime_error(
                    "Could not read angle limits for motor " +
                    std::to_string(id)
                );
            }
            originalAngleLimits[idx] = {cwLimit, ccwLimit};
        }

        std::cout
            << "\nMoving motor " << LOCKED_JOINT_ID << " (elbow_yaw) to "
            << "its calibrated midpoint (" << LOCKED_JOINT_TICK
            << ") before anything else...\n";

        if (!motor.moveJointSafely(
                LOCKED_JOINT_ID,
                LOCKED_JOINT_TICK,
                LOCK_MOVE_SPEED,
                LOCK_MOVE_TOLERANCE_TICKS,
                LOCK_MOVE_TIMEOUT_SECONDS,
                true
            )) {
            throw std::runtime_error(
                "Motor " + std::to_string(LOCKED_JOINT_ID) +
                " did not reach its midpoint target."
            );
        }
        if (!motor.enableTorque(LOCKED_JOINT_ID)) {
            throw std::runtime_error(
                "Failed to confirm torque is enabled on motor " +
                std::to_string(LOCKED_JOINT_ID) + " to lock it."
            );
        }
        std::cout
            << "Motor " << LOCKED_JOINT_ID << " locked at "
            << LOCKED_JOINT_TICK << ".\n";

        std::cout
            << "\nWidening every OTHER motor's hardware angle limits to "
            << "the full [" << FULL_RANGE_CW_LIMIT << ", "
            << FULL_RANGE_CCW_LIMIT << "] range for this session, so any "
            << "hand-posed position gets accepted. Original limits will "
            << "be restored automatically on normal exit or error -- but "
            << "NOT if this program is killed via Ctrl+C, so avoid that "
            << "this run, or restore them manually afterward if you do.\n";
        for (int id : motorIds) {
            if (!motor.writeAngleLimits(
                    id, FULL_RANGE_CW_LIMIT, FULL_RANGE_CCW_LIMIT
                )) {
                throw std::runtime_error(
                    "Could not widen angle limits for motor " +
                    std::to_string(id)
                );
            }
        }
        angleLimitsWidened = true;

        for (int id : motorIds) {
            if (id == LOCKED_JOINT_ID) {
                continue;
            }
            motor.disableTorque(id);
        }

        std::ofstream csv(
            OUTPUT_CSV,
            std::ios::out | (resuming ? std::ios::app : std::ios::trunc)
        );
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + OUTPUT_CSV
            );
        }
        if (!resuming) {
            csv << "pose_id";
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                csv << ",tick_" << i;
            }
            csv << '\n';
        } else {
            std::cout
                << "Resuming: " << existingPoseCount << " pose(s) already "
                << "recorded in " << OUTPUT_CSV << ".\n";
        }

        std::cout
            << "\nHand-pose recorder -- elbow_yaw fixed at midpoint\n"
            << "Motor " << LOCKED_JOINT_ID << " stays locked at "
            << LOCKED_JOINT_TICK << " for the whole session. The other 6 "
            << "joints are free -- hand-pose them into a configuration, "
            << "then record it.\n"
            << "Poses are written to " << OUTPUT_CSV << " immediately "
            << "after each one (not batched at the end), so Ctrl+C at any "
            << "point loses at most the pose in progress.\n";

        while (promptForNextAction()) {
            std::array<uint16_t, JOINT_COUNT> ticks{};
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                uint16_t position = 0;
                if (!motor.readPosition(motorIds[i], position)) {
                    throw std::runtime_error(
                        "Failed to read position for motor " +
                        std::to_string(motorIds[i])
                    );
                }
                ticks[i] = position;
            }

            ++poseCount;
            std::cout
                << "Recorded pose " << poseCount << ": ";
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                const JointCalibration& joint = jointCalibrations[i];
                if (ticks[i] < joint.minTick || ticks[i] > joint.maxTick) {
                    std::cout
                        << "\n  WARNING: motor " << i << " reading "
                        << ticks[i] << " is outside its calibrated range ["
                        << joint.minTick << ", " << joint.maxTick << "]";
                }
                std::cout << ticks[i] << ' ';
            }
            std::cout << '\n';

            csv << poseCount;
            for (uint16_t tick : ticks) {
                csv << ',' << tick;
            }
            csv << '\n';
            csv.flush();
        }

        std::cout
            << "\nDone -- " << poseCount << " total pose(s) in "
            << OUTPUT_CSV << ".\nReleasing all motors now.\n";

        for (int id : motorIds) {
            motor.disableTorque(id);
        }
        restoreAngleLimits();
        motor.disconnect();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << '\n';
        std::cerr
            << poseCount << " pose(s) were saved to " << OUTPUT_CSV
            << " before this error. Re-run this program to resume.\n";

        for (int id : motorIds) {
            motor.disableTorque(id);
        }
        restoreAngleLimits();
        motor.disconnect();
        return 1;
    }
}
