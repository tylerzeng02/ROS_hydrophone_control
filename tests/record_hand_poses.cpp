#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dynamixel_motor.h"
#include "robot_calibration.h"

namespace {

constexpr const char* CYTON_DEVICE = "COM4";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;

constexpr std::size_t JOINT_COUNT = 7;
constexpr int POSE_COUNT = 200;

// Full raw tick range for these servos (4096 ticks/revolution). Widening
// each motor's hardware CW/CCW angle-limit registers to this for the
// duration of this program only lets the servo ACCEPT holding wherever a
// human already physically placed it by hand -- it never causes the arm to
// move anywhere it hasn't already been placed, so it introduces no new
// mechanical risk specific to this tool.
constexpr uint16_t FULL_RANGE_CW_LIMIT = 0;
constexpr uint16_t FULL_RANGE_CCW_LIMIT = 4095;

constexpr const char* OUTPUT_CSV = "recorded_hand_poses.csv";
constexpr const char* OUTPUT_CPP_SNIPPET = "recorded_hand_poses_target_poses.txt";

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

// Loads any poses already recorded by a previous run of this program, so a
// new session continues the same batch instead of starting over. Silently
// does nothing if the file doesn't exist yet (first run).
void loadExistingPoses(
    const std::string& path,
    std::vector<std::array<uint16_t, JOINT_COUNT>>& poses
) {
    std::ifstream in(path);
    if (!in) {
        return;
    }

    std::string line;
    std::getline(in, line);  // header row, discarded

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream fields(line);
        std::string field;
        std::getline(fields, field, ',');  // pose_id, discarded (renumbered on write)

        std::array<uint16_t, JOINT_COUNT> pose{};
        bool valid = true;
        for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
            if (!std::getline(fields, field, ',')) {
                valid = false;
                break;
            }
            pose[i] = static_cast<uint16_t>(std::stoi(field));
        }

        if (valid) {
            poses.push_back(pose);
        }
    }
}

}  // namespace

int main() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    std::vector<std::array<uint16_t, JOINT_COUNT>> recordedPoses;
    loadExistingPoses(OUTPUT_CSV, recordedPoses);
    const std::size_t preloadedCount = recordedPoses.size();
    if (preloadedCount > 0) {
        std::cout
            << "Loaded " << preloadedCount << " pose(s) already recorded "
            << "in " << OUTPUT_CSV << " -- continuing that batch.\n";
    }

    std::vector<std::pair<uint16_t, uint16_t>> originalAngleLimits(
        motorIds.size()
    );
    bool angleLimitsWidened = false;

    auto writeCsv = [&]() {
        std::ofstream csvOut(OUTPUT_CSV, std::ios::out | std::ios::trunc);
        if (!csvOut) {
            return;
        }
        csvOut << "pose_id";
        for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
            csvOut << ",tick_" << i;
        }
        csvOut << '\n';
        for (std::size_t p = 0; p < recordedPoses.size(); ++p) {
            csvOut << p + 1;
            for (uint16_t tick : recordedPoses[p]) {
                csvOut << ',' << tick;
            }
            csvOut << '\n';
        }
    };

    auto writeSnippet = [&]() {
        std::ofstream snippet(
            OUTPUT_CPP_SNIPPET,
            std::ios::out | std::ios::trunc
        );
        if (!snippet) {
            return;
        }
        snippet
            << "constexpr std::size_t POSE_COUNT = "
            << recordedPoses.size() << ";\n"
            << "const std::array<std::array<uint16_t, JOINT_COUNT>, "
            << "POSE_COUNT> TARGET_POSES = {{\n";
        for (const auto& pose : recordedPoses) {
            snippet << "    {{";
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                snippet << pose[i];
                if (i + 1 < JOINT_COUNT) {
                    snippet << ", ";
                }
            }
            snippet << "}},\n";
        }
        snippet << "}};\n";
    };

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

        std::cout << "\nServo hardware angle limits (independent of "
                   << "jointCalibrations -- the servo itself rejects any "
                   << "goal position outside these):\n";
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

            const JointCalibration& joint = jointCalibrations[id];
            std::cout
                << "  Motor " << id << " hardware limits: ["
                << cwLimit << ", " << ccwLimit << "]"
                << "  jointCalibrations range: ["
                << joint.minTick << ", " << joint.maxTick << "]";
            if (cwLimit > joint.minTick || ccwLimit < joint.maxTick) {
                std::cout
                    << "  <- hardware is TIGHTER than jointCalibrations "
                    << "here; hand-posing near that edge may be rejected";
            }
            std::cout << '\n';
        }

        std::cout
            << "\nWidening every motor's hardware angle limits to the "
            << "full [" << FULL_RANGE_CW_LIMIT << ", " << FULL_RANGE_CCW_LIMIT
            << "] range for this session, so any hand-posed position gets "
            << "accepted. Original limits are printed above and will be "
            << "restored automatically when this program exits normally "
            << "or hits an error -- but NOT if it's killed via Ctrl+C, so "
            << "avoid that this run, or restore them manually afterward if "
            << "you do.\n";
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

        // Start free-moving so the arm can be posed by hand.
        for (int id : motorIds) {
            motor.disableTorque(id);
        }

        std::cout
            << "\nHand-pose recorder\n"
            << "Torque is OFF -- move the arm by hand into a pose.\n"
            << "Press Enter to lock and record that pose ("
            << POSE_COUNT << " new poses this session"
            << (preloadedCount > 0
                    ? ", added to the " + std::to_string(preloadedCount) +
                          " already loaded"
                    : "")
            << ").\n"
            << "Torque briefly enables to hold exactly where the arm "
            << "already is, the position is recorded, then torque "
            << "releases again so you can move to the next pose.\n"
            << "Press Ctrl+C to stop early -- poses recorded so far are "
            << "already saved to both " << OUTPUT_CSV << " and "
            << OUTPUT_CPP_SNIPPET << " (rewritten after every pose, not "
            << "batched at the end).\n";

        for (int poseIndex = 0; poseIndex < POSE_COUNT; ++poseIndex) {
            waitForEnter(
                "\nMove the arm to new pose " +
                std::to_string(poseIndex + 1) + " of " +
                std::to_string(POSE_COUNT) + " (" +
                std::to_string(preloadedCount + poseIndex + 1) +
                " total so far), then press Enter..."
            );

            std::array<uint16_t, JOINT_COUNT> pose{};
            bool locked = false;

            while (!locked) {
                for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                    uint16_t position = 0;
                    if (!motor.readPosition(motorIds[i], position)) {
                        throw std::runtime_error(
                            "Failed to read position for motor " +
                            std::to_string(motorIds[i])
                        );
                    }
                    pose[i] = position;
                }

                // Lock the arm exactly where it already is: write the
                // just-read position back as the goal BEFORE enabling
                // torque. If torque were enabled first while a stale goal
                // from an earlier pose was still in the register, the arm
                // would lurch toward that old target the instant torque
                // came on. Checked for ALL motors before enabling torque on
                // ANY of them, so a rejection here never leaves some
                // motors torqued and others not.
                //
                // Uses writeGoalPositionRaw() (bypasses the jointCalibrations
                // software range check that setGoalPosition() enforces) --
                // this is deliberately freezing the motor at a position it
                // is already physically at, which the user has already
                // visually verified is safe by hand-posing it there, so a
                // hand-chosen pose outside jointCalibrations' current
                // min/max is still accepted rather than rejected.
                bool allGoalsAccepted = true;
                for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                    if (!motor.writeGoalPositionRaw(motorIds[i], pose[i])) {
                        std::cout
                            << "Motor " << motorIds[i] << "'s current "
                            << "position (" << pose[i] << ") failed to "
                            << "write (see the error above -- likely a "
                            << "communication issue, not a range check).\n";
                        allGoalsAccepted = false;
                        break;
                    }
                }

                if (!allGoalsAccepted) {
                    waitForEnter(
                        "Press Enter once repositioned, to try locking "
                        "pose " + std::to_string(poseIndex + 1) +
                        " again..."
                    );
                    continue;
                }

                bool allTorqueEnabled = true;
                for (int id : motorIds) {
                    if (!motor.enableTorque(id)) {
                        std::cout
                            << "Failed to enable torque on motor " << id
                            << " (see the error above).\n";
                        allTorqueEnabled = false;
                        break;
                    }
                }

                if (!allTorqueEnabled) {
                    waitForEnter(
                        "Press Enter to retry pose " +
                        std::to_string(poseIndex + 1) + "..."
                    );
                    continue;
                }

                locked = true;
            }

            std::cout
                << "Recorded new pose " << poseIndex + 1 << " ("
                << preloadedCount + poseIndex + 1 << " total so far): ";
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                const JointCalibration& joint = jointCalibrations[i];
                if (pose[i] < joint.minTick || pose[i] > joint.maxTick) {
                    std::cout
                        << "\n  WARNING: motor " << i << " reading "
                        << pose[i] << " is outside its calibrated range ["
                        << joint.minTick << ", " << joint.maxTick << "]";
                }
                std::cout << pose[i] << ' ';
            }
            std::cout << '\n';

            recordedPoses.push_back(pose);
            writeCsv();
            writeSnippet();

            for (int id : motorIds) {
                motor.disableTorque(id);
            }
        }

        std::cout
            << "\nAll " << POSE_COUNT << " new poses recorded ("
            << recordedPoses.size() << " total).\n"
            << "Saved to " << OUTPUT_CSV << " and " << OUTPUT_CPP_SNIPPET
            << '\n';

        restoreAngleLimits();
        motor.disconnect();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << '\n';
        std::cerr
            << recordedPoses.size() << " pose(s) were already saved to "
            << OUTPUT_CSV << " before this error.\n";

        for (int id : motorIds) {
            motor.disableTorque(id);
        }
        restoreAngleLimits();
        motor.disconnect();
        return 1;
    }
}
