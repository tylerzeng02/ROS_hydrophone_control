#include <array>
#include <fstream>
#include <iostream>
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
constexpr int POSE_COUNT = 100;

constexpr const char* OUTPUT_CSV = "recorded_hand_poses.csv";
constexpr const char* OUTPUT_CPP_SNIPPET = "recorded_hand_poses_target_poses.txt";

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
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

    std::ofstream csv(OUTPUT_CSV, std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Could not open CSV: " << OUTPUT_CSV << '\n';
        return 1;
    }
    csv << "pose_id";
    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        csv << ",tick_" << i;
    }
    csv << '\n';

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

        // Start free-moving so the arm can be posed by hand.
        for (int id : motorIds) {
            motor.disableTorque(id);
        }

        std::cout
            << "\nHand-pose recorder\n"
            << "Torque is OFF -- move the arm by hand into a pose.\n"
            << "Press Enter to lock and record that pose (up to "
            << POSE_COUNT << " total).\n"
            << "Torque briefly enables to hold exactly where the arm "
            << "already is, the position is recorded, then torque "
            << "releases again so you can move to the next pose.\n"
            << "Press Ctrl+C to stop early -- poses recorded so far are "
            << "already saved to " << OUTPUT_CSV << " (each row is "
            << "written and flushed immediately, not batched at the end).\n";

        for (int poseIndex = 0; poseIndex < POSE_COUNT; ++poseIndex) {
            waitForEnter(
                "\nMove the arm to pose " +
                std::to_string(poseIndex + 1) + " of " +
                std::to_string(POSE_COUNT) +
                ", then press Enter..."
            );

            std::array<uint16_t, JOINT_COUNT> pose{};

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

            // Lock the arm exactly where it already is: write the just-read
            // position back as the goal BEFORE enabling torque. If torque
            // were enabled first while a stale goal from an earlier pose
            // was still in the register, the arm would lurch toward that
            // old target the instant torque came on.
            for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
                if (!motor.setGoalPosition(motorIds[i], pose[i])) {
                    throw std::runtime_error(
                        "Failed to set goal position for motor " +
                        std::to_string(motorIds[i])
                    );
                }
            }

            for (int id : motorIds) {
                if (!motor.enableTorque(id)) {
                    throw std::runtime_error(
                        "Failed to enable torque on motor " +
                        std::to_string(id)
                    );
                }
            }

            std::cout << "Recorded pose " << poseIndex + 1 << ": ";
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

            csv << poseIndex + 1;
            for (uint16_t tick : pose) {
                csv << ',' << tick;
            }
            csv << '\n';
            csv.flush();

            recordedPoses.push_back(pose);

            for (int id : motorIds) {
                motor.disableTorque(id);
            }
        }

        std::cout
            << "\nAll " << POSE_COUNT << " poses recorded.\n"
            << "Saved to " << OUTPUT_CSV << '\n';

        std::ofstream snippet(
            OUTPUT_CPP_SNIPPET,
            std::ios::out | std::ios::trunc
        );
        if (snippet) {
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
            std::cout
                << "Ready-to-paste TARGET_POSES snippet written to "
                << OUTPUT_CPP_SNIPPET << '\n';
        }

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
        motor.disconnect();
        return 1;
    }
}
