// pose_commander: drives the arm through a CSV list of tick-domain joint
// targets via MoveIt's MoveGroupInterface (plan+execute, collision-aware)
// instead of commanding raw ticks directly.
//
// This is the "move" half of the MoveIt-driven NDI accuracy check; the
// "measure" half is cyton_ndi_capture's ndi_measure, run separately in
// another terminal. Workflow: this program moves the arm to pose N and
// waits for Enter; in the meantime, switch to the ndi_measure terminal and
// press Enter there to capture pose N; switch back here and press Enter to
// advance to pose N+1. Ctrl+C to quit early.
//
// Default input is build/repeatability_test_8points_labeled.csv (point_id,
// tick_0..tick_6, physical_only_deviation_mm, gp_corrected_deviation_mm --
// only the first 8 columns are used; the deviation columns are printed for
// reference only, from the earlier physical validation that used this same
// dataset). Deliberately NOT validation_ticks.csv -- that dataset predates
// the elbow_yaw permanent lock and targets tick_4 values (1553-2639) far
// outside its current locked range [2075, 2115]; this one's tick_4 values
// (2086-2094) are all safely inside it.
//
// Tick->radian conversion goes through robot_calibration.cpp's
// ticksToRadians() (compiled directly into this binary, see
// CMakeLists.txt) -- not reimplemented here, per this project's own
// standing rule that every consumer of jointCalibrations must share the
// one calibrated conversion, not a second copy that can drift.

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"

#include "robot_calibration.h"

namespace {

constexpr const char* DEFAULT_INPUT_CSV =
    "/home/temp/dev/cyton_setup/build/repeatability_test_8points_labeled.csv";
constexpr const char* PLANNING_GROUP = "arm";
constexpr double PLANNING_TIME_SECONDS = 5.0;

// Motor-ID order, matching jointCalibrations / the URDF's <ros2_control>
// block / every other joint-name array in this workspace.
constexpr std::array<const char*, 7> JOINT_NAMES = {
    "shoulder_roll_joint", "shoulder_pitch_joint", "shoulder_yaw_joint",
    "elbow_pitch_joint",   "elbow_yaw_joint",       "wrist_pitch_joint",
    "wrist_roll_joint",
};

struct TestPoint {
    int pointId = 0;
    std::array<int, 7> ticks{};
    std::array<double, 7> radians{};
    bool hasHistoricalDeviation = false;
    double physicalOnlyDeviationMm = 0.0;
    double gpCorrectedDeviationMm = 0.0;
};

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<TestPoint> loadTestPoints(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open input CSV: " + path);
    }

    std::vector<TestPoint> points;
    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        auto fields = splitCsvLine(line);

        if (firstLine) {
            firstLine = false;
            // Header row if the first field isn't purely numeric (e.g. "point_id").
            try {
                std::stoi(fields.at(0));
            } catch (...) {
                continue;  // header, skip
            }
        }

        if (fields.size() < 8) {
            throw std::runtime_error("Row has fewer than 8 columns (point_id + 7 ticks): " + line);
        }

        TestPoint point;
        point.pointId = std::stoi(fields[0]);
        for (std::size_t i = 0; i < 7; ++i) {
            point.ticks[i] = std::stoi(fields[1 + i]);
        }
        if (fields.size() >= 10) {
            point.hasHistoricalDeviation = true;
            point.physicalOnlyDeviationMm = std::stod(fields[8]);
            point.gpCorrectedDeviationMm = std::stod(fields[9]);
        }

        for (std::size_t i = 0; i < 7; ++i) {
            if (jointCalibrations.size() <= i) {
                throw std::runtime_error("jointCalibrations has fewer than 7 entries.");
            }
            point.radians[i] = ticksToRadians(jointCalibrations[i], point.ticks[i]);
        }

        points.push_back(point);
    }

    return points;
}

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);  // see the identical comment in cyton_ndi_capture's tools

    const std::string inputCsv = argc > 1 ? argv[1] : DEFAULT_INPUT_CSV;

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("pose_commander");

    int exitCode = 0;
    try {
        std::cout << "Loading test points from " << inputCsv << "...\n";
        std::vector<TestPoint> points = loadTestPoints(inputCsv);
        std::cout << "Loaded " << points.size() << " test points.\n";

        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);

        std::cout << "\nReady. For each pose: this program moves the arm and then waits for "
                     "Enter before continuing -- capture the pose with ndi_measure (in another "
                     "terminal) during that pause.\n";

        for (std::size_t i = 0; i < points.size(); ++i) {
            const TestPoint& point = points[i];

            std::cout << "\n=== Pose " << (i + 1) << "/" << points.size()
                      << " (point_id=" << point.pointId << ") ===\n";
            for (std::size_t j = 0; j < 7; ++j) {
                std::cout << "  " << JOINT_NAMES[j] << ": tick=" << point.ticks[j] << ", rad="
                          << point.radians[j] << '\n';
            }
            if (point.hasHistoricalDeviation) {
                std::cout << "  (historical deviation at this point: physical-only "
                          << point.physicalOnlyDeviationMm << "mm, GP-corrected "
                          << point.gpCorrectedDeviationMm << "mm)\n";
            }

            for (std::size_t j = 0; j < 7; ++j) {
                if (!moveGroup.setJointValueTarget(JOINT_NAMES[j], point.radians[j])) {
                    std::cout << "  WARNING: setJointValueTarget rejected " << JOINT_NAMES[j]
                              << " = " << point.radians[j]
                              << " rad (likely outside its URDF limit) -- skipping this pose.\n";
                    continue;
                }
            }

            std::cout << "  Planning and executing...\n";
            const auto result = moveGroup.move();

            if (result == moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  Move succeeded.\n";
            } else {
                std::cout << "  Move FAILED, error code " << result.val
                          << " -- skipping NDI capture for this pose, continuing to the next.\n";
                continue;
            }

            if (i + 1 < points.size()) {
                waitForEnter(
                    "  Capture this pose with ndi_measure now, then press Enter here to "
                    "continue to the next pose (Ctrl+C to stop)... "
                );
            }
        }

        std::cout << "\nAll poses done.\n";
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    rclcpp::shutdown();
    return exitCode;
}
