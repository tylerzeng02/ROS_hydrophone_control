// replay_ndi_capture: drives the arm through the poses recorded in a
// cyton_ndi_capture/ndi_measure output CSV (moveit_ndi_accuracy_check*.csv --
// capture_id,timestamp_ms,shoulder_roll_joint_rad,...,wrist_roll_joint_rad,
// moveit_pose_x_mm,... and the NDI-frame columns) via MoveIt, so you can
// re-visit exactly the poses that were captured earlier (e.g. to redo NDI
// measurements, or just sanity-check the recorded set) instead of jogging
// back to each one by hand.
//
// Unlike pose_commander.cpp (which reads TICK targets and converts them via
// robot_calibration.cpp's ticksToRadians()), this file's input already has
// the 7 joint angles in RADIANS as named columns -- no tick conversion
// needed, just direct setJointValueTarget() per joint by name. Columns are
// resolved by HEADER NAME (not fixed position), same pattern
// move_between_points.cpp already uses for its own input CSV, so this
// works against any file sharing that column-naming convention regardless
// of what other columns it does or doesn't have.
//
// Everything else (plan-preview-then-confirm workflow, the
// ensureCurrentStateWithinBounds()/sendCorrectiveTrajectory() out-of-bounds
// recovery logic for elbow_pitch/elbow_yaw's razor-thin locked windows) is
// carried over from pose_commander.cpp verbatim -- already hardware-
// validated there, not reimplemented differently here. Recovery still uses
// jointCalibrations directly (from robot_calibration.h, compiled into this
// binary same as pose_commander), which is independent of how the target
// points themselves were loaded.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"
#include "moveit_msgs/msg/move_it_error_codes.hpp"

#include "robot_calibration.h"

namespace {

constexpr const char* DEFAULT_INPUT_CSV =
    "/home/temp/dev/cyton_setup/ros/skull_probe_accuracy_test_target_points.csv";
constexpr const char* PLANNING_GROUP = "arm";
constexpr double PLANNING_TIME_SECONDS = 5.0;

// Same buffer/rationale as pose_commander.cpp's identical constant.
constexpr double RECOVERY_BUFFER_RAD = 0.01;

// Same order as pose_commander.cpp / jointCalibrations / the URDF's
// <ros2_control> block -- and matches the column-name prefixes ndi_measure
// writes (shoulder_roll_joint_rad, etc.).
constexpr std::array<const char*, 7> JOINT_NAMES = {
    "shoulder_roll_joint", "shoulder_pitch_joint", "shoulder_yaw_joint",
    "elbow_pitch_joint",   "elbow_yaw_joint",       "wrist_pitch_joint",
    "wrist_roll_joint",
};

constexpr std::array<const char*, 7> JOINT_RAD_COLUMNS = {
    "shoulder_roll_joint_rad", "shoulder_pitch_joint_rad", "shoulder_yaw_joint_rad",
    "elbow_pitch_joint_rad",   "elbow_yaw_joint_rad",       "wrist_pitch_joint_rad",
    "wrist_roll_joint_rad",
};

struct CapturedPose {
    int captureId = 0;
    std::array<double, 7> radians{};
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

std::vector<CapturedPose> loadCapturedPoses(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open input CSV: " + path);
    }

    std::string headerLine;
    if (!std::getline(file, headerLine)) {
        throw std::runtime_error("Input CSV is empty: " + path);
    }
    const std::vector<std::string> columns = splitCsvLine(headerLine);

    int captureIdCol = -1;
    std::array<int, 7> jointCols{};
    jointCols.fill(-1);
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == "capture_id") {
            captureIdCol = static_cast<int>(i);
        }
        for (std::size_t j = 0; j < 7; ++j) {
            if (columns[i] == JOINT_RAD_COLUMNS[j]) {
                jointCols[j] = static_cast<int>(i);
            }
        }
    }
    for (std::size_t j = 0; j < 7; ++j) {
        if (jointCols[j] < 0) {
            throw std::runtime_error(
                std::string("Input CSV ") + path + " is missing column " + JOINT_RAD_COLUMNS[j] +
                " -- expected the format written by cyton_ndi_capture's ndi_measure."
            );
        }
    }

    std::vector<CapturedPose> poses;
    std::string line;
    int rowIndex = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = splitCsvLine(line);
        const int maxColNeeded =
            *std::max_element(jointCols.begin(), jointCols.end());
        if (static_cast<int>(fields.size()) <= maxColNeeded) {
            std::cout << "WARNING: skipping malformed row: " << line << '\n';
            continue;
        }

        CapturedPose pose;
        pose.captureId = (captureIdCol >= 0) ? std::stoi(fields[static_cast<std::size_t>(captureIdCol)])
                                              : rowIndex;
        for (std::size_t j = 0; j < 7; ++j) {
            pose.radians[j] = std::stod(fields[static_cast<std::size_t>(jointCols[j])]);
        }
        poses.push_back(pose);
        ++rowIndex;
    }

    return poses;
}

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

// --- Everything below is carried over from pose_commander.cpp verbatim ---

std::pair<double, double> safeBoundsRadians(const JointCalibration& calibration) {
    const double a = ticksToRadians(calibration, calibration.minTick);
    const double b = ticksToRadians(calibration, calibration.maxTick);
    const double lo = std::min(a, b);
    const double hi = std::max(a, b);
    return {lo + RECOVERY_BUFFER_RAD, hi - RECOVERY_BUFFER_RAD};
}

bool sendCorrectiveTrajectory(const std::array<double, 7>& radians) {
    const std::string logPath = "/tmp/replay_ndi_capture_recovery.log";

    std::ostringstream cmd;
    cmd << "ros2 action send_goal /arm_controller/follow_joint_trajectory "
           "control_msgs/action/FollowJointTrajectory \"trajectory: {joint_names: [";
    for (std::size_t j = 0; j < 7; ++j) {
        cmd << JOINT_NAMES[j] << (j + 1 < 7 ? ", " : "");
    }
    cmd << "], points: [{positions: [";
    for (std::size_t j = 0; j < 7; ++j) {
        cmd << std::setprecision(10) << radians[j] << (j + 1 < 7 ? ", " : "");
    }
    cmd << "], time_from_start: {sec: 4, nanosec: 0}}]}\" > " << logPath << " 2>&1";

    std::system(cmd.str().c_str());

    std::ifstream log(logPath);
    const std::string content(
        (std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>()
    );
    return content.find("SUCCEEDED") != std::string::npos;
}

bool ensureCurrentStateWithinBounds(moveit::planning_interface::MoveGroupInterface& moveGroup) {
    auto currentState = moveGroup.getCurrentState(2.0);
    if (!currentState) {
        std::cout << "  WARNING: could not read current robot state to check bounds -- "
                     "proceeding without this check.\n";
        return true;
    }

    std::array<double, 7> corrected{};
    bool anyClamped = false;
    for (std::size_t j = 0; j < 7; ++j) {
        double value = currentState->getVariablePosition(JOINT_NAMES[j]);
        const auto bounds = safeBoundsRadians(jointCalibrations[j]);
        if (value < bounds.first) {
            std::cout << "  " << JOINT_NAMES[j] << " is currently " << value
                      << " rad, below safe bound " << bounds.first << " -- clamping.\n";
            value = bounds.first;
            anyClamped = true;
        } else if (value > bounds.second) {
            std::cout << "  " << JOINT_NAMES[j] << " is currently " << value
                      << " rad, above safe bound " << bounds.second << " -- clamping.\n";
            value = bounds.second;
            anyClamped = true;
        }
        corrected[j] = value;
    }

    if (!anyClamped) {
        return true;
    }

    std::cout << "  Current state has a joint out of its safe bound (most likely servo "
                 "settling noise left over from an earlier move) -- sending a corrective "
                 "trajectory directly to the controller (bypassing MoveIt, since it won't "
                 "plan from an invalid start state)...\n";
    return sendCorrectiveTrajectory(corrected);
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [input_csv]\n"
              << "  input_csv: a cyton_ndi_capture/ndi_measure output CSV (capture_id,"
                 "...,shoulder_roll_joint_rad,...,wrist_roll_joint_rad,...).\n"
              << "  Default: " << DEFAULT_INPUT_CSV << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }
    const std::string inputCsv = argc > 1 ? argv[1] : DEFAULT_INPUT_CSV;

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("replay_ndi_capture");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        std::cout << "Loading captured poses from " << inputCsv << "...\n";
        std::vector<CapturedPose> poses = loadCapturedPoses(inputCsv);
        std::cout << "Loaded " << poses.size() << " captured poses.\n";
        if (poses.empty()) {
            throw std::runtime_error("No poses loaded -- nothing to do.");
        }

        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);

        std::cout << "\nReady. For each pose: this program PLANS the move (check RViz for a "
                     "preview) and waits for Enter before actually executing it. Ctrl+C to stop "
                     "at any point.\n";

        for (std::size_t i = 0; i < poses.size(); ++i) {
            const CapturedPose& pose = poses[i];

            std::cout << "\n=== Pose " << (i + 1) << "/" << poses.size()
                      << " (capture_id=" << pose.captureId << ") ===\n";
            for (std::size_t j = 0; j < 7; ++j) {
                std::cout << "  " << JOINT_NAMES[j] << ": rad=" << pose.radians[j] << '\n';
            }

            bool anyTargetRejected = false;
            for (std::size_t j = 0; j < 7; ++j) {
                if (!moveGroup.setJointValueTarget(JOINT_NAMES[j], pose.radians[j])) {
                    std::cout << "  WARNING: setJointValueTarget rejected " << JOINT_NAMES[j]
                              << " = " << pose.radians[j]
                              << " rad (likely outside its URDF limit) -- skipping this pose.\n";
                    anyTargetRejected = true;
                }
            }
            if (anyTargetRejected) {
                continue;
            }

            if (!ensureCurrentStateWithinBounds(moveGroup)) {
                std::cout << "  Could not correct an out-of-bounds current state -- skipping "
                             "this pose.\n";
                continue;
            }

            std::cout << "  Planning...\n";
            moveit::planning_interface::MoveGroupInterface::Plan planMsg;
            auto planResult = moveGroup.plan(planMsg);

            if (planResult != moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  Planning FAILED, error code " << planResult.val
                          << " -- checking for an out-of-bounds current state and retrying "
                             "once...\n";
                if (!ensureCurrentStateWithinBounds(moveGroup)) {
                    std::cout << "  Could not correct current state -- skipping this pose.\n";
                    continue;
                }
                for (std::size_t j = 0; j < 7; ++j) {
                    moveGroup.setJointValueTarget(JOINT_NAMES[j], pose.radians[j]);
                }
                planResult = moveGroup.plan(planMsg);
                if (planResult != moveit::core::MoveItErrorCode::SUCCESS) {
                    std::cout << "  Planning FAILED again, error code " << planResult.val
                              << " -- skipping this pose.\n";
                    continue;
                }
            }

            std::cout << "  Plan found -- check RViz's MotionPlanning display for a preview "
                         "(it's published automatically, no action needed here).\n";
            waitForEnter(
                "  Press Enter to execute this planned move, or Ctrl+C to stop without "
                "moving... "
            );

            std::cout << "  Executing...\n";
            auto execResult = moveGroup.execute(planMsg);
            if (execResult == moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  Execution succeeded.\n";
            } else {
                std::cout << "  Execution FAILED, error code " << execResult.val
                          << " -- continuing to the next pose.\n";
            }
        }

        std::cout << "\nAll poses done.\n";
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    executor.cancel();
    spinThread.join();
    rclcpp::shutdown();
    return exitCode;
}
