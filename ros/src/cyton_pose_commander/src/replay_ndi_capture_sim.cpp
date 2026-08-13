// replay_ndi_capture_sim: automatic-animation version of
// replay_ndi_capture.cpp -- visits every pose in a cyton_ndi_capture/
// ndi_measure output CSV (moveit_ndi_accuracy_check*.csv) in order via
// MoveIt, pausing PAUSE_AFTER_POSE_MS between each, with NO per-pose Enter
// confirmation. Built for demo/simulation purposes: showing the arm sweep
// through a real recorded set of points (e.g. the skull-probing target
// set) as a hands-off animation, same intent as waypoint_sequence_demo.cpp
// but consuming the joint-radian CSV format ndi_measure writes instead of
// a plain x,y,z,roll,pitch,yaw waypoint file.
//
// Safety note: unlike replay_ndi_capture.cpp (which pauses for Enter
// before every single execute -- deliberately, so you can sanity-check
// each move first), this tool moves through the WHOLE list automatically
// once started. Fine and intended for hardware_type:=mock_components (a
// simulated demo, nothing physically moves). If pointed at a real-hardware
// launch, this WILL physically drive the arm through every recorded pose
// unsupervised -- only do that deliberately, with the same care as any
// other unattended real-hardware sequence in this project.
//
// CSV parsing, JOINT_NAMES, safeBoundsRadians()/sendCorrectiveTrajectory()/
// ensureCurrentStateWithinBounds() are carried over from
// replay_ndi_capture.cpp verbatim -- same already-validated recovery logic
// for elbow_pitch/elbow_yaw's razor-thin locked windows.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
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
    "/home/temp/dev/cyton_setup/ros/moveit_ndi_accuracy_check_easypoints.csv";
constexpr const char* PLANNING_GROUP = "arm";
constexpr double PLANNING_TIME_SECONDS = 5.0;
constexpr int PAUSE_AFTER_POSE_MS = 1000;  // "pausing for a second" per pose

constexpr double RECOVERY_BUFFER_RAD = 0.01;

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
        const int maxColNeeded = *std::max_element(jointCols.begin(), jointCols.end());
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

// --- Carried over from replay_ndi_capture.cpp / pose_commander.cpp verbatim ---

std::pair<double, double> safeBoundsRadians(const JointCalibration& calibration) {
    const double a = ticksToRadians(calibration, calibration.minTick);
    const double b = ticksToRadians(calibration, calibration.maxTick);
    const double lo = std::min(a, b);
    const double hi = std::max(a, b);
    return {lo + RECOVERY_BUFFER_RAD, hi - RECOVERY_BUFFER_RAD};
}

bool sendCorrectiveTrajectory(const std::array<double, 7>& radians) {
    const std::string logPath = "/tmp/replay_ndi_capture_sim_recovery.log";

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

    std::cout << "  Current state has a joint out of its safe bound -- sending a corrective "
                 "trajectory directly to the controller...\n";
    return sendCorrectiveTrajectory(corrected);
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [input_csv]\n"
              << "  input_csv: a cyton_ndi_capture/ndi_measure output CSV (capture_id,"
                 "...,shoulder_roll_joint_rad,...,wrist_roll_joint_rad,...).\n"
              << "  Default: " << DEFAULT_INPUT_CSV << '\n'
              << "  Fully automatic: visits every pose in order, pausing "
              << PAUSE_AFTER_POSE_MS << "ms between each, no per-pose confirmation.\n"
              << "  Intended for hardware_type:=mock_components demos -- if run against "
                 "real hardware it WILL physically move through every pose unattended.\n";
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
    auto node = std::make_shared<rclcpp::Node>("replay_ndi_capture_sim");

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

        std::cout << "\nRunning fully automatically -- visiting all " << poses.size()
                  << " poses in order, pausing " << PAUSE_AFTER_POSE_MS
                  << "ms between each. Ctrl+\\ (not Ctrl+C, see project notes) to stop early.\n";

        int succeeded = 0;
        for (std::size_t i = 0; i < poses.size(); ++i) {
            const CapturedPose& pose = poses[i];

            std::cout << "\n=== Pose " << (i + 1) << "/" << poses.size()
                      << " (capture_id=" << pose.captureId << ") ===\n";

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

            std::cout << "  Planning and executing...\n";
            auto result = moveGroup.move();

            if (result == moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  Reached pose " << (i + 1) << ".\n";
                ++succeeded;
            } else {
                std::cout << "  FAILED, error code " << result.val
                          << " -- skipping to next pose.\n";
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(PAUSE_AFTER_POSE_MS));
        }

        std::cout << "\nDone. " << succeeded << "/" << poses.size() << " poses reached.\n";
        if (succeeded < static_cast<int>(poses.size())) {
            exitCode = 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    executor.cancel();
    spinThread.join();
    rclcpp::shutdown();
    return exitCode;
}
