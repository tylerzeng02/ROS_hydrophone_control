/**
 * @file pose_commander.cpp
 * @brief Drives the arm through a CSV list of tick-domain joint targets
 * via MoveIt's MoveGroupInterface (collision-aware planning), instead of
 * commanding raw ticks directly.
 *
 * Plan-preview-then-confirm workflow: for each pose, plans the move
 * (moveGroup.plan(), not move()) and stops. MoveIt automatically
 * publishes the planned trajectory to /display_planned_path, which
 * RViz's MotionPlanning display shows with no extra code here. Waits for
 * Enter before calling execute() on that plan, so a visual preview
 * happens before committing. Ctrl+C at any point quits early.
 *
 * Useful for visually sanity-checking a batch of joint configurations via
 * RViz preview before trusting them; cyton_accuracy_check's combined
 * program covers the synchronized move-then-NDI-capture case this does
 * not.
 *
 * Default input is build/repeatability_test_8points_labeled.csv (only the
 * first 8 columns used; deviation columns printed for reference).
 *
 * Tick-to-radian conversion goes through robot_calibration.cpp's
 * ticksToRadians() (compiled directly into this binary), not
 * reimplemented, per this project's rule that every jointCalibrations
 * consumer shares one calibrated conversion.
 */

#include <algorithm>
#include <array>
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
    "/home/temp/dev/cyton_setup/build/repeatability_test_8points_labeled.csv";
constexpr const char* PLANNING_GROUP = "arm";
constexpr double PLANNING_TIME_SECONDS = 5.0;

// How far inside jointCalibrations' min/maxTick-derived bounds a recovery
// correction lands, so it doesn't just clamp back onto the boundary it
// just tripped (which real settling noise could immediately re-cross).
// ~0.01 rad is roughly 6.5 ticks, small enough to be just enough headroom.
constexpr double RECOVERY_BUFFER_RAD = 0.01;

// Motor-ID order, matching jointCalibrations / the URDF's <ros2_control>
// block / every other joint-name array in this workspace.
constexpr std::array<const char*, 7> JOINT_NAMES = {
    "shoulder_roll_joint", "shoulder_pitch_joint", "shoulder_yaw_joint",
    "elbow_pitch_joint",   "elbow_yaw_joint",       "wrist_pitch_joint",
    "wrist_roll_joint",
};

/**
 * @brief One target pose read from the input CSV: its raw ticks and the
 * corresponding radians (via ticksToRadians()), plus optional historical
 * deviation figures carried through for reference only.
 */
struct TestPoint {
    int pointId = 0;
    std::array<int, 7> ticks{};
    std::array<double, 7> radians{};
    bool hasHistoricalDeviation = false;
    double physicalOnlyDeviationMm = 0.0;
    double gpCorrectedDeviationMm = 0.0;
};

/**
 * @brief Splits one CSV line on commas.
 * @param line Line to split.
 * @return The comma-separated fields, in order.
 */
std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

/**
 * @brief Loads TestPoint rows from a CSV (point_id, tick_0..tick_6[,
 * physical_only_deviation_mm, gp_corrected_deviation_mm]).
 * @param path CSV path to load.
 * @return The loaded points, with radians already computed via
 *         ticksToRadians().
 * @throws std::runtime_error if `path` cannot be opened.
 */
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

/**
 * @brief Prints `message` and blocks until Enter is pressed.
 * @param message Prompt text to print before waiting.
 */
void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

/**
 * @brief Computes [min, max] radian bounds for a joint from
 * jointCalibrations, pulled in by RECOVERY_BUFFER_RAD. Takes min/max of
 * both endpoints' converted values rather than assuming minTick maps to
 * the lower bound, in case `direction` is ever -1 for a future joint (it
 * is always +1 today).
 * @param calibration Joint calibration to compute bounds for.
 * @return (lower, upper) radian bounds, each pulled in from the raw
 *         min/maxTick by RECOVERY_BUFFER_RAD.
 */
std::pair<double, double> safeBoundsRadians(const JointCalibration& calibration) {
    const double a = ticksToRadians(calibration, calibration.minTick);
    const double b = ticksToRadians(calibration, calibration.maxTick);
    const double lo = std::min(a, b);
    const double hi = std::max(a, b);
    return {lo + RECOVERY_BUFFER_RAD, hi - RECOVERY_BUFFER_RAD};
}

/**
 * @brief Sends a one-off corrective trajectory directly to the
 * arm_controller, bypassing MoveIt's planning pipeline entirely.
 *
 * Deliberate: move_group refuses to plan anything while the current
 * state is out of bounds (START_STATE_INVALID), so MoveGroupInterface
 * cannot fix it itself. Shells out to `ros2 action send_goal` rather than
 * a hand-rolled action client, avoiding any interaction with
 * MoveGroupInterface's own internal spinning on this node.
 *
 * @param radians Target position, one value per joint in JOINT_NAMES
 *        order.
 * @return True if the action reported SUCCEEDED.
 */
bool sendCorrectiveTrajectory(const std::array<double, 7>& radians) {
    const std::string logPath = "/tmp/pose_commander_recovery.log";

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

    // Check the actual reported outcome in the log, not just the shell's
    // exit code, matching this project's own established practice of
    // verifying the real result rather than trusting exit status alone.
    std::ifstream log(logPath);
    const std::string content(
        (std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>()
    );
    return content.find("SUCCEEDED") != std::string::npos;
}

/**
 * @brief Checks the arm's live current state against jointCalibrations'
 * safe bounds and sends a corrective trajectory if any joint is out.
 *
 * Ordinary servo settling noise crossing a locked joint's razor-thin
 * URDF window is the recurring cause. Called proactively before every
 * plan attempt, not just reactively after a START_STATE_INVALID failure,
 * since the same problem can also surface through move_group's
 * CheckStartStateBounds adapter, which aborts before assigning a specific
 * error code (the client just sees a generic FAILURE) and so cannot be
 * distinguished from other failures after the fact.
 *
 * @param moveGroup Interface to read the current state from and, if
 *        needed, correct.
 * @return True if the state was already within bounds, or the correction
 *         succeeded.
 */
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

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);  // see the identical comment in cyton_ndi_capture's tools

    const std::string inputCsv = argc > 1 ? argv[1] : DEFAULT_INPUT_CSV;

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("pose_commander");

    // getCurrentState() (used by the recovery path) depends on this node's
    // CurrentStateMonitor subscription to /joint_states, which needs the
    // node spinning. Without it, getCurrentState() times out ("latest
    // received state has time 0.000000"). This is a standard, supported
    // pattern alongside MoveGroupInterface's own internal handling.
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        std::cout << "Loading test points from " << inputCsv << "...\n";
        std::vector<TestPoint> points = loadTestPoints(inputCsv);
        std::cout << "Loaded " << points.size() << " test points.\n";

        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);

        std::cout << "\nReady. For each pose: this program PLANS the move (check RViz for a "
                     "preview) and waits for Enter before actually executing it. Ctrl+C to stop "
                     "at any point.\n";

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

            bool anyTargetRejected = false;
            for (std::size_t j = 0; j < 7; ++j) {
                if (!moveGroup.setJointValueTarget(JOINT_NAMES[j], point.radians[j])) {
                    std::cout << "  WARNING: setJointValueTarget rejected " << JOINT_NAMES[j]
                              << " = " << point.radians[j]
                              << " rad (likely outside its URDF limit) -- skipping this pose.\n";
                    anyTargetRejected = true;
                    // Keep looping so every rejected joint gets reported, not just the first.
                }
            }
            if (anyTargetRejected) {
                // Must continue the outer per-pose loop, not just the inner
                // per-joint one, otherwise move() runs with a stale or
                // partial target.
                continue;
            }

            // Proactive check before planning; see ensureCurrentStateWithinBounds().
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
                    moveGroup.setJointValueTarget(JOINT_NAMES[j], point.radians[j]);
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
