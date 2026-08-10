// pose_commander: drives the arm through a CSV list of tick-domain joint
// targets via MoveIt's MoveGroupInterface (collision-aware planning)
// instead of commanding raw ticks directly.
//
// Plan-preview-then-confirm workflow (2026-08-07): for each pose, this
// program PLANS the move (moveGroup.plan(), not move()) and stops there --
// MoveIt automatically publishes the planned trajectory to
// /display_planned_path when plan() is called (the DisplayMotionPath
// response adapter, already loaded in this project's move_group config),
// which RViz's MotionPlanning display shows on its own, no extra code
// needed here. The program then waits for Enter before calling execute()
// on that same plan -- so you get a visual preview of exactly what's about
// to happen before committing to it, rather than the arm just moving.
// Ctrl+C at any point (including during the pause) to quit early.
//
// This was originally the "move" half of a two-terminal MoveIt-driven NDI
// accuracy check (pose_commander here + cyton_ndi_capture's ndi_measure in
// another terminal) -- that workflow was later found to have a real timing
// race (see CLAUDE.md) and mostly superseded by cyton_accuracy_check's
// single combined program for that specific purpose. This tool is still
// useful on its own for exactly what it does now: visually sanity-checking
// a batch of joint configurations (e.g. a freshly hand-posed dataset) via
// MoveIt's collision-aware planner and RViz preview before trusting them.
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
// ~0.01 rad is roughly 6.5 ticks -- small, just enough headroom.
constexpr double RECOVERY_BUFFER_RAD = 0.01;

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

// Returns [min, max] radian bounds for a joint from jointCalibrations,
// pulled in by RECOVERY_BUFFER_RAD. Takes min/max of both endpoints'
// converted values rather than assuming minTick maps to the lower bound,
// in case direction is ever -1 for a future joint (it's always +1 today).
std::pair<double, double> safeBoundsRadians(const JointCalibration& calibration) {
    const double a = ticksToRadians(calibration, calibration.minTick);
    const double b = ticksToRadians(calibration, calibration.maxTick);
    const double lo = std::min(a, b);
    const double hi = std::max(a, b);
    return {lo + RECOVERY_BUFFER_RAD, hi - RECOVERY_BUFFER_RAD};
}

// Sends a one-off corrective trajectory directly to the arm_controller,
// bypassing MoveIt's planning pipeline entirely -- deliberate, since the
// whole reason this function exists is that move_group refuses to plan
// *anything* while the current state is out of bounds (the
// START_STATE_INVALID error this recovers from), so MoveGroupInterface
// itself cannot be used to fix it. Shells out to the same `ros2 action
// send_goal` mechanism used manually, by hand, to recover from this exact
// situation multiple times earlier this project session (see CLAUDE.md) --
// proven reliable, and far simpler/lower-risk than a hand-rolled
// rclcpp_action client sharing a node with MoveGroupInterface's own
// internal (and not fully documented) spinning behavior.
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
    // exit code -- matches this project's own established "verify the
    // real result, don't trust exit status alone" practice.
    std::ifstream log(logPath);
    const std::string content(
        (std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>()
    );
    return content.find("SUCCEEDED") != std::string::npos;
}

// Checks the arm's LIVE current state against jointCalibrations' safe
// bounds and, if any joint is out (ordinary servo settling noise crossing
// a locked joint's razor-thin URDF window is the recurring real-world
// cause here -- see CLAUDE.md), sends a corrective trajectory to pull it
// back in before returning. Returns true if the state was already fine or
// correction succeeded; false if correction was needed but failed.
//
// Originally this correction only ran REACTIVELY, after a plan() call
// failed with error code START_STATE_INVALID specifically. Real bug found
// 2026-08-07: the SAME underlying problem (elbow_yaw stuck ~3-4 ticks past
// its bound) can also surface through move_group's CheckStartStateBounds
// planning-request-adapter, which aborts the pipeline before the planner
// itself ever assigns a specific error code -- the client just sees the
// generic FAILURE (99999), which the old code didn't recognize as
// recoverable. Confirmed directly in move_group's own log: 20 consecutive
// poses all failed identically, in ~20ms each (far too fast for real
// planning), all logging "Joint 'elbow_yaw_joint' from the starting state
// is outside bounds" -- the arm was stuck the whole time and every pose
// kept re-discovering the same un-fixed problem. Now called PROACTIVELY
// before every plan attempt, so it isn't dependent on guessing which
// error code a given failure path happens to report.
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

    // MoveGroupInterface's move()/plan() calls work without this (they use
    // their own internal service/action response handling), but
    // getCurrentState() -- used by the START_STATE_INVALID recovery path --
    // depends on this node's CurrentStateMonitor subscription to
    // /joint_states, which never receives anything unless the node is
    // actually being spun by someone. First real run without this spin
    // thread confirmed the failure directly: "Didn't receive robot state...
    // latest received state has time 0.000000" -- getCurrentState() timed
    // out because nothing had ever spun the node, not a real connectivity
    // problem. Spinning in a background thread alongside MoveGroupInterface
    // is a standard, supported pattern (same as cyton_ndi_capture's
    // ndi_measure), not a conflict with its own internal handling.
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
                // This used to only `continue` the per-joint loop above, not this
                // outer per-pose loop -- meaning move() still ran afterward with a
                // stale/partial target (whatever the rejected joint's value was left
                // at from a previous iteration). Real bug, caught 2026-08-07 when a
                // "skipping this pose" warning was immediately followed by
                // "Planning and executing..." for that same pose.
                continue;
            }

            // Proactive check (2026-08-07): correct an out-of-bounds current state
            // BEFORE attempting to plan, rather than only reacting after a failure --
            // see ensureCurrentStateWithinBounds()'s own comment for why the reactive-
            // only version missed this same problem when it resurfaced through a
            // different move_group error path.
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
