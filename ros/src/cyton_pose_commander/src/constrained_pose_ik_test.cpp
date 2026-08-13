// constrained_pose_ik_test: deliberately replicates the ORIGINAL,
// harder-than-necessary IK scenario that move_between_points.cpp used
// BEFORE its 2026-08-11 seeded-IK fix -- setPoseTarget() with the current
// orientation held fixed (not just position), which hands a full 6-DOF
// Cartesian constraint to OMPL's own random-seeded goal-region sampling.
// That's exactly what produced the real, logged 30-second RRTConnect
// timeouts against elbow_yaw's razor-thin locked window ("Unable to sample
// any valid states for goal tree").
//
// Purpose: this project fixed that specific problem by switching to seeded
// setJointValueTarget() instead -- which sidesteps the question of whether
// a different IK PLUGIN (TRAC-IK vs. KDL) would have handled the harder,
// random-seeded version any better. This tool exists purely to answer that
// now-separate question: run it once against ik_solver:=kdl and once
// against ik_solver:=trac_ik (same launch, same target list) and compare
// success rate / timing directly.
//
// Deliberately has NO NDI dependency at all -- pure MoveIt planning/
// execution stress test, so it's usable regardless of tracker availability
// and isolates the IK question from any measurement concerns. Reads a
// small Cartesian offset (meters, position-only) per line from a CSV, and
// for each one: reads the arm's LIVE current pose via getCurrentPose(),
// builds an absolute target = current position + offset (orientation held
// at whatever it currently is -- this is the part that makes it "harder"
// than the seeded approach), calls setPoseTarget() + move(), and reports
// success/failure and wall-clock time taken for that point. Continues to
// the next point on failure rather than aborting the run, same "skip and
// continue" convention as move_between_points.cpp/waypoint_sequence_demo.

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"

namespace {

constexpr const char* DEFAULT_INPUT_CSV = "constrained_pose_ik_test_offsets.csv";
constexpr const char* PLANNING_GROUP = "arm";
// Same values established for this exact class of problem elsewhere in
// this project (move_x_test.cpp, the original move_between_points.cpp) --
// kept identical here so a KDL run and a TRAC-IK run are compared under
// the same budget, not an apples-to-oranges one.
constexpr double PLANNING_TIME_SECONDS = 30.0;
constexpr unsigned int NUM_PLANNING_ATTEMPTS = 15;

// Built-in default offsets (meters), used if no CSV is given or found --
// deliberately spans a similar small-to-moderate magnitude range as the
// real easypoints commanded deltas seen elsewhere in this project
// (roughly 1-36mm), so this test is representative of real usage rather
// than artificially easy or hard.
const std::vector<std::array<double, 3>> DEFAULT_OFFSETS = {
    {0.030, -0.010, 0.015},  {0.005, 0.004, -0.002},  {0.012, -0.014, -0.002},
    {0.015, -0.017, -0.001}, {0.010, -0.011, -0.004}, {0.007, -0.005, -0.005},
    {-0.001, 0.022, 0.001},  {-0.001, 0.001, 0.001},  {-0.001, -0.001, 0.0002},
    {-0.005, -0.014, -0.004}, {0.022, 0.008, 0.006},  {0.001, -0.002, -0.004},
};

std::vector<std::array<double, 3>> loadOffsets(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cout << "Could not open " << path << " -- using built-in default offsets ("
                  << DEFAULT_OFFSETS.size() << " points).\n";
        return DEFAULT_OFFSETS;
    }

    std::vector<std::array<double, 3>> offsets;
    std::string line;
    bool firstLine = true;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string field;
        std::vector<double> fields;
        while (std::getline(ss, field, ',')) {
            try {
                fields.push_back(std::stod(field));
            } catch (...) {
                fields.clear();
                break;  // non-numeric row (e.g. header) -- skip
            }
        }
        if (firstLine) {
            firstLine = false;
            if (fields.size() < 3) {
                continue;  // header row, discard
            }
        }
        if (fields.size() < 3) {
            continue;
        }
        offsets.push_back({fields[0], fields[1], fields[2]});
    }

    if (offsets.empty()) {
        std::cout << path << " had no usable rows -- using built-in default offsets ("
                  << DEFAULT_OFFSETS.size() << " points).\n";
        return DEFAULT_OFFSETS;
    }
    return offsets;
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [offsets_csv]\n"
              << "  offsets_csv: dx,dy,dz per row (meters). Optional header row auto-skipped.\n"
              << "  Default (if omitted or not found): " << DEFAULT_INPUT_CSV
              << ", falling back to " << DEFAULT_OFFSETS.size() << " built-in offsets.\n";
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
    auto node = std::make_shared<rclcpp::Node>("constrained_pose_ik_test");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        std::vector<std::array<double, 3>> offsets = loadOffsets(inputCsv);
        std::cout << "Loaded " << offsets.size() << " offsets.\n";

        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);
        moveGroup.setNumPlanningAttempts(NUM_PLANNING_ATTEMPTS);

        std::cout << "\nReplicating the ORIGINAL constrained setPoseTarget() scenario "
                     "(position + current orientation both held, OMPL random-seeded goal "
                     "sampling) -- NOT the seeded setJointValueTarget() fix. Running fully "
                     "automatically.\n";

        int succeeded = 0;
        double totalSeconds = 0.0;
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            const auto& offset = offsets[i];
            std::cout << "\n=== Point " << (i + 1) << "/" << offsets.size() << " === (offset dx="
                      << offset[0] << ", dy=" << offset[1] << ", dz=" << offset[2] << " m)\n";

            geometry_msgs::msg::PoseStamped currentPose = moveGroup.getCurrentPose();
            geometry_msgs::msg::PoseStamped targetPose = currentPose;
            targetPose.pose.position.x += offset[0];
            targetPose.pose.position.y += offset[1];
            targetPose.pose.position.z += offset[2];
            // Orientation deliberately left unchanged (copied from currentPose) --
            // this is the harder constraint, matching the original pre-fix behavior.

            moveGroup.setPoseTarget(targetPose);

            std::cout << "  Planning and executing (up to " << PLANNING_TIME_SECONDS
                      << "s budget)...\n";
            const auto start = std::chrono::steady_clock::now();
            auto result = moveGroup.move();
            const auto end = std::chrono::steady_clock::now();
            const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
            totalSeconds += elapsedSeconds;

            if (result == moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  SUCCESS in " << elapsedSeconds << "s.\n";
                ++succeeded;
            } else {
                std::cout << "  FAILED (error code " << result.val << ") after " << elapsedSeconds
                          << "s -- continuing to next point.\n";
            }
        }

        std::cout << "\n=== Summary ===\n"
                  << succeeded << "/" << offsets.size() << " points succeeded.\n"
                  << "Total wall-clock time: " << totalSeconds << "s\n"
                  << "Average time per point: " << (totalSeconds / static_cast<double>(offsets.size()))
                  << "s\n";
        if (succeeded < static_cast<int>(offsets.size())) {
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
