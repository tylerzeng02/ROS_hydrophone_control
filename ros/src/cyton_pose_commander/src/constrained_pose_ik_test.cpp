// constrained_pose_ik_test: deliberately replicates the harder-than-
// necessary IK scenario move_between_points.cpp used before switching to
// seeded setJointValueTarget(). setPoseTarget() with orientation held
// fixed hands a full 6-DOF Cartesian constraint to OMPL's random-seeded
// goal sampling. That is what produced real 30-second RRTConnect timeouts
// against elbow_yaw's razor-thin locked window.
//
// Purpose: isolate whether a different IK plugin (TRAC-IK vs. KDL) would
// handle this harder, random-seeded case any better. Run once against
// ik_solver:=kdl and once against ik_solver:=trac_ik and compare success
// rate and timing directly.
//
// No NDI dependency; pure MoveIt planning/execution stress test. Reads
// a small Cartesian offset (meters, position-only) per CSV line; for each,
// reads the arm's live current pose, builds an absolute target = current
// position + offset (orientation held at whatever it currently is), calls
// setPoseTarget() + move(), and reports success/failure and timing.
// Continues on failure rather than aborting the run.

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
// this project (move_x_test.cpp, the original move_between_points.cpp),
// kept identical here so a KDL run and a TRAC-IK run are compared under
// the same budget, not an apples-to-oranges one.
constexpr double PLANNING_TIME_SECONDS = 30.0;
constexpr unsigned int NUM_PLANNING_ATTEMPTS = 15;

// Built-in default offsets (meters), used if no CSV is given or found.
// Deliberately spans a similar small-to-moderate magnitude range as the
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
                break;  // non-numeric row (e.g. header); skip
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
            // Orientation deliberately left unchanged (copied from currentPose).
            // This is the harder constraint, matching the original pre-fix behavior.

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
